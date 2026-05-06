"""
Generate text with STAR-LDM.

Batch mode:
    python scripts/generate.py --model_path PATH \\
        --prompts "The movie was" "Once upon a time"

Interactive mode:
    python scripts/generate.py --model_path PATH --interactive

Classifier-guided generation:
    python scripts/generate.py --model_path PATH \\
        --classifier_path PATH --cls_guidance 3.0 --cls_target 1.0 \\
        --prompts "The movie was"
"""

import argparse

from omegaconf import OmegaConf

from star_ldm.interface import TransfusionGPTInterface


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None,
                            help='Optional YAML config; CLI flags override it')
    pre_args, remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description='Generate text with STAR-LDM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        parents=[pre_parser],
    )

    # Model
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to STAR-LDM checkpoint directory or .pt file')
    parser.add_argument('--device', type=str, default='cuda')

    # Mode
    parser.add_argument('--prompts', type=str, nargs='+',
                        help='One or more prompts for batch generation')
    parser.add_argument('--interactive', action='store_true',
                        help='Enter interactive REPL mode')

    # Classifier guidance
    parser.add_argument('--classifier_path', type=str, default=None,
                        help='Path to a pretrained classifier checkpoint for guided generation')
    parser.add_argument('--cls_guidance', type=float, default=0.0,
                        help='Classifier guidance scale (0 = disabled)')
    parser.add_argument('--cls_target', type=float, default=None,
                        help='Target class for guidance (0.0 or 1.0)')

    # Selector guidance
    parser.add_argument('--selector_path', type=str, default=None,
                        help='Path to a pretrained selector checkpoint')
    parser.add_argument('--selector_guidance', type=float, default=0.0,
                        help='Selector guidance scale (0 = disabled)')
    parser.add_argument('--selector_guidance_start', type=float, default=0.0)
    parser.add_argument('--selector_guidance_end', type=float, default=1.0)
    parser.add_argument('--guidance_clip', type=float, default=1.0)
    parser.add_argument('--normalize_guidance_grad', action='store_true')
    parser.add_argument('--num_plan_branches', type=int, default=1)
    parser.add_argument('--repulsion_scale', type=float, default=0.0)
    parser.add_argument('--repulsion_metric', type=str, default='cosine',
                        choices=['cosine', 'l2'])
    parser.add_argument('--repulsion_start', type=float, default=0.0)
    parser.add_argument('--repulsion_end', type=float, default=0.7)
    parser.add_argument('--quality_weighted_repulsion', action='store_true')
    parser.add_argument('--select_best_plan', action='store_true')
    parser.add_argument('--save_plan_stats', type=str, default=None,
                        help='Optional JSONL path for branch diagnostics')

    # Sampling
    parser.add_argument('--sampling_timesteps', type=int, default=50,
                        help='Number of diffusion sampling steps')
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm', 'ddim'])
    parser.add_argument('--cls_free_guidance', type=float, default=1.0,
                        help='Classifier-free guidance scale')

    # Token generation
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)

    if pre_args.config is not None:
        cfg = OmegaConf.load(pre_args.config)
        parser.set_defaults(**OmegaConf.to_container(cfg, resolve=True))

    args = parser.parse_args(remaining)
    if args.model_path is None:
        parser.error('--model_path is required')
    if not args.prompts and not args.interactive:
        parser.error('Provide --prompts or --interactive')
    return args


def main():
    args = parse_args()

    print('Loading model...')
    interface = TransfusionGPTInterface(
        model_path=args.model_path,
        device=args.device,
        classifier_path=args.classifier_path,
        selector_path=args.selector_path,
    )

    sample_kwargs = dict(
        sampling_timesteps=args.sampling_timesteps,
        sampler=args.sampler,
        cls_free_guidance=args.cls_free_guidance,
        cls_guidance=args.cls_guidance,
        cls_target=args.cls_target,
        selector_guidance=args.selector_guidance,
        selector_guidance_start=args.selector_guidance_start,
        selector_guidance_end=args.selector_guidance_end,
        guidance_clip=args.guidance_clip,
        normalize_guidance_grad=args.normalize_guidance_grad,
        num_plan_branches=args.num_plan_branches,
        repulsion_scale=args.repulsion_scale,
        repulsion_metric=args.repulsion_metric,
        repulsion_start=args.repulsion_start,
        repulsion_end=args.repulsion_end,
        quality_weighted_repulsion=args.quality_weighted_repulsion,
        select_best_plan=args.select_best_plan,
        save_plan_stats=args.save_plan_stats,
        generate_kwargs=dict(
            do_sample=True,
            num_beams=1,
            pad_token_id=interface.tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ),
    )

    if args.interactive:
        print('\nSTAR-LDM Interactive Demo')
        if args.classifier_path and args.cls_guidance != 0:
            print(f'  Classifier guidance: scale={args.cls_guidance}, target={args.cls_target}')
        if args.selector_path:
            print(f'  Selector: {args.selector_path}')
        print(f'  Diffusion steps: {args.sampling_timesteps} ({args.sampler})')
        print('  Type "quit" to exit.\n')

        while True:
            try:
                prompt = input('Prompt> ')
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip().lower() == 'quit':
                break
            if not prompt.strip():
                continue

            result = interface.generate([prompt], **sample_kwargs)
            generations = result.decoded[0]
            if len(generations) == 1:
                print(f'\n{generations[0]}\n')
            else:
                for idx, generation in enumerate(generations):
                    score = None
                    if result.selector_scores is not None:
                        score = result.selector_scores[0][idx]
                    score_text = f' score={score:.4f}' if score is not None else ''
                    print(f'\nBranch {idx}{score_text}: {generation}')
                print()
    else:
        result = interface.generate(args.prompts, **sample_kwargs)
        for prompt, generations, prompt_idx in zip(args.prompts, result.decoded, range(len(args.prompts))):
            print(f'\nPrompt: {prompt}')
            if len(generations) == 1:
                print(f'Generation: {generations[0]}')
            else:
                for branch_idx, generation in enumerate(generations):
                    score = None
                    if result.selector_scores is not None:
                        score = result.selector_scores[prompt_idx][branch_idx]
                    score_text = f' score={score:.4f}' if score is not None else ''
                    print(f'Branch {branch_idx}{score_text}: {generation}')
        if result.plan_stats_path is not None:
            print(f'\nPlan stats: {result.plan_stats_path}')


if __name__ == '__main__':
    main()
