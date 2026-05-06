"""
Evaluate STAR-LDM end to end on GSM8K, with and without selector guidance.
"""

import argparse
import json
import os
import random

from scripts.prepare_gsm8k_prompts import format_prompt, load_gsm8k
from star_ldm.verification.gsm8k import extract_gsm8k_answer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate STAR-LDM on GSM8K')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--selector_path', type=str, required=True)
    parser.add_argument('--gsm8k_path', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm', 'ddim'])
    parser.add_argument('--cls_free_guidance', type=float, default=1.0)
    parser.add_argument('--selector_guidance', type=float, default=1.0)
    parser.add_argument('--num_plan_branches', type=int, default=8)
    parser.add_argument('--repulsion_scale', type=float, default=0.0)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def is_correct(decoded, gold):
    decoded_answer = extract_gsm8k_answer(decoded)
    gold_answer = extract_gsm8k_answer(gold)
    return decoded_answer is not None and gold_answer is not None and decoded_answer == gold_answer


def generate_one(interface, prompt, args, *, use_selector):
    generate_kwargs = dict(
        do_sample=args.do_sample,
        num_beams=1,
        pad_token_id=interface.tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    if args.do_sample:
        generate_kwargs['top_p'] = args.top_p
    result = interface.generate(
        [prompt],
        sampling_timesteps=args.sampling_timesteps,
        sampler=args.sampler,
        cls_free_guidance=args.cls_free_guidance,
        selector_guidance=args.selector_guidance if use_selector else 0.0,
        num_plan_branches=args.num_plan_branches if use_selector else 1,
        repulsion_scale=args.repulsion_scale if use_selector else 0.0,
        select_best_plan=use_selector,
        generate_kwargs=generate_kwargs,
    )
    return result


def main():
    args = parse_args()
    from star_ldm.interface import TransfusionGPTInterface
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = load_gsm8k(args.gsm8k_path)
    split = dataset[args.split]
    if args.limit is not None:
        split = split.select(range(min(args.limit, len(split))))

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    interface = TransfusionGPTInterface(
        model_path=args.model_path,
        device=args.device,
        selector_path=args.selector_path,
    )

    baseline_correct = 0
    selector_correct = 0
    count = 0

    with open(args.output_path, 'w') as f:
        for idx, row in enumerate(split):
            prompt = format_prompt(row['question'])
            gold = extract_gsm8k_answer(row['answer'])

            baseline_result = generate_one(interface, prompt, args, use_selector=False)
            baseline_decoded = baseline_result.decoded[0][0]
            baseline_extracted = extract_gsm8k_answer(baseline_decoded)
            baseline_ok = is_correct(baseline_decoded, gold)

            selector_result = generate_one(interface, prompt, args, use_selector=True)
            selector_decoded = selector_result.decoded[0][0]
            selector_extracted = extract_gsm8k_answer(selector_decoded)
            selector_ok = is_correct(selector_decoded, gold)

            count += 1
            baseline_correct += int(baseline_ok)
            selector_correct += int(selector_ok)

            record = {
                'prompt_id': idx,
                'split': args.split,
                'question': row['question'],
                'gold': gold,
                'baseline': {
                    'decoded': baseline_decoded,
                    'extracted': baseline_extracted,
                    'correct': baseline_ok,
                },
                'selector': {
                    'decoded': selector_decoded,
                    'extracted': selector_extracted,
                    'correct': selector_ok,
                    'scores': selector_result.selector_scores[0] if selector_result.selector_scores else None,
                    'selected_branch': selector_result.selected_branch[0] if selector_result.selected_branch else None,
                },
            }
            f.write(json.dumps(record) + '\n')
            f.flush()

            print(
                f'[{count}/{len(split)}] '
                f'baseline={baseline_correct / count:.4f} '
                f'selector={selector_correct / count:.4f}'
            )

        summary = {
            'num_examples': count,
            'baseline_accuracy': baseline_correct / max(1, count),
            'selector_accuracy': selector_correct / max(1, count),
            'settings': vars(args),
        }
        f.write(json.dumps({'summary': summary}) + '\n')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
