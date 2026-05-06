"""
Collect supervised selector training data from a frozen STAR-LDM checkpoint.
"""

import argparse
import json
import os
import random

import torch
from tqdm import tqdm

from star_ldm.diffusion.noise_schedule import get_scaled_noise_schedule
from star_ldm.interface import TransfusionGPTInterface
from star_ldm.models.transfusion import variance_preserving_map
from star_ldm.verification import get_verifier


def parse_args():
    parser = argparse.ArgumentParser(description='Collect selector training data')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--prompts_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--num_samples_per_prompt', type=int, default=16)
    parser.add_argument('--verifier', type=str, default='exact_match')
    parser.add_argument('--save_noisy_timesteps', action='store_true')
    parser.add_argument('--noisy_per_clean', type=int, default=4)
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm', 'ddim'])
    parser.add_argument('--cosine_scale', type=float, default=3.0)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_id_to_long(prompt_id, fallback):
    if prompt_id is None:
        return fallback
    try:
        return int(prompt_id)
    except (TypeError, ValueError):
        return fallback


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    verifier = get_verifier(args.verifier)
    rows = read_jsonl(args.prompts_path)
    interface = TransfusionGPTInterface(args.model_path, device=args.device)
    model = interface.model
    tokenizer = interface.tokenizer

    prompt_ids = []
    prompt_embs = []
    z0_internal = []
    labels = []
    meta = []

    generate_kwargs = dict(
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
    )

    for row_idx, row in enumerate(tqdm(rows, desc='Collecting selector data')):
        prompt = row['prompt']
        gold = row.get('gold')
        prompt_id = row.get('prompt_id', row_idx)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(interface.device)
        prompt_emb = model.get_sentence_embedding(prompt).to(interface.device).float()

        for sample_idx in range(args.num_samples_per_prompt):
            sample_output = model.sample(
                input_ids,
                sampling_timesteps=args.sampling_timesteps,
                sampler=args.sampler,
                cosine_scale=args.cosine_scale,
                generate_kwargs=generate_kwargs,
                return_internal_latents=True,
            )
            decoded = sample_output.generations[0]
            correct = verifier.verify(prompt, decoded, gold)

            prompt_ids.append(prompt_id_to_long(prompt_id, row_idx))
            prompt_embs.append(prompt_emb.squeeze(0).detach().cpu())
            z0_internal.append(sample_output.x_start_internal[0].detach().cpu())
            labels.append(float(correct))
            meta.append({
                'prompt_id': prompt_id,
                'sample_idx': sample_idx,
                'prompt': prompt,
                'decoded': decoded,
                'gold': gold,
            })

    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
    prompt_embs = torch.stack(prompt_embs).float()
    z0_internal = torch.stack(z0_internal).float()
    labels = torch.tensor(labels, dtype=torch.float32)

    zt = None
    alpha2 = None
    if args.save_noisy_timesteps:
        schedule = get_scaled_noise_schedule('cosine', scale=args.cosine_scale)
        t = torch.rand((z0_internal.shape[0], args.noisy_per_clean), dtype=z0_internal.dtype)
        alpha2 = schedule(t).float()
        z0_flat = z0_internal.unsqueeze(1).expand(-1, args.noisy_per_clean, -1).reshape(-1, z0_internal.shape[-1])
        alpha2_flat = alpha2.reshape(-1, 1)
        eps = torch.randn_like(z0_flat)
        zt = variance_preserving_map(z0_flat, alpha2_flat, eps=eps)
        zt = zt.view(z0_internal.shape[0], args.noisy_per_clean, z0_internal.shape[-1]).float()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save({
        'prompt_ids': prompt_ids,
        'prompt_embs': prompt_embs,
        'z0_internal': z0_internal,
        'zt': zt,
        'alpha2': alpha2,
        'labels': labels,
        'meta': meta,
    }, args.output_path)


if __name__ == '__main__':
    main()
