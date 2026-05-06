"""
Download all assets needed for offline training on Zaratan (or any no-internet node).
Run this ONCE on a login node (which has internet access) before submitting jobs.

Usage:
    python scripts/download_assets.py \
        --hf_home /scratch/user/.hf_cache \
        --fineweb_path /scratch/user/datasets/fineweb-10BT \
        --c4_path /scratch/user/datasets/c4-validation \
        --fineweb_subset sample-10BT \
        --lm_name gpt2-large \
        --sentence_encoder sentence-transformers/sentence-t5-xl

After this completes, submit jobs with:
    HF_HOME=/scratch/user/.hf_cache \
    FINEWEB_LOCAL_PATH=/scratch/user/datasets/fineweb-10BT \
    C4_LOCAL_PATH=/scratch/user/datasets/c4-validation \
      zaratan/submit_train.sh
"""

import sys
import os

# On HPC clusters (e.g. Zaratan), PYTHONPATH can inject system site-packages
# before the venv's, causing version mismatches (e.g. old pyarrow lacking json_()).
# Force the venv's site-packages to the front of the path.
if sys.prefix != sys.base_prefix:
    venv_site = os.path.join(
        sys.prefix, 'lib',
        f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages',
    )
    sys.path = [venv_site] + [p for p in sys.path if p != venv_site]

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Download assets for offline use')
    parser.add_argument('--hf_home', required=True,
                        help='Directory to use as HF_HOME (model cache)')
    parser.add_argument('--fineweb_path', required=True,
                        help='Path to save the FineWeb dataset')
    parser.add_argument('--c4_path', required=True,
                        help='Path to save the C4 validation dataset')
    parser.add_argument('--fineweb_subset', default='sample-10BT',
                        choices=['sample-10BT', 'sample-100BT'],
                        help='FineWeb subset to download')
    parser.add_argument('--lm_name', default='gpt2-large',
                        help='Language model name (must match train config)')
    parser.add_argument('--sentence_encoder',
                        default='sentence-transformers/sentence-t5-xl',
                        help='Sentence encoder model name')
    parser.add_argument('--c4_num_examples', type=int, default=5000,
                        help='Number of C4 validation examples to save')
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ['HF_HOME'] = args.hf_home
    os.makedirs(args.hf_home, exist_ok=True)

    # --- Language model + tokenizer ---
    print(f'\n[1/4] Downloading {args.lm_name} model and tokenizer -> {args.hf_home}')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained(args.lm_name)
    AutoModelForCausalLM.from_pretrained(args.lm_name)
    print('      Done.')

    # --- Sentence encoder ---
    print(f'\n[2/4] Downloading sentence encoder {args.sentence_encoder} -> {args.hf_home}')
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(args.sentence_encoder)
    print('      Done.')

    # --- FineWeb training dataset ---
    print(f'\n[3/4] Downloading FineWeb ({args.fineweb_subset}) -> {args.fineweb_path}')
    print('      This may take a long time and significant disk space.')
    from datasets import load_dataset
    fineweb = load_dataset(
        'HuggingFaceFW/fineweb',
        name=args.fineweb_subset,
        split='train',
        streaming=False,
    )
    fineweb.save_to_disk(args.fineweb_path)
    print(f'      Saved {len(fineweb):,} examples.')

    # --- C4 validation dataset ---
    print(f'\n[4/4] Downloading C4 validation ({args.c4_num_examples} examples) -> {args.c4_path}')
    c4 = load_dataset(
        'allenai/c4', 'en', split='validation', streaming=True,
    ).shuffle(seed=42, buffer_size=50_000).take(args.c4_num_examples)
    # Materialize the iterable dataset before saving
    from datasets import Dataset
    c4_materialized = Dataset.from_list(list(c4))
    c4_materialized.save_to_disk(args.c4_path)
    print(f'      Saved {len(c4_materialized):,} examples.')

    print('\nAll assets downloaded. Set the following env vars when submitting jobs:')
    print(f'  HF_HOME={args.hf_home}')
    print(f'  FINEWEB_LOCAL_PATH={args.fineweb_path}')
    print(f'  C4_LOCAL_PATH={args.c4_path}')


if __name__ == '__main__':
    main()
