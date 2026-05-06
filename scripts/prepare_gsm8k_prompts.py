"""
Prepare GSM8K prompts for selector data collection.
"""

import argparse
import json
import os

from star_ldm.verification.gsm8k import extract_gsm8k_answer


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare GSM8K JSONL prompts')
    parser.add_argument('--gsm8k_path', type=str, default=None,
                        help='Local GSM8K dataset saved with save_to_disk()')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def load_gsm8k(path):
    if path:
        from datasets import load_from_disk
        return load_from_disk(path)

    from datasets import load_dataset
    return load_dataset('openai/gsm8k', 'main')


def format_prompt(question):
    return f'Question: {question.strip()}\nAnswer:'


def main():
    args = parse_args()
    dataset = load_gsm8k(args.gsm8k_path)
    split = dataset[args.split]
    if args.limit is not None:
        split = split.select(range(min(args.limit, len(split))))

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, 'w') as f:
        for idx, row in enumerate(split):
            gold = extract_gsm8k_answer(row['answer'])
            record = {
                'prompt_id': idx,
                'prompt': format_prompt(row['question']),
                'gold': gold,
                'question': row['question'],
                'answer': row['answer'],
                'split': args.split,
            }
            f.write(json.dumps(record) + '\n')

    print(f'Wrote {len(split):,} GSM8K {args.split} prompts to {args.output_path}')


if __name__ == '__main__':
    main()
