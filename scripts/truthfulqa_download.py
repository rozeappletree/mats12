#!/usr/bin/env python3
"""
truthfulqa_download.py -- download the TruthfulQA "generation" split and dump
it to plain JSON so the rest of the pipeline (truthfulqa_generate.py,
truthfulqa_score.py) doesn't depend on the `datasets` library at all.

SETUP
  conda activate talktuner-gpu     # or any env with `datasets` installed

USAGE
  python scripts/truthfulqa_download.py
  python scripts/truthfulqa_download.py --output data/truthfulqa/truthful_qa.json
"""

import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "data", "truthfulqa", "truthful_qa.json")


def main():
    ap = argparse.ArgumentParser(description="Download truthfulqa/truthful_qa as JSON.")
    ap.add_argument("--dataset", default="truthfulqa/truthful_qa")
    ap.add_argument("--config", default="generation", choices=["generation", "multiple_choice"])
    ap.add_argument("--split", default="validation", help="TruthfulQA only ships a 'validation' split")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"[..] downloading {args.dataset} ({args.config}/{args.split})")
    ds = load_dataset(args.dataset, args.config, split=args.split)
    records = [dict(row) for row in ds]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    print(f"[ok] wrote {len(records)} examples to {os.path.relpath(args.output, REPO_ROOT)}")
    print(f"     fields: {sorted(records[0].keys())}")


if __name__ == "__main__":
    main()
