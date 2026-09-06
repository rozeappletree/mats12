#!/usr/bin/env python3
"""Plot the accuracy-by-condition bar chart from an already-scored
{attribute}_scored.json (written by score_truthfulqa_responses.py, which runs
in the "embed" conda env and has no matplotlib). Run this in an env that has
matplotlib instead, e.g. talktuner-gpu:

  python nb/causality_tests/plot_scored_accuracy.py --attribute gullibility
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_ROOT, "intervention_results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute", required=True)
    args = parser.parse_args()

    out_dir = os.path.join(RESULTS_DIR, args.attribute)
    with open(os.path.join(out_dir, f"{args.attribute}_scored.json")) as f:
        payload = json.load(f)
    accuracy = payload["accuracy_by_condition"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    conds = list(accuracy.keys())
    vals = [accuracy[c] for c in conds]
    ax.bar(conds, vals)
    ax.set_ylabel("fraction judged correct (embedding-similarity match)")
    ax.set_title(f"{args.attribute}: TruthfulQA correctness by steering condition")
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, f"{args.attribute}_accuracy_by_condition.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
