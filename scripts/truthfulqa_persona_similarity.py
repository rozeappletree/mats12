#!/usr/bin/env python3
"""
Compare each persona one-liner against TruthfulQA's answer options using
sentence-embedding cosine similarity, and write the closest-matching answer,
its score, and whether that answer is correct back onto each record. Every
persona field found on the records (any key besides the base TruthfulQA
fields) is compared against the full pool of correct_answers (+ best_answer)
and incorrect_answers for that question.

Model: Qwen/Qwen3-Embedding-8B -- an 8B-parameter embedding model, among the
top performers on the MTEB multilingual/STS leaderboards, comfortably fits on
this GPU (~46GB). Needs transformers>=4.51 (for the Qwen3 architecture), so
this runs in its own "embed" conda env, separate from "talktuner-gpu" which
has an older pinned transformers used by other scripts.

USAGE
  conda run -n embed python scripts/truthfulqa_persona_similarity.py --input <file> --output <file>
"""

import argparse
import json
import os

import torch
from sentence_transformers import SentenceTransformer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(
    REPO_ROOT, "data", "truthfulqa", "truthful_qa.personas.oneliner.json"
)
DEFAULT_OUTPUT = os.path.join(
    REPO_ROOT, "data", "truthfulqa", "truthful_qa.personas.oneliner.similarity.json"
)

BASE_FIELDS = {
    "type",
    "category",
    "question",
    "best_answer",
    "correct_answers",
    "incorrect_answers",
    "source",
}


def output_prefix(field):
    return field[len("SystemPrompt") :] if field.startswith("SystemPrompt") else field


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)

    persona_fields = [k for k in records[0] if k not in BASE_FIELDS]
    fields = {field: output_prefix(field) for field in persona_fields}
    print(f"Persona fields: {list(fields)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        args.model,
        device=device,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    def encode(texts):
        return model.encode(
            texts,
            batch_size=args.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    # Build per-record candidate pools: correct_answers gets best_answer appended.
    correct_lists = [r["correct_answers"] + [r["best_answer"]] for r in records]
    incorrect_lists = [r["incorrect_answers"] for r in records]
    pool_lists = [c + i for c, i in zip(correct_lists, incorrect_lists)]

    # Flatten all candidate answers into one big encode call, tracking offsets.
    flat_texts = []
    offsets = []  # (start, num_correct, num_total) per record
    for c, i in zip(correct_lists, incorrect_lists):
        start = len(flat_texts)
        flat_texts.extend(c)
        flat_texts.extend(i)
        offsets.append((start, len(c), len(c) + len(i)))

    candidate_emb = encode(flat_texts)

    persona_emb = {field: encode([r.get(field, "") or "" for r in records]) for field in fields}

    for idx, r in enumerate(records):
        start, n_correct, n_total = offsets[idx]
        pool_emb = candidate_emb[start : start + n_total]
        pool_texts = pool_lists[idx]

        for field, prefix in fields.items():
            emb = persona_emb[field][idx]
            sims = pool_emb @ emb
            best_idx = int(torch.argmax(sims).item())
            r[f"{prefix}BestAnswer"] = pool_texts[best_idx]
            r[f"{prefix}BestScore"] = float(sims[best_idx].item())
            r[f"{prefix}BestIsCorrect"] = best_idx < n_correct

    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
