#!/usr/bin/env python3
"""
truthfulqa_generate.py -- generate NousResearch/Llama-2-13b-chat-hf answers to
the TruthfulQA questions produced by truthfulqa_download.py, and add them back
onto each record as a new "output" field.

SETUP
  conda activate talktuner-gpu     # has torch/transformers/accelerate pinned
  huggingface-cli login            # Llama-2 is gated; request access first

USAGE
  python scripts/truthfulqa_generate.py
  python scripts/truthfulqa_generate.py --sample-size 50 --seed 0
  python scripts/truthfulqa_generate.py --model NousResearch/Llama-2-13b-chat-hf \
      --batch-size 8 --max-new-tokens 256
"""

import argparse
import json
import os
import random

import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(REPO_ROOT, "data", "truthfulqa", "truthful_qa.json")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "data", "truthfulqa", "truthful_qa.generated.json")


def llama_v2_prompt(question, system_prompt):
    """Ported from scripts/chat.py:llama_v2_prompt (single-turn: one user message,
    no assistant history) so the model sees the exact prompt format it was
    fine-tuned on."""
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS = "<s>"

    content = question.strip()
    if system_prompt:
        content = B_SYS + system_prompt + E_SYS + content
    return f"{BOS}{B_INST} {content} {E_INST}"


@torch.no_grad()
def generate_batch(model, tokenizer, questions, system_prompt, device, args):
    prompts = [llama_v2_prompt(q, system_prompt) for q in questions]
    enc = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=2048, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if args.greedy:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)
    else:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    out = model.generate(**enc, **gen_kwargs)
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [t.strip() for t in texts]


def main():
    ap = argparse.ArgumentParser(description="Generate Llama-2-13b-chat answers for TruthfulQA.")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--model", default="NousResearch/Llama-2-13b-chat-hf")
    ap.add_argument("--system-prompt", default="",
                     help="empty by default so the model is probed directly, the way the "
                          "original TruthfulQA generation task does")
    ap.add_argument("--sample-size", type=int, default=None,
                     help="only generate for a random subset of N questions (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--greedy", action="store_true", default=True,
                     help="deterministic decoding (default: on, for reproducible eval)")
    ap.add_argument("--sample", dest="greedy", action="store_false",
                     help="use temperature/top-p sampling instead of greedy decoding")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--save-every", type=int, default=20,
                     help="checkpoint the output file every N examples")
    args = ap.parse_args()

    with open(args.input) as f:
        records = json.load(f)

    if args.sample_size is not None and args.sample_size < len(records):
        rng = random.Random(args.seed)
        records = rng.sample(records, args.sample_size)
        print(f"[..] sampled {len(records)} of the full dataset (seed={args.seed})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no CUDA device visible; fp16 on CPU will be extremely slow.")

    print(f"[..] loading {args.model} in fp16 (this takes a few minutes on first run)")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    print("[ok] loaded")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for start in tqdm(range(0, len(records), args.batch_size), desc="generating"):
        batch = records[start:start + args.batch_size]
        questions = [r["question"] for r in batch]
        answers = generate_batch(model, tokenizer, questions, args.system_prompt, device, args)
        for r, a in zip(batch, answers):
            r["output"] = a

        done = start + len(batch)
        if done % args.save_every < args.batch_size or done == len(records):
            with open(args.output, "w") as f:
                json.dump(records, f, indent=2)

    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[ok] wrote {len(records)} examples with generated 'output' to {os.path.relpath(args.output, REPO_ROOT)}")


if __name__ == "__main__":
    main()
