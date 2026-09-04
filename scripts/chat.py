#!/usr/bin/env python3
"""
chat.py -- a minimal REPL for chatting with Llama-2-13b-chat in its original
fp16 precision. No probes, no steering -- just the base chat model, for when
you want to talk to it directly rather than through the TalkTuner dashboard
(see TalkTuner-chatbot-llm-dashboard/scripts/cli.py for that).

SETUP
  conda activate talktuner-gpu     # has torch/transformers/accelerate pinned
  huggingface-cli login            # Llama-2 is gated; request access first

USAGE
  python scripts/chat.py
  python scripts/chat.py --model NousResearch/Llama-2-13b-chat-hf --temperature 0.7

REPL COMMANDS
  /system <text>   set the system prompt and clear history
  /reset           clear conversation history (keeps system prompt)
  /save <name>     save the conversation to data/manual.conversations/<name>
  /quit            exit
"""

import argparse
import datetime
import json
import os
import sys
import textwrap

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(REPO_ROOT, "data", "manual.conversations")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe. Please ensure that your responses "
    "are socially unbiased and positive in nature. If a question does not make "
    "any sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don't know the answer to a question, please "
    "don't share false information."
)


def llama_v2_prompt(messages, system_prompt):
    """Ported from src/dataset.py:llama_v2_prompt so the model sees the exact
    prompt format it was fine-tuned on."""
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"

    msgs = [{"role": "system", "content": system_prompt}] + list(messages)
    msgs = [{
        "role": msgs[1]["role"],
        "content": B_SYS + msgs[0]["content"] + E_SYS + msgs[1]["content"],
    }] + msgs[2:]

    out = [
        f"{BOS}{B_INST} {p['content'].strip()} {E_INST} {a['content'].strip()} {EOS}"
        for p, a in zip(msgs[::2], msgs[1::2])
    ]
    if msgs[-1]["role"] == "user":
        out.append(f"{BOS}{B_INST} {msgs[-1]['content'].strip()} {E_INST}")
    return "".join(out)


def wrap(s, indent="  "):
    return "\n".join(
        textwrap.fill(line, 88, initial_indent=indent, subsequent_indent=indent) or indent
        for line in s.split("\n")
    )


def save_conversation(name, system_prompt, messages):
    if not name:
        print("  usage: /save <filename>")
        return
    if not messages:
        print("  nothing to save yet")
        return
    if os.path.basename(name) != name:
        print("  filename must not contain path separators")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, name if os.path.splitext(name)[1] else name + ".json")
    record = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "system_prompt": system_prompt,
        "messages": messages,
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  saved conversation to {os.path.relpath(path, REPO_ROOT)}")


@torch.no_grad()
def generate(model, tokenizer, messages, system_prompt, device, args):
    text = llama_v2_prompt(messages, system_prompt)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=3500)
    enc = {k: v.to(device) for k, v in enc.items()}
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if args.greedy:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)
    else:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    out = model.generate(**enc, **gen_kwargs)
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description="Chat with Llama-2-13b-chat in fp16.")
    ap.add_argument("--model", default="NousResearch/Llama-2-13b-chat-hf")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--greedy", action="store_true", help="deterministic decoding instead of sampling")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no CUDA device visible; fp16 on CPU will be extremely slow.")

    print(f"[..] loading {args.model} in fp16 (this takes a few minutes on first run)")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print("[ok] loaded. /help for commands, /quit to exit\n")

    system_prompt = args.system_prompt
    messages = []

    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0]
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/help":
                print("  /system <text>   set system prompt and clear history\n"
                      "  /reset           clear conversation history\n"
                      "  /save <name>     save conversation to data/manual.conversations/<name>\n"
                      "  /quit            exit")
            elif cmd == "/reset":
                messages = []
                print("  conversation cleared")
            elif cmd == "/system":
                if len(parts) < 2:
                    print("  usage: /system <text>")
                else:
                    system_prompt = parts[1]
                    messages = []
                    print("  system prompt updated, conversation cleared")
            elif cmd == "/save":
                name = parts[1].strip() if len(parts) > 1 else None
                save_conversation(name, system_prompt, messages)
            else:
                print("  unrecognized; /help")
            continue

        messages.append({"role": "user", "content": line})
        ans = generate(model, tokenizer, messages, system_prompt, device, args)
        print(f"\nbot >\n{wrap(ans)}\n")
        messages.append({"role": "assistant", "content": ans})


if __name__ == "__main__":
    main()
