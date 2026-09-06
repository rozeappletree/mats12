"""
Synthetic dataset generation — Gullibility / Rationality / Seriousness / Certainty-Seeking

Converted from TalkTuner's llama2chat_dataset_generation.ipynb (Chen et al. 2024).

Differences from the original notebook, all deliberate (see docs/llama_dataset_synthesis.md):

  * A real StoppingCriteria for '<End of Conversation>', so generation halts early
    instead of always running to max_new_tokens.
  * Behavioural prompts for the four new attributes, written so that no prompt
    contains a word it would be leakage for the model to echo. The notebook's
    "you may or may not include the user's <attribute>" line is dropped.
  * Generated conversations are validated BEFORE being written to disk, and are
    rejected (and retried) if they are truncated or structurally malformed.
  * Retries are capped, so a persistent fault aborts instead of spinning forever.
  * Existing output files are skipped, so an interrupted run can be resumed.

Usage:
    export HF_TOKEN=hf_...
    python scripts/gen_llama_dataset.py --num_samples 500
    python scripts/gen_llama_dataset.py --num_samples 1000 --attributes gullibility rationality
"""

import argparse
import hashlib
import os
import re
import time

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Reuse TalkTuner's dataset.py helpers (split_conversation, llama_v2_prompt).
# Must be importable — copy dataset.py from the TalkTuner repo into this directory
# or onto your PYTHONPATH.
from dataset import llama_v2_prompt, split_conversation

MODEL_NAME = "NousResearch/Llama-2-13b-chat-hf"
SEED = 75241239
END_TAG = "<End of Conversation>"
START_TAG = "HUMAN: "

# A conversation must contain at least this many complete user/assistant turn
# pairs to be kept. One-exchange stubs carry very little attribute signal.
MIN_TURN_PAIRS = 2

# Abort a subcategory after this many consecutive rejected/failed generations,
# rather than retrying forever.
MAX_CONSECUTIVE_FAILURES = 25


class ConversationRejected(Exception):
    """A generation completed but failed a quality check. Retryable."""


# --------------------------------------------------------------------------- #
# Attribute definitions
# --------------------------------------------------------------------------- #
#
# Each level supplies:
#   persona  — a clause completing "This human user {persona}"
#   behavior — concrete observable behaviour, no trait labels
#
# `show_through` names the channel the trait should be expressed in.
#
# `leak_stems` are NOT sent to the model. Naming a word in the prompt in order
# to forbid it both contradicts the surrounding description and primes the model
# toward that word, so the prompts avoid these stems entirely and they are used
# only as a post-hoc leakage check against the generated user turns.

ATTRIBUTES = {
    "gullibility": {
        "levels": ["low", "medium", "high"],
        "persona": {
            "low": "rarely takes a claim at face value",
            "medium": "goes along with most of what they are told, with occasional pushback",
            "high": "takes what they are told at face value",
        },
        "behavior": {
            "low": (
                "they ask where information comes from, push back when something "
                "sounds unsupported, and want to see evidence before acting on advice"
            ),
            "medium": (
                "they generally act on the assistant's answers, but now and then ask "
                "a clarifying question or want one detail confirmed"
            ),
            "high": (
                "they accept the assistant's claims without asking where the "
                "information comes from, and may restate an unsupported claim later "
                "in the conversation as though it were established fact"
            ),
        },
        "show_through": "how the user reacts to the assistant's claims and advice",
        "leak_stems": ["gullib", "credul", "skeptic", "sceptic", "naive", "naïve", "trust"],
    },
    "rationality": {
        "levels": ["low", "medium", "high"],
        "persona": {
            "low": "jumps to conclusions and argues from feeling and anecdote",
            "medium": "thinks things through most of the time",
            "high": "weighs evidence carefully before settling on a view",
        },
        "behavior": {
            "low": (
                "they generalise from a single story, hold on to a position when "
                "given evidence against it, and may contradict something they said "
                "in an earlier turn"
            ),
            "medium": (
                "they usually weigh the options, but sometimes build on an "
                "assumption they never checked"
            ),
            "high": (
                "they ask what a claim is based on, name the trade-offs explicitly, "
                "and change their position when given a strong counterargument"
            ),
        },
        "show_through": "how the user frames requests and responds to the answers they get",
        "leak_stems": ["rational", "irrational", "logic", "reasonab"],
    },
    "seriousness": {
        "levels": ["low", "medium", "high"],
        "persona": {
            "low": "keeps things light and playful",
            "medium": "is mostly down to business, with the occasional aside",
            "high": "treats the exchange as consequential",
        },
        "behavior": {
            "low": (
                "they use puns and wordplay, pose absurd hypotheticals, write in "
                "casual slang, and are happy with a loose or imprecise answer"
            ),
            "medium": (
                "they ask practical questions and want usable answers, but will "
                "play along with a bit of levity"
            ),
            "high": (
                "they write in a measured, professional register, ask about matters "
                "with real stakes — health, legal, financial, or safety — and want "
                "precise, carefully qualified answers"
            ),
        },
        "show_through": "the user's tone, word choice, and the stakes of what they ask about",
        "leak_stems": ["serious", "silly", "jok", "formal"],
    },
    "certainty_seeking": {
        # NOTE: original attribute spec listed High/Neutral/Neutral, assumed to be
        # a typo for Low/Neutral/High to keep the same 3-way structure as the rest.
        "levels": ["low", "neutral", "high"],
        "persona": {
            "low": "is comfortable when a question has no single right answer",
            "neutral": "usually accepts a hedged answer, but likes a bottom line",
            "high": "wants one definite answer to everything",
        },
        "behavior": {
            "low": (
                'they accept "it depends" or a range of possible outcomes without '
                "pressing further, and are content to leave open questions open"
            ),
            "neutral": (
                "they are fine with caveats, though they will sometimes ask the "
                "assistant for its best guess"
            ),
            "high": (
                "they push back on caveats and hedging, ask again for one clear "
                "conclusion, and press for a firm verdict even on questions that "
                "genuinely do not have one"
            ),
        },
        "show_through": (
            "how the user reacts when the assistant gives a hedged or open-ended answer"
        ),
        "leak_stems": ["certain", "uncertain", "ambigu", "unsure"],
    },
}


# --------------------------------------------------------------------------- #
# Stopping criterion
# --------------------------------------------------------------------------- #

class StopOnSubstring(StoppingCriteria):
    """Stops generation once the decoded suffix contains the given substring.
    Re-decodes only the newly generated tail each step to keep this cheap."""

    def __init__(self, tokenizer, stop_string, prompt_len, check_every=8):
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.prompt_len = prompt_len
        self.check_every = check_every
        self._step = 0

    def __call__(self, input_ids, scores, **kwargs):
        self._step += 1
        if self._step % self.check_every != 0:
            return False
        # Only decode the generated portion, not the whole prompt, each check.
        tail_ids = input_ids[0][self.prompt_len :]
        tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
        return self.stop_string in tail_text


# --------------------------------------------------------------------------- #
# Text cleanup
# --------------------------------------------------------------------------- #

def normalize_line_whitespace(document: str) -> str:
    """Collapse runs of whitespace within each line and strip leading/trailing
    whitespace from each line, preserving line breaks.

    This is byte-for-byte what the original notebook's
    `remove_whitespaces_before_word(document, word)` did. That function took a
    `word` argument and claimed to strip whitespace before it, but the comparison
    was against a bare "HUMAN"/"ASSISTANT" while the text carries "HUMAN:" with a
    colon, so the branch never fired — and the `.rstrip()` it guarded was a no-op
    on `.split()` output anyway. Renamed to describe the actual behaviour; output
    is unchanged.
    """
    return "\n".join(" ".join(line.split()) for line in document.split("\n"))


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

def build_prompt(attribute_name: str, level: str) -> str:
    cfg = ATTRIBUTES[attribute_name]
    persona = cfg["persona"][level]
    behavior = cfg["behavior"][level]
    show_through = cfg["show_through"]
    return (
        "SYSTEM: Generate a conversation between a human user and an AI assistant. "
        f"This human user {persona}: {behavior}. "
        f"Convey this through {show_through} — the user should never name or "
        "describe this characteristic about themselves. "
        "Be creative on the topics of conversation. "
        "Make the conversation sound natural. "
        "The user's response should start with 'HUMAN:', and the AI assistant's "
        "response should start with 'ASSISTANT:'. "
        f"Mark the end of the generated conversation with '{END_TAG}'"
    )


def leaked_stems(user_msgs, attribute_name: str):
    """Return the leak stems that appear in the user's own turns."""
    cfg = ATTRIBUTES[attribute_name]
    text = " ".join(user_msgs).lower()
    return [s for s in cfg["leak_stems"] if re.search(r"\b" + re.escape(s), text)]


# --------------------------------------------------------------------------- #
# Parsing and validation
# --------------------------------------------------------------------------- #

def extract_conversation(output: str) -> str:
    """Slice the conversation out of a raw generation, or raise
    ConversationRejected if it is unusable."""
    conversation_start = output.find(START_TAG)
    if conversation_start == -1:
        raise ConversationRejected(f"no {START_TAG!r} tag in generated output")

    conversation = output[conversation_start:]

    conversation_end = conversation.find(END_TAG)
    if conversation_end == -1:
        # Generation hit max_new_tokens before emitting the end tag. Keeping it
        # would append a truncated half-turn to the dataset.
        raise ConversationRejected(
            f"no {END_TAG!r} tag — generation truncated at max_new_tokens"
        )
    conversation = conversation[:conversation_end]

    return normalize_line_whitespace(conversation.strip())


def validate_conversation(conversation: str, attribute_name: str):
    """Check turn structure. Returns (messages_dict, user_msgs); raises
    ConversationRejected if the conversation is malformed.

    Note the original notebook only rejected conversations that produced *zero*
    turn pairs — `llama_v2_prompt` raises IndexError on an empty list and nothing
    else. Unbalanced or single-exchange conversations passed silently, because
    `zip(user_msgs, ai_msgs)` truncates to the shorter of the two.
    """
    user_msgs, ai_msgs = split_conversation(conversation)

    if len(user_msgs) < MIN_TURN_PAIRS or len(ai_msgs) < MIN_TURN_PAIRS:
        raise ConversationRejected(
            f"only {len(user_msgs)} user / {len(ai_msgs)} assistant turns, "
            f"need >= {MIN_TURN_PAIRS} of each"
        )

    # Well-formed alternation leaves at most one dangling user turn.
    if not 0 <= len(user_msgs) - len(ai_msgs) <= 1:
        raise ConversationRejected(
            f"turns do not alternate: {len(user_msgs)} user vs "
            f"{len(ai_msgs)} assistant"
        )

    if any(not m.strip() for m in user_msgs + ai_msgs):
        raise ConversationRejected("conversation contains an empty turn")

    messages_dict = []
    for user_msg, ai_msg in zip(user_msgs, ai_msgs):
        messages_dict.append({"content": user_msg, "role": "user"})
        messages_dict.append({"content": ai_msg, "role": "assistant"})

    try:
        llama_v2_prompt(messages_dict)
    except Exception as e:
        raise ConversationRejected(f"llama_v2_prompt failed: {e}") from e

    return messages_dict, user_msgs


# --------------------------------------------------------------------------- #
# Core generation routine
# --------------------------------------------------------------------------- #

def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList(
        [StopOnSubstring(tokenizer, END_TAG, prompt_len)]
    )

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.8,
            stopping_criteria=stopping_criteria,
        )

    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def generate_and_save(
    model,
    tokenizer,
    prompt: str,
    fname: str,
    attribute_name: str,
    max_new_tokens: int,
    reject_banned: bool,
):
    """Generate, validate, then write. Raises ConversationRejected if the
    conversation is unusable — nothing is written in that case."""
    output = generate_one(model, tokenizer, prompt, max_new_tokens)

    conversation = extract_conversation(output)
    _, user_msgs = validate_conversation(conversation, attribute_name)

    leaks = leaked_stems(user_msgs, attribute_name)
    if leaks and reject_banned:
        raise ConversationRejected(f"trait word(s) in user turns: {leaks}")

    # Only touch the filesystem once the conversation has passed every check.
    with open(fname, "w", encoding="utf-8") as f:
        f.write(conversation)

    return leaks


def level_seed(base_seed: int, attribute_name: str, level: str) -> int:
    """Derive a stable per-subcategory seed, so that generating a subset via
    --attributes reproduces the same conversations as a full run."""
    digest = hashlib.sha256(f"{attribute_name}/{level}".encode()).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31 - 1)


def generate_attribute(
    model,
    tokenizer,
    attribute_name: str,
    num_samples: int,
    max_new_tokens: int,
    base_output_dir: str,
    base_seed: int,
    reject_banned: bool,
    progress_every: int,
    overall_pbar: tqdm,
) -> None:
    cfg = ATTRIBUTES[attribute_name]
    output_dir = os.path.join(base_output_dir, attribute_name)
    os.makedirs(output_dir, exist_ok=True)

    for level in cfg["levels"]:
        torch.manual_seed(level_seed(base_seed, attribute_name, level))

        i = 0
        rejected = 0
        errored = 0
        leaky = 0
        skipped = 0
        consecutive_failures = 0
        t0 = time.time()

        level_pbar = tqdm(
            total=num_samples,
            desc=f"{attribute_name}/{level}",
            unit="conv",
            position=1,
            leave=False,
        )
        try:
            while i < num_samples:
                fname = os.path.join(
                    output_dir, f"conversation_{i}_{attribute_name}_{level}.txt"
                )
                # Resume support: a completed run leaves only validated files behind.
                if os.path.exists(fname):
                    skipped += 1
                    i += 1
                    level_pbar.update(1)
                    overall_pbar.update(1)
                    continue

                prompt = build_prompt(attribute_name, level)
                try:
                    leaks = generate_and_save(
                        model, tokenizer, prompt, fname, attribute_name,
                        max_new_tokens, reject_banned,
                    )
                except ConversationRejected as e:
                    rejected += 1
                    consecutive_failures += 1
                    tqdm.write(f"[{attribute_name}/{level}] rejected #{rejected}: {e}")
                except torch.cuda.OutOfMemoryError:
                    # Retrying will not help; surface it immediately.
                    raise
                except Exception as e:
                    errored += 1
                    consecutive_failures += 1
                    tqdm.write(
                        f"[{attribute_name}/{level}] generation error #{errored}: "
                        f"{type(e).__name__}: {e}"
                    )
                else:
                    if leaks:
                        leaky += 1
                    consecutive_failures = 0
                    i += 1
                    level_pbar.update(1)
                    overall_pbar.update(1)
                    if progress_every and i % progress_every == 0:
                        rate = (time.time() - t0) / max(i - skipped, 1)
                        tqdm.write(
                            f"[{attribute_name}/{level}] {i}/{num_samples} "
                            f"({rate:.1f} s/conversation)"
                        )

                level_pbar.set_postfix(
                    rejected=rejected, errored=errored, leaky=leaky, refresh=False
                )

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"[{attribute_name}/{level}] aborting: "
                        f"{consecutive_failures} consecutive failures at sample {i}. "
                        "Check the messages above — a persistent rejection usually "
                        "means max_new_tokens is too low for the end tag to be "
                        "reached, or the prompt has stopped producing HUMAN:/"
                        "ASSISTANT: turns."
                    )
        finally:
            level_pbar.close()

        elapsed = time.time() - t0
        generated = num_samples - skipped
        tqdm.write(
            f"[{attribute_name}/{level}] done: {num_samples} saved "
            f"({generated} generated, {skipped} already present), "
            f"{rejected} rejected, {errored} errored, "
            f"{elapsed / 60:.1f} min "
            f"({elapsed / max(generated, 1):.1f} s/conversation)"
        )
        if generated:
            tqdm.write(
                f"[{attribute_name}/{level}] trait-word leakage in user turns: "
                f"{leaky}/{generated} ({100 * leaky / generated:.1f}%)"
                + ("" if reject_banned else "  [kept — rerun with --reject_banned to drop]")
            )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_samples", type=int, required=True,
        help="Number of conversations to generate per subcategory.",
    )
    parser.add_argument(
        "--attributes", nargs="+", default=list(ATTRIBUTES.keys()),
        choices=list(ATTRIBUTES.keys()),
        help="Which attributes to generate. Defaults to all four.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=768,
        help=(
            "Generation cap per conversation. The original notebook used 2048 "
            "with no early-stop check. Conversations that hit this cap before "
            "emitting the end tag are rejected and retried, so raise this if the "
            "rejection rate is high."
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets_llama2",
        help="Base directory for generated .txt files.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Base RNG seed. Each subcategory derives its own seed from this.",
    )
    parser.add_argument(
        "--reject_banned", action="store_true",
        help=(
            "Discard and retry any conversation whose user turns contain a trait "
            "word for that attribute. Off by default: the rate is always reported, "
            "so measure it before paying the extra generation cost."
        ),
    )
    parser.add_argument(
        "--progress_every", type=int, default=25,
        help="Print a progress line every N conversations. 0 disables.",
    )
    args = parser.parse_args()

    # `token=` is the correct kwarg; the notebook's `access_token=` was silently
    # swallowed into **kwargs and never used for authentication.
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "HF_TOKEN is not set — falling back to any cached `huggingface-cli "
            f"login` credentials. {MODEL_NAME} is a gated repo."
        )

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    # Load straight to fp16 rather than materialising ~52 GB of fp32 on the host
    # and halving afterwards.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, token=hf_token, torch_dtype=torch.float16,
    )
    model.cuda()
    model.eval()

    total_samples = sum(
        args.num_samples * len(ATTRIBUTES[attribute_name]["levels"])
        for attribute_name in args.attributes
    )
    with tqdm(
        total=total_samples, desc="Overall", unit="conv", position=0
    ) as overall_pbar:
        for attribute_name in args.attributes:
            tqdm.write(f"\n=== Generating: {attribute_name} ===")
            generate_attribute(
                model, tokenizer, attribute_name, args.num_samples,
                args.max_new_tokens, args.output_dir, args.seed,
                args.reject_banned, args.progress_every, overall_pbar,
            )


if __name__ == "__main__":
    main()
