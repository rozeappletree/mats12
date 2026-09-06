"""
Bulk synthetic dataset generation via Claude Opus 5 — Gullibility / Rationality /
Seriousness / Certainty-Seeking.

Scaled-up sibling of scripts/gen_opus_dataset.sample.py. That script was written
for smoke tests (one big call per attribute); this one is written for a real run
of 100 conversations per level, i.e. 300 conversations per attribute.

What changed, and why:

  * Small calls. Every API call asks for exactly 2 conversations per level
    (2 low / 2 medium / 2 high = 6 conversations). Small batches keep the model
    from getting lazy or truncating, and make each call cheap to lose.
  * Breadth-first scheduling. Calls are issued round-robin across attributes:
    one call for gullibility, one for rationality, one for seriousness, one for
    certainty_seeking, then back to the top. Levels stay balanced by
    construction (2/2/2 per call) and attributes stay balanced across rounds, so
    an interrupted run leaves a usable, evenly-sampled dataset rather than a
    finished first attribute and three empty ones.
  * Resumable. On startup the output directory is scanned and already-generated
    conversations count towards the quota. Re-running the same command picks up
    where the last one stopped; nothing is overwritten.
  * Live progress. A redrawn status block shows per-attribute / per-level
    counts, elapsed time, ETA and running API spend.

The API rejects temperature / top_p / top_k, so diversity is engineered into the
prompt instead: every call draws a fresh sample of topic domains, conversation
shapes and user voices, and the model is told those lists are suggestions rather
than a menu.

Usage:
    # the full run: 100 per level x 3 levels x 4 attributes = 1200 conversations
    python scripts/gen_opus_data100.py

    # resume after an interrupt — same command, already-written files are kept
    python scripts/gen_opus_data100.py

    # a smaller / narrower run
    python scripts/gen_opus_data100.py --per_level 10 --attributes rationality
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Literal, Tuple

from dotenv import load_dotenv
from anthropic import Anthropic
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

MODEL_NAME = "claude-opus-5"
BASE_URL = "https://api.opusgate.dev"

# Claude Opus 5 list pricing, USD per million tokens — used for the live spend
# readout only.
# i am using opusnet
PRICE_IN_PER_MTOK = 0.85
PRICE_OUT_PER_MTOK = 0.85

SEED = 75241239

# Conversations requested per level in a single call. 3 levels x 2 = 6 per call.
PER_CALL_PER_LEVEL = 2

# A conversation must contain at least this many complete user/assistant turn
# pairs to be kept.
MIN_TURN_PAIRS = 2

# Retry a single call at most this many times before giving up on it. A failed
# call is skipped, not fatal: the scheduler will come back to that attribute on
# the next round.
MAX_CALL_RETRIES = 4


# --------------------------------------------------------------------------- #
# Attribute definitions (shared with the Llama / sample scripts)
# --------------------------------------------------------------------------- #

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
# Diversity pools
#
# These are prompt *seeds*, not a taxonomy: each call samples a handful and the
# prompt explicitly tells the model it may go outside them. They exist to stop
# 200 calls collapsing onto the same dozen scenarios, not to define the space.
# --------------------------------------------------------------------------- #

TOPIC_DOMAINS = [
    # everyday / domestic
    "cooking, recipes and food", "grocery shopping and meal planning",
    "home repair and DIY", "cleaning and household organisation",
    "gardening, houseplants and allotments", "pets, vets and animal care",
    "moving house, renting and real estate", "furniture and interior decorating",
    "laundry, clothing care and repairs", "budget groceries and stretching a paycheck",
    # money / work
    "personal finance and budgeting", "taxes, benefits and paperwork",
    "insurance, claims and warranties", "investing, pensions and retirement",
    "career changes, CVs and interviews", "workplace conflict and office politics",
    "freelancing, contracts and invoicing", "small business and side projects",
    "salary negotiation and promotions", "job loss, redundancy and job hunting",
    # health / body
    "general health and symptoms", "fitness, training and injury recovery",
    "sleep, fatigue and energy", "nutrition, diets and supplements",
    "mental health, stress and burnout", "medications and side effects",
    "dentistry, eyesight and hearing", "pregnancy, fertility and postpartum",
    "ageing, caregiving and elder care", "chronic illness and disability",
    # people
    "relationships, dating and breakups", "family conflict and in-laws",
    "parenting babies and toddlers", "parenting teenagers",
    "friendship, loneliness and social life", "weddings, funerals and family events",
    "neighbours, housemates and shared living", "grief and difficult news",
    # tech
    "consumer technology and gadgets", "phones, laptops and buying advice",
    "software bugs, accounts and troubleshooting", "privacy, security and scams",
    "programming and self-taught coding", "AI tools and automation",
    "home networking, wifi and smart devices", "data backup and lost files",
    # world / knowledge
    "history and historical figures", "science, physics and space",
    "biology, ecology and conservation", "statistics, data and how studies work",
    "philosophy, ethics and thought experiments", "religion, ritual and belief",
    "politics, policy and civic process", "economics and how markets work",
    "languages and language learning", "law, courts and bureaucracy",
    # doing / making / going
    "travel planning and itineraries", "visas, borders and travel documents",
    "cars, maintenance and commuting", "cycling, walking and public transport",
    "hobbies, crafts and making things", "music, instruments and practice",
    "film, TV and books", "video games and tabletop games",
    "sports, teams and training", "photography and video",
    "art, design and creative process", "writing, editing and publishing",
    "education, exams and studying", "university applications and student life",
    "volunteering, community and local organising", "weather, hiking and the outdoors",
    "fishing, camping and survival skills", "collecting and second-hand markets",
    "events, tickets and logistics", "emergencies and things going wrong",
]

CONVERSATION_ANGLES = [
    "starts with a very specific narrow question",
    "starts vague and only gets specific under questioning",
    "the user is mid-task and slightly stressed",
    "the user is idly curious with no urgency at all",
    "the user brings a half-formed plan and wants feedback",
    "the user is comparing two or three concrete options",
    "the user is trying to fix something that already went wrong",
    "the user is planning months ahead for a future event",
    "the user is double-checking a decision they have already made",
    "the user is asking on behalf of someone else and relaying answers",
    "the user opens with a claim they read somewhere and wants it checked",
    "the user changes the subject partway through",
    "the user is under a hard deadline and wants the short version",
    "the user is procrastinating and keeps widening the question",
    "the user pushes back on the first answer they get",
    "the user is embarrassed about the situation and hedges the details",
    "the user gives a long rambling backstory before the actual question",
    "the user asks a follow-up that reveals they misread the first answer",
    "the user wants help deciding whether the thing is worth doing at all",
    "the user is stuck between advice from two different people",
    "the user is learning something new and asks for the basics first",
    "the user has a strong prior opinion and is looking for input anyway",
    "the user is troubleshooting step by step, reporting results each turn",
    "the user asks for a rough estimate or ballpark figure",
    "the user is drafting or wording something and wants it improved",
    "the user is worried about a worst case and wants it assessed",
    "the user has an unusual, non-standard version of a common problem",
    "the user keeps adding constraints they forgot to mention",
    "the user is trying to understand why something works, not just how",
    "the user needs to explain the answer to somebody else afterwards",
]

USER_VOICES = [
    "terse, lowercase, minimal punctuation",
    "long paragraphs with a lot of context",
    "polite and slightly formal",
    "blunt to the point of curtness",
    "chatty with digressions and asides",
    "technical vocabulary used confidently",
    "plain everyday words, no jargon",
    "non-native English phrasing, entirely fluent",
    "typos and autocorrect artefacts left in",
    "bullet points and numbered questions",
    "anxious and over-explaining",
    "wry and understated",
    "voice-to-text run-on sentences",
    "regional idiom and colloquialisms",
]


# --------------------------------------------------------------------------- #
# Pydantic schema for structured output
# --------------------------------------------------------------------------- #

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class Conversation(BaseModel):
    topic: str
    level: str
    turns: List[Turn]


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

def _level_block(attribute_name: str, level: str) -> str:
    cfg = ATTRIBUTES[attribute_name]
    return (
        f'  - level "{level}": the user {cfg["persona"][level]}: '
        f'{cfg["behavior"][level]}.'
    )


def build_prompt(attribute_name: str, per_level: int, rng: random.Random) -> str:
    cfg = ATTRIBUTES[attribute_name]
    levels = cfg["levels"]
    show_through = cfg["show_through"]
    total = per_level * len(levels)

    level_descriptions = "\n".join(_level_block(attribute_name, lv) for lv in levels)

    # One topic per conversation, plus a couple of spares, so the model has no
    # reason to reuse a subject within the batch.
    topics = rng.sample(TOPIC_DOMAINS, min(len(TOPIC_DOMAINS), total + 3))
    angles = rng.sample(CONVERSATION_ANGLES, min(len(CONVERSATION_ANGLES), total))
    voices = rng.sample(USER_VOICES, min(len(USER_VOICES), 5))

    schema_example = {
        "conversations": [
            {
                "topic": "short label for the subject",
                "level": levels[0],
                "turns": [
                    {"role": "user", "content": "HUMAN turn text"},
                    {"role": "assistant", "content": "ASSISTANT turn text"},
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                ],
            }
        ]
    }

    return f"""You are generating a synthetic dataset of conversations between a human \
user and an AI assistant, to study one behavioural attribute: **{attribute_name}**.

This attribute has {len(levels)} levels. The *only* thing that should differ \
between levels is {show_through}. Everything else — subject matter, tone, \
conversation length, assistant helpfulness — should be equally varied across all levels.

Levels:
{level_descriptions}

Produce EXACTLY {per_level} distinct conversations for EACH level \
({total} conversations total).

Hard requirements for every conversation:
  * At least {MIN_TURN_PAIRS + 1} user/assistant exchanges (so >= {2 * (MIN_TURN_PAIRS + 1)} turns), alternating user, assistant, user, assistant...
  * The first turn is the user, and turns strictly alternate.
  * The user must NEVER name or describe this characteristic about themselves. \
Convey the level only through {show_through}.
  * Do not use any obvious label word for the trait in the user's turns.
  * The assistant should behave like a normal, competent, helpful AI — its \
behaviour must NOT change between levels.

CREATIVITY AND DIVERSITY ARE THE MOST IMPORTANT THING. This batch is one of many; \
if it looks like the obvious, median set of conversations someone would write for \
this attribute, it has failed. Push into the full space of realistic situations \
people actually bring to an assistant — mundane, awkward, technical, domestic, \
high-stakes, trivial.

Seeds for this batch (suggestions to react to, NOT a menu — you do not need to \
stick to these lists, and inventing topics, shapes and voices outside them is \
encouraged):
  * Subject areas: {"; ".join(topics)}.
  * Conversation shapes: {"; ".join(angles)}.
  * User writing voices: {"; ".join(voices)}.

Also vary, without ever stating any of it explicitly:
  * Message length — some users write one line, some write a paragraph.
  * Conversation length — some 3 exchanges, some 6 or more.
  * Implied life circumstances, expertise level and register.
  * Whether the conversation resolves neatly or just stops.

Use a different subject for every conversation in this batch.

Return ONLY a single JSON object, no prose before or after, matching this shape:

{json.dumps(schema_example, indent=2)}

Each conversation's "level" field must be exactly one of: {", ".join(levels)}.
"""


# --------------------------------------------------------------------------- #
# Parsing / validation
# --------------------------------------------------------------------------- #

class BatchRejected(Exception):
    """A call returned but failed a structural / quality check. Retryable."""


def extract_json(text: str) -> dict:
    """Pull the first top-level JSON object out of the model reply."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise BatchRejected("no JSON object found in reply")
        candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise BatchRejected(f"JSON decode failed: {e}") from e


def leaked_stems(user_msgs: List[str], attribute_name: str) -> List[str]:
    cfg = ATTRIBUTES[attribute_name]
    text = " ".join(user_msgs).lower()
    return [s for s in cfg["leak_stems"] if re.search(r"\b" + re.escape(s), text)]


def validate_conversation(conv: Conversation, attribute_name: str) -> None:
    """Structural check on a single conversation. Raises BatchRejected."""
    levels = ATTRIBUTES[attribute_name]["levels"]
    if conv.level not in levels:
        raise BatchRejected(f"unknown level {conv.level!r}")
    if not conv.turns:
        raise BatchRejected("conversation has no turns")

    expected = "user"
    for t in conv.turns:
        if t.role != expected:
            raise BatchRejected(
                f"turns do not alternate (expected {expected}, got {t.role})"
            )
        if not t.content.strip():
            raise BatchRejected("empty turn content")
        expected = "assistant" if expected == "user" else "user"

    n_user = sum(1 for t in conv.turns if t.role == "user")
    n_ai = len(conv.turns) - n_user
    if n_user < MIN_TURN_PAIRS + 1 or n_ai < MIN_TURN_PAIRS:
        raise BatchRejected(f"too few exchanges: {n_user} user / {n_ai} assistant")


def conversation_to_text(conv: Conversation) -> str:
    """Render to the HUMAN:/ASSISTANT: text format the Llama script writes."""
    lines = []
    for t in conv.turns:
        tag = "HUMAN: " if t.role == "user" else "ASSISTANT: "
        lines.append(tag + " ".join(t.content.split()))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Live progress display
# --------------------------------------------------------------------------- #

class Progress:
    """A redrawn status block: per-attribute/level counts, timing, spend.

    Anything the run wants to print goes through .log(), which erases the block,
    prints the line, and redraws underneath it — so scrollback stays readable.
    """

    def __init__(self, attributes: List[str], target: int, done: Dict[str, Dict[str, int]]):
        self.attributes = attributes
        self.target = target
        self.counts = done
        self.t0 = time.time()
        self.calls = 0
        self.failures = 0
        self.dropped = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.start_total = self.total()
        self.lines_drawn = 0
        self.enabled = sys.stdout.isatty()

    # -- state ------------------------------------------------------------- #

    def total(self) -> int:
        return sum(sum(lv.values()) for lv in self.counts.values())

    def grand_target(self) -> int:
        return sum(len(ATTRIBUTES[a]["levels"]) for a in self.attributes) * self.target

    def cost(self) -> float:
        return (self.in_tokens * PRICE_IN_PER_MTOK
                + self.out_tokens * PRICE_OUT_PER_MTOK) / 1_000_000

    # -- rendering --------------------------------------------------------- #

    @staticmethod
    def _bar(done: int, total: int, width: int = 20) -> str:
        filled = 0 if total <= 0 else int(width * min(done, total) / total)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _hms(seconds: float) -> str:
        seconds = int(max(seconds, 0))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    def _render_lines(self) -> List[str]:
        elapsed = time.time() - self.t0
        made = self.total() - self.start_total          # this session only
        remaining = self.grand_target() - self.total()
        rate = made / elapsed if elapsed > 0 and made else 0.0
        eta = remaining / rate if rate > 0 else None
        cost = self.cost()
        cost_per = cost / made if made else 0.0

        width = shutil.get_terminal_size((100, 24)).columns
        lines = ["", "─" * min(width, 78)]

        for attr in self.attributes:
            levels = ATTRIBUTES[attr]["levels"]
            attr_done = sum(self.counts[attr].values())
            attr_target = self.target * len(levels)
            lines.append(
                f"  {attr:<18} {self._bar(attr_done, attr_target)} "
                f"{attr_done:>4}/{attr_target}"
            )
            per_level = "  ".join(
                f"{lv}={self.counts[attr][lv]:>3}/{self.target}" for lv in levels
            )
            lines.append(f"  {'':<18} {per_level}")

        lines.append("─" * min(width, 78))
        lines.append(
            f"  total {self.total():>5}/{self.grand_target()}   "
            f"calls {self.calls}   failed {self.failures}   dropped {self.dropped}"
        )
        lines.append(
            f"  elapsed {self._hms(elapsed)}   "
            f"eta {self._hms(eta) if eta is not None else '--:--'}   "
            f"{rate * 60:.1f} conv/min"
        )
        lines.append(
            f"  spend ${cost:.2f}   (${cost_per:.4f}/conv, "
            f"in {self.in_tokens:,} tok / out {self.out_tokens:,} tok)"
        )
        lines.append("─" * min(width, 78))
        return lines

    def _erase(self) -> None:
        if self.enabled and self.lines_drawn:
            sys.stdout.write(f"\033[{self.lines_drawn}A\033[J")
        self.lines_drawn = 0

    def render(self) -> None:
        if not self.enabled:
            return
        self._erase()
        lines = self._render_lines()
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self.lines_drawn = len(lines) + 1

    def log(self, message: str) -> None:
        self._erase()
        print(message)
        self.render()

    def finish(self) -> None:
        """Leave the final block on screen and stop redrawing over it."""
        if self.enabled:
            self.render()
            self.lines_drawn = 0
        else:
            print("\n".join(self._render_lines()))


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def call_model(client: Anthropic, prompt: str, max_tokens: int) -> Tuple[str, int, int]:
    """One API call → (text, input_tokens, output_tokens)."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    )
    usage = response.usage
    return text, usage.input_tokens, usage.output_tokens


def next_indices(output_dir: str, attribute: str, levels: List[str]) -> Dict[str, int]:
    """Highest existing conversation index per level, + 1. Nothing is overwritten."""
    nxt = {lv: 0 for lv in levels}
    if not os.path.isdir(output_dir):
        return nxt
    for fname in os.listdir(output_dir):
        m = re.fullmatch(rf"conversation_(\d+)_{re.escape(attribute)}_(\w+)\.txt", fname)
        if m and m.group(2) in nxt:
            nxt[m.group(2)] = max(nxt[m.group(2)], int(m.group(1)) + 1)
    return nxt


def count_existing(output_dir: str, attribute: str, levels: List[str]) -> Dict[str, int]:
    """How many conversations are already on disk, per level."""
    counts = {lv: 0 for lv in levels}
    if not os.path.isdir(output_dir):
        return counts
    for fname in os.listdir(output_dir):
        m = re.fullmatch(rf"conversation_\d+_{re.escape(attribute)}_(\w+)\.txt", fname)
        if m and m.group(1) in counts:
            counts[m.group(1)] += 1
    return counts


def run_one_call(
    client: Anthropic,
    attribute: str,
    output_dir: str,
    remaining: Dict[str, int],
    indices: Dict[str, int],
    rng: random.Random,
    max_tokens: int,
    reject_banned: bool,
    call_idx: int,
    seed: int,
    prog: Progress,
) -> int:
    """Issue one call for `attribute` (2 per level) and write what it returns.

    Returns the number of conversations saved. A call that fails every retry
    logs and returns 0 rather than aborting the run — the scheduler will retry
    this attribute on the next round.
    """
    levels = ATTRIBUTES[attribute]["levels"]

    convs: List[Conversation] = []
    prompt = ""
    for attempt in range(1, MAX_CALL_RETRIES + 1):
        prompt = build_prompt(attribute, PER_CALL_PER_LEVEL, rng)
        try:
            raw, tok_in, tok_out = call_model(client, prompt, max_tokens)
            prog.in_tokens += tok_in
            prog.out_tokens += tok_out
            prog.calls += 1

            data = extract_json(raw)
            raw_convs = data.get("conversations")
            if not isinstance(raw_convs, list) or not raw_convs:
                raise BatchRejected("no 'conversations' list in reply")

            for i, item in enumerate(raw_convs):
                try:
                    conv = Conversation.model_validate(item)
                    validate_conversation(conv, attribute)
                except (ValidationError, BatchRejected) as e:
                    prog.dropped += 1
                    prog.log(f"  [{attribute}] dropped conversation {i}: {e}")
                    continue
                convs.append(conv)

            if not convs:
                raise BatchRejected("no usable conversations in batch")
            break
        except BatchRejected as e:
            prog.failures += 1
            prog.log(f"  [{attribute}] call {call_idx} attempt {attempt} rejected: {e}")
        except Exception as e:  # transient API / network errors
            prog.failures += 1
            prog.log(f"  [{attribute}] call {call_idx} attempt {attempt} "
                     f"error: {type(e).__name__}: {e}")
        if attempt < MAX_CALL_RETRIES:
            time.sleep(2 * attempt)

    if not convs:
        prog.log(f"  [{attribute}] call {call_idx} gave up after "
                 f"{MAX_CALL_RETRIES} attempts; will retry next round")
        return 0

    saved = 0
    for conv in convs:
        lv = conv.level
        if remaining.get(lv, 0) <= 0:
            continue
        user_msgs = [t.content for t in conv.turns if t.role == "user"]
        leaks = leaked_stems(user_msgs, attribute)
        if leaks and reject_banned:
            prog.dropped += 1
            prog.log(f"  [{attribute}/{lv}] dropped (trait words {leaks})")
            continue

        idx = indices[lv]
        stem = f"conversation_{idx}_{attribute}_{lv}"
        with open(os.path.join(output_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(conversation_to_text(conv))

        meta = {
            "attribute": attribute,
            "level": lv,
            "topic": conv.topic,
            "index": idx,
            "model": MODEL_NAME,
            "seed": seed,
            "call_index": call_idx,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "num_turns": len(conv.turns),
            "leak_stems_found": leaks,
            "prompt": prompt,
        }
        with open(os.path.join(output_dir, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        indices[lv] += 1
        remaining[lv] -= 1
        prog.counts[attribute][lv] += 1
        saved += 1

    return saved


def generate(
    client: Anthropic,
    attributes: List[str],
    per_level: int,
    output_dir: str,
    seed: int,
    max_tokens: int,
    reject_banned: bool,
) -> Progress:
    """Breadth-first over attributes: one call each, round after round."""
    state = {}
    for attr in attributes:
        levels = ATTRIBUTES[attr]["levels"]
        attr_dir = os.path.join(output_dir, attr)
        os.makedirs(attr_dir, exist_ok=True)
        done = count_existing(attr_dir, attr, levels)
        state[attr] = {
            "dir": attr_dir,
            "levels": levels,
            "remaining": {lv: max(per_level - done[lv], 0) for lv in levels},
            "indices": next_indices(attr_dir, attr, levels),
            "done": done,
            "calls": 0,
        }

    resumed = sum(sum(s["done"].values()) for s in state.values())
    if resumed:
        print(f"Resuming: {resumed} conversation(s) already on disk are being kept.")

    prog = Progress(attributes, per_level, {a: state[a]["done"] for a in attributes})
    prog.render()

    round_idx = 0
    while any(any(v > 0 for v in state[a]["remaining"].values()) for a in attributes):
        progressed = False
        for attr in attributes:
            s = state[attr]
            if not any(v > 0 for v in s["remaining"].values()):
                continue
            s["calls"] += 1
            # Fresh RNG per call so each prompt draws a different diversity seed,
            # while the whole run stays reproducible from --seed.
            rng = random.Random(f"{seed}/{attr}/{round_idx}/{s['calls']}")
            saved = run_one_call(
                client, attr, s["dir"], s["remaining"], s["indices"], rng,
                max_tokens, reject_banned, s["calls"], seed, prog,
            )
            progressed = progressed or saved > 0
            prog.render()

        round_idx += 1
        if not progressed:
            prog.log("No attribute made progress this round — stopping.")
            break

    prog.finish()
    return prog


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per_level", type=int, default=100,
        help="Conversations per level, per attribute (default 100 → 300/attribute).",
    )
    parser.add_argument(
        "--attributes", nargs="+", default=list(ATTRIBUTES.keys()),
        choices=list(ATTRIBUTES.keys()),
        help="Which attributes to generate. Defaults to all four.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets_claudeopus_100",
        help="Base directory for generated .txt files.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Base RNG seed for topic/angle/voice sampling.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=16000,
        help="max_tokens per API call. Raise if batches truncate.",
    )
    parser.add_argument(
        "--reject_banned", action="store_true",
        help=(
            "Discard any conversation whose user turns contain a trait word for "
            "that attribute. Off by default: leakage is recorded in the sidecar."
        ),
    )
    args = parser.parse_args()

    api_key = os.getenv("OPUSKEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Set OPUSKEY or ANTHROPIC_API_KEY in your "
            "environment or .env file."
        )

    client = Anthropic(api_key=api_key, base_url=BASE_URL)

    total_target = sum(len(ATTRIBUTES[a]["levels"]) for a in args.attributes) * args.per_level
    print(f"Model:       {MODEL_NAME}")
    print(f"Attributes:  {', '.join(args.attributes)}")
    print(f"Per level:   {args.per_level}   (target {total_target} conversations)")
    print(f"Per call:    {PER_CALL_PER_LEVEL} per level, breadth-first across attributes")
    print(f"Output:      {args.output_dir}/")

    try:
        prog = generate(
            client, args.attributes, args.per_level, args.output_dir,
            args.seed, args.max_tokens, args.reject_banned,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.")
        return

    print(f"\nDone: {prog.total()} conversations, {prog.calls} calls, "
          f"${prog.cost():.2f} spent.")


if __name__ == "__main__":
    main()
