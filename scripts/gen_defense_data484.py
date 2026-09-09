"""
Synthetic dataset generation via GPT-5.6-Sol — gullible vs. non-gullible human
defenses of TruthfulQA-style claims.

Input: data/finetune/split/non_hard_negatives_484.json (484 questions, each with
a best_answer / correct_answers list and an incorrect_answers list).

For every question we make exactly ONE API call and ask for 6 conversations:
  * 3 where the human is NOT gullible and defends/proves the objectively true
    answer(s) against the assistant — saved with gullibility level "low".
  * 3 where the human IS gullible and defends one of the objectively false
    answer(s) as though it were true — saved with gullibility level "high".
A question is only ever sent once — see "Resumability" below.

Output files are named conversation_{i}_{category}_gullibility_{low|high}.txt,
the layout src/probe_common.py's TextDataset expects: it takes the label from
the last "_gullibility_" in the filename through ".txt", and ATTRIBUTE_LABELS
recognises only low / medium / high. The leading integer keeps the older
TalkTuner loader in scripts/dataset.py happy, which int()s the token right
after "conversation_".

What changed relative to scripts/gen_sol_data100.py, and why (most of the
plumbing — resumability, progress display, retry handling, prompt-diversity
seeding — is lifted straight from that script):

  * Unit of work is a QUESTION, not an attribute/level. One call per question
    yields 3 low-gullibility + 3 high-gullibility conversations (6 total),
    instead of 2-per-level across 3 levels. Only the two poles are generated
    here — there is no "medium" — since a defense of a claim is either taken
    in or seen through.
  * Round-robin scheduling is over the dataset's "category" field rather than
    over attributes. The global processing queue is built by cycling through
    categories and popping one question from each in turn, so an interrupted
    run leaves every category proportionally covered rather than exhausting
    categories in file order.
  * Resumability is question-level: a JSON manifest
    (<output_dir>/processed_questions.json) records the index of every
    question a call has already been attempted for. Re-running the same
    command skips those questions outright — a question is sent to the model
    at most once, successful or not, matching the "don't ask twice" spec.
  * The API rejects temperature / top_p / top_k, so diversity is engineered
    into the prompt instead: every call samples a few conversation shapes and
    user voices and is told to make the 6 outputs stylistically distinct from
    one another.
  * Conversations must be SELF-CONTAINED. The dataset exists to support
    inferring a user model from a conversation alone, and the .txt is all a
    reader or a probe ever sees — both loaders keep only the HUMAN:/ASSISTANT:
    lines and drop everything else, so the sidecar .json is provenance, not
    context. An early batch showed the model treating the source question as
    ambient shared context: openers like "new york city" answering a question
    neither speaker ever states, and both parties discussing "the question" as
    though looking at a quiz sheet.

    Self-containment is a property of the whole conversation, not of its first
    turn. Requiring the opener to carry it was tried and backfired — all six
    conversations for a question then began by reciting the same quoted saying
    verbatim, which is both repetitive and a worse opener than the ones it
    replaced. The prompt now asks for the subject and the human's position to
    be established somewhere in the exchange, and explicitly asks the six to
    differ in where and how that happens.

    Enforcement is deliberately thin. Only quiz-meta talk (QUIZ_META_RE) is
    rejected, because it is unambiguous wherever it appears. Mechanical proxies
    for the rest do not survive contact with the data: lexical overlap with the
    source question scores a perfectly self-contained conversation no higher
    than a context-free one, and a rejected conversation is dropped rather than
    retried, so a false positive costs data silently. Overlap is therefore
    reported only (LOW_COVERAGE_WARN, "flagged" in the progress display).


Usage:
    # the full run over all 484 questions
    python scripts/gen_defense_data484.py

    # resume after an interrupt — same command, already-attempted questions
    # are skipped
    python scripts/gen_defense_data484.py

    # a smaller / narrower run
    python scripts/gen_defense_data484.py --limit 20
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
from typing import Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

MODEL_NAME = "gpt-5.6-sol"
BASE_URL = "https://api.opusgate.dev/v1"

# GPT-5.6-Sol list pricing, USD per million tokens — used for the live spend
# readout only.
PRICE_IN_PER_MTOK = 0.85
PRICE_OUT_PER_MTOK = 0.85

SEED = 75241239

DEFAULT_INPUT = "data/finetune/split/non_hard_negatives_484.json"
DEFAULT_OUTPUT_DIR = "datasets_defense_484"

# Conversations requested per label, per call. 2 labels x 3 = 6 per call.
PER_CALL_PER_LABEL = 3

# A conversation must contain at least this many complete user/assistant turn
# pairs to be kept.
MIN_TURN_PAIRS = 2

# Self-containment is a property of the WHOLE conversation, not of its first
# turn: the context may be established in turn 1, or emerge in turn 3, as long
# as a stranger reading only this .txt ends up knowing what is discussed and
# what the human believes. Requiring it in the opener was tried and produced
# six near-identical conversations per question, each reciting the source
# question verbatim before saying anything.
#
# That property is semantic, and mechanical checks for it do not survive
# contact with the data — lexical overlap with the source question scores a
# perfectly self-contained conversation ("Let's reason from the transaction
# record. The ATM sends an encrypted PIN verification request...") no higher
# than a genuinely context-free one. So overlap is reported, never enforced:
# LOW_COVERAGE_WARN only flags a conversation in the run log for a human to
# look at. The prompt is what actually secures self-containment.
LOW_COVERAGE_WARN = 0.34

_COVERAGE_STOPWORDS = frozenset("""
a an the and or but if then than that this these those is are was were be been being
do does did doing have has had having will would shall should can could may might must
of in on at to for from by with about into over after before between out against during
what which who whom whose when where why how all any both each few more most other some
such no nor not only own same so too very just also as it its you your yours i me my we
our they them their he she his her there here many much get got
""".split())


def _coverage_terms(text: str) -> "set[str]":
    """Content words of `text`, crudely stemmed so "backwards" matches
    "backward" and "Americans" matches "American"."""
    terms = set()
    for word in re.findall(r"[a-z']{3,}", text.lower()):
        if word in _COVERAGE_STOPWORDS:
            continue
        for suffix in ("ies", "es", "ed", "ing", "s"):
            if len(word) - len(suffix) >= 3 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        terms.add(word)
    return terms


def topic_coverage(conv: "Conversation", question: str) -> float:
    """Fraction of the question's content words the conversation also uses.

    A rough proxy for "does this conversation visibly concern its question",
    used only to flag conversations worth eyeballing. See LOW_COVERAGE_WARN.
    """
    q_terms = _coverage_terms(question)
    if not q_terms:
        return 1.0
    conv_terms = _coverage_terms(" ".join(t.content for t in conv.turns))
    return len(q_terms & conv_terms) / len(q_terms)


# Talking about a quiz item rather than about the world. Rejected anywhere in
# the conversation, human or assistant turns alike.
QUIZ_META_RE = re.compile(
    r"\b(?:the|this)\s+question\s+(?:asks|says|states|is asking)\b"
    r"|\bwhat\s+the\s+question\s+(?:asks|is asking)\b"
    r"|\bthe\s+(?:correct|right|intended)\s+answer\s+to\s+(?:the|this)\s+question\b",
    re.IGNORECASE,
)

# Retry a single call at most this many times before giving up on it. A failed
# call still counts the question as "processed" (see Resumability above) —
# retries are for transient errors within one attempt, not a second attempt
# on a later run.
MAX_CALL_RETRIES = 4

ATTRIBUTE = "gullibility"

# Gullibility levels, matching ATTRIBUTE_LABELS in src/probe_common.py: a human
# who defends the false answer is "high", one who sees through it is "low".
LABELS = ["low", "high"]


# --------------------------------------------------------------------------- #
# Diversity pools (prompt seeds, not a taxonomy — see gen_sol_data100.py)
# --------------------------------------------------------------------------- #

CONVERSATION_ANGLES = [
    "the human opens by flatly asserting their answer as settled fact",
    "the human asks a leading question that assumes their answer is right",
    "the human brings up a source (real, vague, or misremembered) to back their claim",
    "the human reacts to the assistant's first answer with pushback",
    "the human tells a personal anecdote as evidence",
    "the human cites something 'everyone knows' or a common saying",
    "the human is mid-argument with someone else and wants ammunition",
    "the human is quizzing the assistant to test if it 'really' knows",
    "the human brings up the topic casually, almost in passing",
    "the human is annoyed the assistant hasn't already confirmed their view",
    "the human builds their case turn by turn, adding a new point each time",
    "the human concedes a small point but holds the main line",
    "the human asks the assistant to just confirm rather than explain",
    "the human challenges the assistant to prove them wrong",
    "the human references a book, documentary, teacher, or relative as their source",
    "the human starts uncertain and firms up their position as the exchange goes on",
]

USER_VOICES = [
    "terse, lowercase, minimal punctuation",
    "long paragraphs with a lot of context",
    "polite and slightly formal",
    "blunt to the point of curtness",
    "chatty with digressions and asides",
    "confident and a little condescending",
    "plain everyday words, no jargon",
    "non-native English phrasing, entirely fluent",
    "bullet points and numbered claims",
    "wry and understated",
    "voice-to-text run-on sentences",
    "regional idiom and colloquialisms",
    "argumentative, quoting the assistant back at it",
]


# --------------------------------------------------------------------------- #
# Pydantic schema for structured output
# --------------------------------------------------------------------------- #

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class Conversation(BaseModel):
    turns: List[Turn]


class BatchResponse(BaseModel):
    not_gullible_conversations: List[Conversation] = Field(default_factory=list)
    gullible_conversations: List[Conversation] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loading + round-robin queue over categories
# --------------------------------------------------------------------------- #

def load_questions(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"expected a JSON list in {path}, got {type(data).__name__}")
    return data


def build_round_robin_queue(questions: List[dict], rng: random.Random) -> List[int]:
    """Indices into `questions`, ordered by round-robin over the category field.

    Categories are visited in the order they first appear in the file. Within
    a category, question order is shuffled (deterministically, from `rng`) so
    repeated runs with the same --seed are reproducible but not tied to file
    order. One question is popped from each non-exhausted category per round.
    """
    by_category: Dict[str, List[int]] = {}
    category_order: List[str] = []
    for idx, q in enumerate(questions):
        cat = q.get("category", "Uncategorized")
        if cat not in by_category:
            by_category[cat] = []
            category_order.append(cat)
        by_category[cat].append(idx)

    for cat in category_order:
        rng.shuffle(by_category[cat])

    queue: List[int] = []
    while any(by_category[cat] for cat in category_order):
        for cat in category_order:
            if by_category[cat]:
                queue.append(by_category[cat].pop(0))
    return queue


# --------------------------------------------------------------------------- #
# Resumability: which question indices have already been attempted
# --------------------------------------------------------------------------- #

def load_processed(output_dir: str) -> "set[int]":
    path = os.path.join(output_dir, "processed_questions.json")
    if not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def mark_processed(output_dir: str, processed: "set[int]", qidx: int) -> None:
    processed.add(qidx)
    _write_index(output_dir, "processed_questions.json", processed)


def load_failed(output_dir: str) -> "set[int]":
    path = os.path.join(output_dir, "failed_questions.json")
    if not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def mark_failed(output_dir: str, failed: "set[int]", qidx: int) -> None:
    """A question whose call produced nothing usable.

    Recorded separately from processed_questions.json so that a run lost to
    transient API errors can be rescued with --retry_failed, without ever
    re-sending a question that actually produced conversations.
    """
    failed.add(qidx)
    _write_index(output_dir, "failed_questions.json", failed)


def unmark_failed(output_dir: str, failed: "set[int]", qidx: int) -> None:
    if qidx in failed:
        failed.discard(qidx)
        _write_index(output_dir, "failed_questions.json", failed)


def _write_index(output_dir: str, filename: str, values: "set[int]") -> None:
    path = os.path.join(output_dir, filename)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sorted(values), f)
    os.replace(tmp_path, path)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

def _numbered(items: List[str]) -> str:
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_prompt(question: dict, rng: random.Random) -> str:
    q_text = question["question"]
    best_answer = question["best_answer"]
    correct = _dedup_keep_order([best_answer] + list(question.get("correct_answers", [])))
    incorrect = _dedup_keep_order(list(question.get("incorrect_answers", [])))

    angles = rng.sample(CONVERSATION_ANGLES, min(len(CONVERSATION_ANGLES), 6))
    voices = rng.sample(USER_VOICES, min(len(USER_VOICES), 6))

    schema_example = {
        "not_gullible_conversations": [
            {
                "turns": [
                    {"role": "user", "content": "HUMAN: ..."},
                    {"role": "assistant", "content": "ASSISTANT: ..."},
                    {"role": "user", "content": "HUMAN: ..."},
                    {"role": "assistant", "content": "ASSISTANT: ..."},
                ]
            }
        ],
        "gullible_conversations": [
            {
                "turns": [
                    {"role": "user", "content": "HUMAN: ..."},
                    {"role": "assistant", "content": "ASSISTANT: ..."},
                    {"role": "user", "content": "HUMAN: ..."},
                    {"role": "assistant", "content": "ASSISTANT: ..."},
                ]
            }
        ],
    }

    return f"""Given below is a genuine question seeking objective truth, which is \
sometimes concealed by plausible seemingly right but incorrect answers. Sometimes \
it is easy to see through them and sometimes it is not due to various factors. You \
are also given its objectively true right answer.

You are generating conversations between a human and an AI assistant where the \
human is either gullible by not being able to see through the facade, or he is NOT \
gullible by being able to see the whole truth in its entirety.

Question: {q_text}

Correct Answer(s):
{_numbered(correct)}

Seemingly Correct But Wrong Answer(s):
{_numbered(incorrect) if incorrect else "(none given)"}

It is important to note that the above correct answer(s) are 100% objectively true \
for the question and the above incorrect answer(s) are 100% objectively false for \
the same question.

Generate {PER_CALL_PER_LABEL} conversations where the human is NOT gullible and can \
prove to the AI assistant that the correct answer(s) is objectively true. Be \
creative on how the AI responds.

Generate {PER_CALL_PER_LABEL} conversations where the human IS gullible and tries to \
defend one of the objectively false statements above. Be creative on how the AI \
responds.

Hard requirements for every conversation:
  * At least {MIN_TURN_PAIRS + 1} user/assistant exchanges (so >= {2 * (MIN_TURN_PAIRS + 1)} turns), \
alternating user, assistant, user, assistant, ...
  * The first turn is the user (human), and turns strictly alternate.
  * Every human turn's content must start with "HUMAN: " and every assistant \
turn's content must start with "ASSISTANT: ".
  * A "gullible" conversation must have the human asserting or defending one of \
the Seemingly Correct But Wrong Answers above as if it were true — never the \
correct answer.
  * A "not gullible" conversation must have the human asserting, arriving at, or \
defending one of the Correct Answers above — never one of the wrong answers.
  * The assistant should behave like a normal, competent, helpful AI throughout \
(it should not be scripted to simply "win" the argument for its own sake — let the \
human's gullibility or lack of it come through in how they engage with what the \
assistant says).
  * SELF-CONTAINED. Each conversation is stored and read entirely on its own. \
Nobody reading it can see the question, the answer lists, or anything else above \
— only the turns themselves. By the time it ends, a stranger who has read only \
this conversation must be able to tell what subject is being discussed and what \
the human believes about it.
  * That does NOT have to happen in the opening turn. The human may start \
obliquely, vaguely, or in the middle of a thought and let the subject come into \
focus over the next turn or two; the assistant may be the one who names it. A \
human may also arrive at their position gradually rather than declaring it up \
front. What is not allowed is for anything essential to live OUTSIDE the \
conversation. Two speakers arguing about "the answer" while neither ever says \
what the underlying claim is have failed this, however many turns they take.
  * VARY WHERE THE CONTEXT COMES FROM. All 6 conversations concern the same \
subject, so do not let all 6 establish it the same way — and in particular do \
not have each of them recite the same quoted saying, scenario or question \
verbatim before getting started. Some should land the subject in the first \
sentence; others should let it emerge by the second or third turn.
  * TALK ABOUT THE WORLD, NOT ABOUT A QUIZ. The human and the assistant are two \
people discussing a topic, not two people looking at a test item. Neither may \
refer to "the question", "the answer", "the options" or "the passage" as things \
they can both see, and neither may lean on a demonstrative ("that label", "this \
claim", "the scenario") pointing at something the conversation never introduces \
at any point. Where the question above quotes a saying, riddle or scenario, one \
of the speakers has to bring that material in themselves — quote it, paraphrase \
it, or say where they ran into it.
  * Terse, lowercase, distracted or casual humans are all still welcome, in the \
opening turn as much as anywhere else. Brevity is not the problem; a subject \
that never appears is.

CREATIVITY AND DIVERSITY ARE IMPORTANT. All 6 conversations are about the same \
question, so they must be varied in shape, tone, length and how the human argues, \
or the batch will read as repetitive. Give each of the 6 conversations a distinct \
feel.

Seeds for this batch (suggestions to react to, NOT a menu — you do not need to \
stick to these lists):
  * Conversation shapes: {"; ".join(angles)}.
  * User writing voices: {"; ".join(voices)}.

Also vary, without ever stating any of it explicitly:
  * Message length — some turns are one line, some are a paragraph.
  * Conversation length — some 3 exchanges, some 6 or more.
  * Whether the conversation resolves neatly, stays unresolved, or ends mid-argument.

Return ONLY a single JSON object, no prose before or after, matching this shape:

{json.dumps(schema_example, indent=2)}

Return the 6 conversations (3 in "not_gullible_conversations", 3 in \
"gullible_conversations") as this JSON object.
"""


# --------------------------------------------------------------------------- #
# Parsing / validation
# --------------------------------------------------------------------------- #

class BatchRejected(Exception):
    """A call returned but failed a structural / quality check. Retryable."""


def extract_json(text: str) -> dict:
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


def validate_conversation(conv: Conversation, label: str) -> None:
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
        prefix = "HUMAN: " if t.role == "user" else "ASSISTANT: "
        if not t.content.strip().startswith(prefix):
            raise BatchRejected(f"turn missing required '{prefix}' prefix")
        expected = "assistant" if expected == "user" else "user"

    n_user = sum(1 for t in conv.turns if t.role == "user")
    n_ai = len(conv.turns) - n_user
    if n_user < MIN_TURN_PAIRS + 1 or n_ai < MIN_TURN_PAIRS:
        raise BatchRejected(f"too few exchanges: {n_user} user / {n_ai} assistant")

    # Quiz-meta talk is the one self-containment failure worth rejecting on:
    # it is unambiguous wherever it appears, and it marks a conversation whose
    # speakers are looking at a test item the reader cannot see. Everything
    # else about self-containment is left to the prompt and to the
    # LOW_COVERAGE_WARN report — see the note by that constant.
    for t in conv.turns:
        meta = QUIZ_META_RE.search(t.content)
        if meta:
            raise BatchRejected(
                f"{t.role} turn treats the topic as a quiz item: {meta.group(0)!r}"
            )


def conversation_to_text(conv: Conversation) -> str:
    lines = []
    for t in conv.turns:
        lines.append(" ".join(t.content.split()))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Live progress display
# --------------------------------------------------------------------------- #

class Progress:
    def __init__(self, target_questions: int, done_questions: int):
        self.target_questions = target_questions
        self.done_questions = done_questions
        self.start_done_questions = done_questions
        self.saved = {lb: 0 for lb in LABELS}
        self.t0 = time.time()
        self.calls = 0
        self.failures = 0
        self.dropped = 0
        self.flagged = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.lines_drawn = 0
        self.enabled = sys.stdout.isatty()

    def cost(self) -> float:
        return (self.in_tokens * PRICE_IN_PER_MTOK
                + self.out_tokens * PRICE_OUT_PER_MTOK) / 1_000_000

    @staticmethod
    def _bar(done: int, total: int, width: int = 30) -> str:
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
        made = self.done_questions - self.start_done_questions
        remaining = self.target_questions - self.done_questions
        rate = made / elapsed if elapsed > 0 and made else 0.0
        eta = remaining / rate if rate > 0 else None
        cost = self.cost()
        cost_per_q = cost / made if made else 0.0

        width = shutil.get_terminal_size((100, 24)).columns
        lines = ["", "─" * min(width, 78)]
        lines.append(
            f"  questions {self._bar(self.done_questions, self.target_questions)} "
            f"{self.done_questions:>4}/{self.target_questions}"
        )
        lines.append(
            f"  saved: gullibility low={self.saved['low']:>4}  "
            f"high={self.saved['high']:>4}"
        )
        lines.append(
            f"  calls {self.calls}   failed {self.failures}   "
            f"dropped {self.dropped}   flagged {self.flagged}"
        )
        lines.append(
            f"  elapsed {self._hms(elapsed)}   "
            f"eta {self._hms(eta) if eta is not None else '--:--'}   "
            f"{rate * 60:.1f} q/min"
        )
        lines.append(
            f"  spend ${cost:.2f}   (${cost_per_q:.4f}/question, "
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
        if self.enabled:
            self.render()
            self.lines_drawn = 0
        else:
            print("\n".join(self._render_lines()))


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def call_model(client: OpenAI, prompt: str, max_tokens: int) -> Tuple[str, int, int]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    return text, usage.prompt_tokens, usage.completion_tokens


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def run_one_question(
    client: OpenAI,
    qidx: int,
    question: dict,
    output_dir: str,
    rng: random.Random,
    max_tokens: int,
    seed: int,
    prog: Progress,
) -> int:
    """Issue exactly one call for this question. Returns conversations saved.

    Regardless of outcome, the caller marks this question as processed
    afterwards — a question is sent to the model at most once.
    """
    cat_slug = slugify(question.get("category", "uncategorized"))
    parsed: Optional[BatchResponse] = None
    prompt = ""

    for attempt in range(1, MAX_CALL_RETRIES + 1):
        prompt = build_prompt(question, rng)
        try:
            raw, tok_in, tok_out = call_model(client, prompt, max_tokens)
            prog.in_tokens += tok_in
            prog.out_tokens += tok_out
            prog.calls += 1

            data = extract_json(raw)
            try:
                parsed = BatchResponse.model_validate(data)
            except ValidationError as e:
                raise BatchRejected(f"schema mismatch: {e}") from e

            if not parsed.not_gullible_conversations and not parsed.gullible_conversations:
                raise BatchRejected("no conversations in reply")
            break
        except BatchRejected as e:
            prog.failures += 1
            prog.log(f"  [q{qidx}] attempt {attempt} rejected: {e}")
            parsed = None
        except Exception as e:  # transient API / network errors
            prog.failures += 1
            prog.log(f"  [q{qidx}] attempt {attempt} error: {type(e).__name__}: {e}")
            parsed = None
        if attempt < MAX_CALL_RETRIES:
            time.sleep(2 * attempt)

    if parsed is None:
        prog.log(f"  [q{qidx}] gave up after {MAX_CALL_RETRIES} attempts; "
                  f"question will NOT be retried (marked processed)")
        return 0

    saved = 0
    by_label = {
        "low": parsed.not_gullible_conversations,
        "high": parsed.gullible_conversations,
    }
    for label, convs in by_label.items():
        for n, item in enumerate(convs):
            try:
                validate_conversation(item, label)
            except BatchRejected as e:
                prog.dropped += 1
                prog.log(f"  [q{qidx}/{label}] dropped conversation {n}: {e}")
                continue

            # Reported, never enforced — the conversation is saved either way.
            coverage = topic_coverage(item, question["question"])
            if coverage < LOW_COVERAGE_WARN:
                prog.flagged += 1
                prog.log(
                    f"  [q{qidx}/{label}] conversation {n} shares only "
                    f"{coverage:.0%} of the question's wording — worth a look, "
                    f"kept anyway"
                )

            stem = f"conversation_{qidx:04d}_{n}_{cat_slug}_{ATTRIBUTE}_{label}"
            with open(os.path.join(output_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(conversation_to_text(item))

            meta = {
                "question_index": qidx,
                "question": question["question"],
                "category": question.get("category"),
                "type": question.get("type"),
                "best_answer": question.get("best_answer"),
                "correct_answers": question.get("correct_answers"),
                "incorrect_answers": question.get("incorrect_answers"),
                "attribute": ATTRIBUTE,
                "level": label,
                "model": MODEL_NAME,
                "seed": seed,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "num_turns": len(item.turns),
                "prompt": prompt,
            }
            with open(os.path.join(output_dir, stem + ".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            prog.saved[label] += 1
            saved += 1

    return saved


def generate(
    client: OpenAI,
    questions: List[dict],
    output_dir: str,
    seed: int,
    max_tokens: int,
    limit: Optional[int],
    retry_failed: bool = False,
) -> Progress:
    os.makedirs(output_dir, exist_ok=True)
    processed = load_processed(output_dir)
    failed = load_failed(output_dir)

    # Questions that were attempted but yielded nothing are re-queued only when
    # asked for: they produced no conversations, so re-sending them cannot
    # duplicate data.
    skip = processed - failed if retry_failed else processed

    queue_rng = random.Random(seed)
    queue = build_round_robin_queue(questions, queue_rng)
    queue = [qidx for qidx in queue if qidx not in skip]
    if limit is not None:
        queue = queue[:limit]

    target = len(skip) + len(queue)
    prog = Progress(target, len(skip))
    if processed:
        print(f"Resuming: {len(processed)} question(s) already processed are being kept.")
    if failed:
        print(f"{len(failed)} previously-failed question(s) "
              f"{'re-queued' if retry_failed else 'skipped (use --retry_failed to re-send)'}.")
    prog.render()

    try:
        for qidx in queue:
            question = questions[qidx]
            rng = random.Random(f"{seed}/{qidx}")
            saved = run_one_question(
                client, qidx, question, output_dir, rng, max_tokens, seed, prog
            )
            mark_processed(output_dir, processed, qidx)
            if saved:
                unmark_failed(output_dir, failed, qidx)
            else:
                mark_failed(output_dir, failed, qidx)
            prog.done_questions += 1
            prog.render()
    except KeyboardInterrupt:
        prog.finish()
        print("\nInterrupted. Re-run the same command to resume "
              "(already-attempted questions are skipped).")
        raise

    prog.finish()
    return prog


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT,
        help="Path to the non_hard_negatives-style questions JSON.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated .txt/.json files and the resume manifest.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Base RNG seed for the round-robin category queue and per-call diversity sampling.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=16000,
        help="max_tokens per API call. Raise if batches truncate.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process this many (not-yet-processed) questions this run.",
    )
    parser.add_argument(
        "--retry_failed", action="store_true",
        help=(
            "Re-send questions whose earlier call produced no usable "
            "conversations. They generated nothing, so this cannot duplicate data."
        ),
    )
    args = parser.parse_args()

    api_key = os.getenv("OPUSKEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Set OPUSKEY or ANTHROPIC_API_KEY in your "
            "environment or .env file."
        )

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    questions = load_questions(args.input)

    print(f"Model:       {MODEL_NAME}")
    print(f"Input:       {args.input} ({len(questions)} questions)")
    print(f"Output:      {args.output_dir}/")
    print(f"Per call:    {PER_CALL_PER_LABEL} gullibility=low + {PER_CALL_PER_LABEL} gullibility=high "
          f"(round-robin over categories, one call per question)")

    try:
        prog = generate(client, questions, args.output_dir, args.seed,
                        args.max_tokens, args.limit, args.retry_failed)
    except KeyboardInterrupt:
        return

    print(f"\nDone: {prog.done_questions - prog.start_done_questions} question(s) processed "
          f"this run, {prog.saved['low'] + prog.saved['high']} conversations saved, "
          f"{prog.calls} calls, ${prog.cost():.2f} spent.")


if __name__ == "__main__":
    main()
