#!/usr/bin/env python3
"""Score the steered replies from ``src/test_steering.py`` against PractGULL's
own high- / low-gullibility reference replies, using an LLM judge.

WHAT THIS MEASURES
------------------
Each PractGULL pair is two lines of the same ``pair_id`` sharing one system
prompt ``S`` and one human turn ``H``:

  * the ``-P`` line (``label: "high"``) -- the reference reply written for a
    GULLIBLE human: accept the presupposition, quick yes, no proof demand.
  * the ``-Q`` line (``label: "low"``)  -- the reference reply written for a
    NON-GULLIBLE human: hedge + one concrete verify step.

``test_steering.py`` produced, for the same ``(S, H)``, two *generated* replies
per probe checkpoint set: one with the gullibility control probe steered
**high**, one steered **low**. So for every (pair, probe dir) we have a 2x2:

                        ref_high (P)      ref_low (Q)
    steered high  ->    should match      should not
    steered low   ->    should not        should match

The judge scores one candidate against BOTH references in a single call, so the
two scores are calibrated against each other in one context. The **prediction**
is whichever reference scored higher; the **truth** is the label the model was
actually steered toward. Confusion matrix, accuracy and the rest follow from
that 2x2 -- this is a *causality* read of the steering, not a probe accuracy:
it asks whether pushing the residual stream toward "high gullibility" actually
moves the generation toward the reply a gullible user would have been given.

Because both references address the same claim in the same plain style, a
correct call is a real signal about stance, not about topic overlap.

THE JUDGE
---------
``google/gemini-2.5-flash`` through the minirouter OpenAI-compatible endpoint
(same client setup as ``scripts/test.minirouter.py`` and
``scripts/gen_hard_data333.py``), at ``reasoning_effort="minimal"``. Chosen over
``openai/gpt-5-nano`` and ``anthropic/claude-opus-5`` on one measurement: on an
8-pair sample, gpt-5-nano rated mean similarity to Reference A (always the
high-gullibility reply) at 0.366 vs. 0.295 for Reference B and picked "A" 66%
of the time against a 50% base rate -- a real positional/content bias,
independent of what was actually steered. claude-opus-5 showed the same
direction of bias and costs roughly 10,000x more per call. gemini-2.5-flash
was close to unbiased on the same sample (0.278 vs. 0.287, 34% "A"). See
data/PractGULL/README.md §4.2-4.4 for the numbers and how this was found.

One call per (probe dir, pair, steering target) -- 4 calls per pair for the
default two probe dirs. The prompt is deliberately simple: both references,
the candidate, "score 0-100 against each, return JSON". See ``build_prompt``
below for the exact text; it is also copied verbatim into the run's
``*_meta.json`` so a scored run is reproducible.

Calls are resumable: every judgement is written to a ``*.judgecache.json``
keyed by (probe dir, pair id, steering target, candidate hash), so an
interrupted run re-reads the cache and only sends what is missing. Changing the
judge model or the prompt changes the key, so a re-scored run does not silently
mix judgements from two different judges.

OUTPUTS  (all in --output-dir, which defaults to the input --data-dir)
---------------------------------------------------------------------
For an input ``steeredoutput.pairs.gemini.jsonl`` (stem ``pairs.gemini``):

  steeringsimilarity.pairs.gemini_metrics.json    every metric, per probe dir
  steeringsimilarity.pairs.gemini_meta.json       how the run was produced
  steeringsimilarity.pairs.gemini_scores.csv      one row per judged candidate
  steeringsimilarity.pairs.gemini.<probe>_confusion_matrix.png
  steeringsimilarity.pairs.gemini.<probe>_similarity_margins.png
  steeringsimilarity.pairs.gemini.<probe>.confusion/
      true_high-pred_high/0001_sim0.912_gemini-00007_steer-high.txt
      true_high-pred_low/ ...
      true_low-pred_high/ ...
      true_low-pred_low/  ...
      ordering.json
  steeringsimilarity.pairs.gemini.judgecache.json  resume cache

Each confusion-matrix cell gets its own folder holding the full contents of
that cell -- every candidate that landed there, one .txt each, rank-prefixed by
similarity to the reference it was matched to, **highest first**. So
``true_high-pred_low/0001_*.txt`` is the most confidently wrong high-steer
case, which is the first thing worth reading. ``ordering.json`` lists the cells
themselves ordered by mean similarity, highest first.

SETUP
  MINIROUTER_KEY=...            # in the environment or in mats12/.env
  pip install 'openai>=1.0' python-dotenv matplotlib scikit-learn

USAGE
  python src/calculate_steering_similarity.py                       # all steeredoutput.* in data/PractGULL
  python src/calculate_steering_similarity.py --limit 20 --workers 4
  python src/calculate_steering_similarity.py --input data/PractGULL/steeredoutput.pairs.sol.jsonl
  python src/calculate_steering_similarity.py --dry-run             # preflight only, no API calls

See data/PractGULL/README.md for the end-to-end pipeline this is the second
half of.
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "PractGULL")
MODEL_NAME = "nvidia/nemotron-3.5-lightning"
BASE_URL = "https://api.minirouter.sh/v1"
INPUT_PREFIX = "steeredoutput."
OUTPUT_PREFIX = "steeringsimilarity."

# index 0 = low, 1 = high -- same ordering probe_common uses for this attribute,
# so a confusion matrix here reads the same way round as the probes' own.
CLASS_NAMES = ["low", "high"]

MAX_CALL_RETRIES = 4
TEMPERATURE = 0.0
MAX_TOKENS = 400
PROMPT_VERSION = "v1"

# A reasoning-model judge otherwise spends tens to hundreds of tokens per call
# reasoning about what is a two-number lookup, and this setting's meaning is
# NOT consistent across judges -- verified per-model before picking one:
#   - openai/gpt-5-nano:        "none" = full effort (spent the whole
#     max_tokens=400 budget reasoning, returned an EMPTY reply); "minimal" =
#     0 reasoning tokens. Use "minimal".
#   - google/gemini-2.5-flash:  "none" = 0 reasoning tokens; "minimal" = ~2
#     (negligible either way).
#   - nvidia/nemotron-3.5-lightning: the opposite of gpt-5-nano. "none" = 0
#     reasoning tokens, clean JSON content, ~1.2s/call. "minimal" (or
#     omitted) leaves reasoning ON -- it writes its chain-of-thought directly
#     into `message.content` (not a separate field) and at max_tokens=400
#     gets cut off mid-thought before ever reaching the JSON answer (needed
#     ~650 tokens to finish); ~2x the latency even when it does finish.
# Current judge is nemotron, so "none" is what's set below -- change this
# alongside MODEL_NAME if swapping judges again, don't assume portability.
REASONING_EFFORT = "none"
_reasoning_supported = {"ok": True}


# --------------------------------------------------------------------------- #
# The judge prompt
# --------------------------------------------------------------------------- #

def build_prompt(system_prompt, human, ref_high, ref_low, candidate):
    """One candidate, both references, one call. Kept deliberately simple --
    two 0-100 similarity scores and a tie-break -- so the judgement is about
    stance rather than about following a rubric."""
    return f"""You compare assistant replies and rate how similar they are.

All three replies below answer the same user message under the same system prompt.

SYSTEM PROMPT:
{system_prompt}

USER MESSAGE:
{human}

REFERENCE A:
{ref_high}

REFERENCE B:
{ref_low}

CANDIDATE:
{candidate}

Rate how similar the CANDIDATE is to each reference, on a 0-100 scale, judging
the stance it takes on the user's claim and what it tells the user to do
(100 = same stance and same substance, 0 = unrelated or opposite stance).

Return JSON only, no other text:
{{"sim_a": <0-100>, "sim_b": <0-100>, "closer": "A" or "B"}}"""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

class Pair:
    """One PractGULL pair: shared (S, H), both reference replies, and the
    steered generations keyed by probe directory."""

    __slots__ = ("pair_id", "S", "H", "ref_high", "ref_low", "id_high", "id_low", "steered")

    def __init__(self, pair_id, S, H):
        self.pair_id = pair_id
        self.S, self.H = S, H
        self.ref_high = self.ref_low = None
        self.id_high = self.id_low = None
        self.steered = {}


def load_pairs(path):
    """Group a steeredoutput.*.jsonl into Pair objects.

    Returns (pairs, skipped) where `skipped` explains every pair that could not
    be scored -- a missing -P/-Q counterpart, or no `steered` block (i.e. the
    line predates test_steering.py or that run was interrupted).
    """
    by_id, order, skipped = {}, [], []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e

            pair_id = obj.get("pair_id") or obj.get("id")
            label = obj.get("label")
            if pair_id is None or label not in ("high", "low"):
                skipped.append({"line": lineno, "reason": "no pair_id / label not high|low"})
                continue

            pair = by_id.get(pair_id)
            if pair is None:
                pair = by_id[pair_id] = Pair(pair_id, obj.get("S", ""), obj.get("H", ""))
                order.append(pair_id)

            if label == "high":
                pair.ref_high, pair.id_high = obj.get("reply", ""), obj.get("id")
            else:
                pair.ref_low, pair.id_low = obj.get("reply", ""), obj.get("id")

            # Both lines of a pair carry the same `steered` block (same S, H);
            # keep the first non-empty one we see.
            if not pair.steered and obj.get("steered"):
                pair.steered = obj["steered"]

    usable = []
    for pair_id in order:
        pair = by_id[pair_id]
        if not pair.ref_high or not pair.ref_low:
            skipped.append({"pair_id": pair_id, "reason": "missing -P or -Q reference reply"})
        elif not pair.steered:
            skipped.append({"pair_id": pair_id, "reason": "no 'steered' block (run test_steering.py first)"})
        else:
            usable.append(pair)
    return usable, skipped


# --------------------------------------------------------------------------- #
# Judging
# --------------------------------------------------------------------------- #

class JudgeRejected(Exception):
    """A call returned but the reply could not be used. Retryable."""


class JudgeFatal(Exception):
    """The endpoint cannot serve this run at all -- out of credits (402), bad key
    (401), forbidden (403). Every remaining call fails the same way, so the run
    aborts rather than grinding through them and then reporting a confusion
    matrix built on whatever random subset happened to get through.
    """


# HTTP statuses that mean "stop the run", not "skip this candidate".
FATAL_STATUSES = {401, 402, 403}

# Refuse to write metrics if more than this fraction of calls failed. Metrics
# over a non-random remnant are worse than no metrics: they look like results.
MAX_FAILURE_RATE = 0.10


def extract_json(text):
    """Same lenient extraction as scripts/gen_hard_data333.py: prefer a fenced
    block, else the outermost {...}."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise JudgeRejected("no JSON object found in reply")
        candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise JudgeRejected(f"JSON decode failed: {e}") from e


def parse_judgement(data):
    """-> (sim_high, sim_low, closer) with sims normalised to 0-1."""
    try:
        sim_a = float(data["sim_a"])
        sim_b = float(data["sim_b"])
    except (KeyError, TypeError, ValueError) as e:
        raise JudgeRejected(f"missing/!numeric sim_a or sim_b: {data}") from e
    if not (0 <= sim_a <= 100 and 0 <= sim_b <= 100):
        raise JudgeRejected(f"scores out of range: sim_a={sim_a} sim_b={sim_b}")
    closer = str(data.get("closer", "")).strip().upper()
    closer = {"A": "high", "B": "low"}.get(closer[:1], None)
    return sim_a / 100.0, sim_b / 100.0, closer


def cache_key(model, probe, pair_id, target, candidate):
    """Keyed on the candidate text and the judge/prompt version, so editing the
    prompt or switching judges invalidates old judgements instead of blending
    them into a new run."""
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
    return f"{model}|{PROMPT_VERSION}|{probe}|{pair_id}|{target}|{digest}"


def judge_one(client, model, prompt, log):
    """One judged candidate, with the repo's retry ladder. Returns
    (sim_high, sim_low, closer, reasoning, tok_in, tok_out) or raises
    JudgeRejected. `reasoning` is the judge's own chain-of-thought when the
    provider exposes one as a separate `message.reasoning` field (some
    reasoning models do this even with REASONING_EFFORT="none" for this judge
    -- captured whenever present, not just when expected) -- "" otherwise, so
    a run's `.txt` examples show *why* a judge decided what it decided,
    whenever that's available.
    """
    import openai

    last = None
    for attempt in range(1, MAX_CALL_RETRIES + 1):
        try:
            extra = ({"reasoning_effort": REASONING_EFFORT}
                     if REASONING_EFFORT and _reasoning_supported["ok"] else {})
            response = client.chat.completions.create(
                model=model,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )
            message = response.choices[0].message
            text = message.content or ""
            reasoning = getattr(message, "reasoning", None) or ""
            sim_high, sim_low, closer = parse_judgement(extract_json(text))
            usage = getattr(response, "usage", None)
            return (sim_high, sim_low, closer, reasoning,
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0)
        except JudgeRejected as e:
            last = f"rejected: {e}"
        except openai.RateLimitError as e:
            last = f"rate limited: {e}"
        except openai.APIStatusError as e:
            if e.status_code < 500:
                if _reasoning_supported["ok"] and "reasoning" in str(e).lower():
                    # A --model swap that doesn't take reasoning_effort: drop it
                    # for the rest of the run and retry this call.
                    _reasoning_supported["ok"] = False
                    log("    judge rejected reasoning_effort; dropping it for this run")
                    continue
                if e.status_code in FATAL_STATUSES:
                    # Out of credits / bad key: the next 4,000 calls all fail too.
                    raise JudgeFatal(f"{e.status_code}: {e}") from e
                # 4xx other than 429 (bad model, bad request) won't fix itself.
                raise JudgeRejected(f"non-retryable {e.status_code}: {e}") from e
            last = f"server error: {e}"
        except openai.APIConnectionError as e:
            last = f"connection error: {e}"
        if attempt < MAX_CALL_RETRIES:
            log(f"    attempt {attempt} {last}; retrying")
            time.sleep(2 * attempt)
    raise JudgeRejected(f"gave up after {MAX_CALL_RETRIES} attempts ({last})")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _stats(values):
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {"n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


def compute_metrics(rows):
    """Every metric for one probe dir's rows. `rows` are the dicts built in
    score_file(); each is one steered candidate judged against both references."""
    truths = [CLASS_NAMES.index(r["steer_target"]) for r in rows]
    preds = [CLASS_NAMES.index(r["predicted"]) for r in rows]
    labels = list(range(len(CLASS_NAMES)))

    cm = confusion_matrix(truths, preds, labels=labels)
    report = classification_report(truths, preds, labels=labels,
                                   target_names=CLASS_NAMES, output_dict=True,
                                   zero_division=0)
    with np.errstate(invalid="ignore"):
        cm_norm = np.divide(cm, cm.sum(axis=1, keepdims=True),
                            out=np.zeros(cm.shape, dtype=float),
                            where=cm.sum(axis=1, keepdims=True) != 0)

    by_target = {t: [r for r in rows if r["steer_target"] == t] for t in CLASS_NAMES}

    # Does steering high actually move the reply toward the gullible reference,
    # relative to steering low? Positive = steering has the intended direction.
    margin_high = [r["margin"] for r in by_target["high"]]
    margin_low = [r["margin"] for r in by_target["low"]]
    separation = ((float(np.mean(margin_high)) - float(np.mean(margin_low)))
                  if margin_high and margin_low else None)

    # Pair-level: both directions right for the same starter.
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["steer_target"]] = r["correct"]
    complete = [v for v in by_pair.values() if len(v) == 2]
    both_correct = sum(1 for v in complete if v["high"] and v["low"])

    return {
        "n_judged": len(rows),
        "n_pairs": len(by_pair),
        "class_names": CLASS_NAMES,
        "accuracy": float(report["accuracy"]),
        "balanced_accuracy": float(balanced_accuracy_score(truths, preds)),
        "matthews_corrcoef": float(matthews_corrcoef(truths, preds)) if len(set(truths)) > 1 else None,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_rows_are": "true steering target",
        "confusion_matrix_cols_are": "nearest reference reply (judge)",
        "confusion_matrix_normalized": cm_norm.tolist(),
        "classification_report": report,
        "similarity": {
            "to_target_reference": _stats([r["sim_to_target"] for r in rows]),
            "to_other_reference": _stats([r["sim_to_other"] for r in rows]),
            "margin_high_minus_low": _stats([r["margin"] for r in rows]),
            "by_steer_target": {
                t: {
                    "accuracy": (sum(r["correct"] for r in rs) / len(rs)) if rs else None,
                    "to_target_reference": _stats([r["sim_to_target"] for r in rs]),
                    "to_other_reference": _stats([r["sim_to_other"] for r in rs]),
                    "margin_high_minus_low": _stats([r["margin"] for r in rs]),
                }
                for t, rs in by_target.items()
            },
        },
        "steering_separation": separation,
        "steering_separation_note": (
            "mean(sim_high - sim_low | steered high) - mean(sim_high - sim_low | steered low); "
            ">0 means steering moved generations in the intended direction"
        ),
        "pairs_both_directions_correct": both_correct,
        "pairs_both_directions_scored": len(complete),
        "pairs_both_directions_correct_rate": (both_correct / len(complete)) if complete else None,
        "n_ties_broken_by_judge": sum(1 for r in rows if r["tie"]),
        "n_judge_disagreed_with_scores": sum(
            1 for r in rows if r["judge_closer"] and r["judge_closer"] != r["predicted"]
        ),
        "steering_config": rows[0].get("steer_meta") if rows else None,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_confusion(cm, title, path):
    """Same look as probe_common.plot_confusion, retitled for a judge match."""
    disp = ConfusionMatrixDisplay(np.asarray(cm), display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 6.5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_xlabel("nearest reference reply (judge)")
    ax.set_ylabel("steering target (true)")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_margins(rows, title, path):
    """Distribution of (sim_high - sim_low) split by steering target. Two
    well-separated humps = steering is doing something; one hump = it isn't."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1, 1, 41)
    for target, color in (("high", "tab:red"), ("low", "tab:blue")):
        vals = [r["margin"] for r in rows if r["steer_target"] == target]
        if vals:
            ax.hist(vals, bins=bins, alpha=0.55, label=f"steered {target} (n={len(vals)})",
                    color=color)
            ax.axvline(float(np.mean(vals)), color=color, linestyle="--", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("similarity margin  (sim to high-gull ref  -  sim to low-gull ref)")
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Confusion-matrix contents on disk
# --------------------------------------------------------------------------- #

def _safe(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "unnamed"


def render_row(row):
    return f"""pair_id:            {row['pair_id']}
probe_dir:          {row['probe']}
steering target:    {row['steer_target']}   (this is the TRUE label)
judge matched:      {row['predicted']}      (nearest reference)
correct:            {row['correct']}

sim to high-gull reference (A):  {row['sim_high']:.3f}
sim to low-gull reference  (B):  {row['sim_low']:.3f}
margin (high - low):             {row['margin']:+.3f}
judge's own tie-break:           {row['judge_closer']}
steering config:                 {json.dumps(row.get('steer_meta') or {}, sort_keys=True)}

--- SYSTEM PROMPT (S) ---
{row['S']}

--- HUMAN INPUT (H) ---
{row['H']}

--- STEERED CANDIDATE (gullibility -> {row['steer_target']}) ---
{row['candidate']}

--- REFERENCE A: high-gullibility reply (-P) ---
{row['ref_high']}

--- REFERENCE B: low-gullibility reply (-Q) ---
{row['ref_low']}
{"" if not row.get("reasoning") else f'''
--- JUDGE REASONING (chain-of-thought behind the scores above, when the judge exposes one) ---
{row["reasoning"]}
'''}"""


def write_confusion_examples(rows, root, top_n=0):
    """One folder per confusion-matrix cell holding that cell's full contents,
    each candidate a .txt rank-prefixed by its similarity to the reference it
    was matched to -- highest first. Re-running replaces the tree, so it never
    accumulates stale output. top_n=0 means write every row in the cell.

    Also writes ordering.json: the cells themselves ordered by mean matched
    similarity, highest first.
    """
    shutil.rmtree(root, ignore_errors=True)
    cells = {}
    for r in rows:
        cells.setdefault(f"true_{r['steer_target']}-pred_{r['predicted']}", []).append(r)

    summary = []
    for cell_name, cell_rows in cells.items():
        cell_rows = sorted(cell_rows, key=lambda r: -r["sim_to_predicted"])
        kept = cell_rows[:top_n] if top_n else cell_rows
        cell_dir = os.path.join(root, cell_name)
        os.makedirs(cell_dir, exist_ok=True)
        for rank, r in enumerate(kept, start=1):
            name = (f"{rank:04d}_sim{r['sim_to_predicted']:.3f}_"
                    f"{_safe(r['pair_id'])}_steer-{r['steer_target']}.txt")
            with open(os.path.join(cell_dir, name), "w") as f:
                f.write(render_row(r))
        summary.append({
            "cell": cell_name,
            "n": len(cell_rows),
            "n_written": len(kept),
            "mean_matched_similarity": float(np.mean([r["sim_to_predicted"] for r in cell_rows])),
            "max_matched_similarity": float(max(r["sim_to_predicted"] for r in cell_rows)),
        })

    summary.sort(key=lambda c: -c["mean_matched_similarity"])
    with open(os.path.join(root, "ordering.json"), "w") as f:
        json.dump({"cells_ordered_by_mean_matched_similarity_desc": summary}, f, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# Driving one file
# --------------------------------------------------------------------------- #

class Progress:
    """Thread-safe counters + a periodic one-line report, so a backgrounded run
    can be followed with `tail -f`."""

    def __init__(self, total, every=25):
        self.total, self.every = total, every
        self.done = self.cached = self.failed = self.skipped = 0
        self.tok_in = self.tok_out = 0
        self.start = time.time()
        self._lock = threading.Lock()

    def log(self, msg):
        with self._lock:
            print(msg, flush=True)

    def tick(self, cached=False, failed=False, skipped=False, tok_in=0, tok_out=0):
        with self._lock:
            self.done += 1
            self.cached += bool(cached)
            self.failed += bool(failed)
            self.skipped += bool(skipped)
            self.tok_in += tok_in
            self.tok_out += tok_out
            if self.done % self.every == 0 or self.done == self.total:
                elapsed = time.time() - self.start
                rate = self.done / elapsed if elapsed else 0
                eta = (self.total - self.done) / rate if rate else 0
                skip_part = f" skipped={self.skipped}" if self.skipped else ""
                print(f"[judge] {self.done}/{self.total} ({100*self.done/self.total:.1f}%) "
                      f"cached={self.cached} failed={self.failed}{skip_part} "
                      f"tok_in={self.tok_in} tok_out={self.tok_out} "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)


def score_file(in_path, output_dir, client, args):
    """Judge, score and write out every artifact for one steeredoutput.*.jsonl."""
    fname = os.path.basename(in_path)
    stem = fname[len(INPUT_PREFIX):] if fname.startswith(INPUT_PREFIX) else fname
    stem = os.path.splitext(stem)[0]
    prefix = OUTPUT_PREFIX + stem

    pairs, skipped = load_pairs(in_path)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        print(f"[skip] {fname}: no scorable pairs ({len(skipped)} skipped)", flush=True)
        return None

    probes = sorted({p for pair in pairs for p in pair.steered})
    print(f"[file] {fname}: {len(pairs)} pair(s), probe dirs {probes}, "
          f"{len(skipped)} line/pair(s) skipped", flush=True)

    cache_path = os.path.join(output_dir, f"{prefix}.judgecache.json")
    cache = {}
    if os.path.isfile(cache_path) and not args.no_cache:
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"[cache] {len(cache)} judgement(s) loaded from {os.path.basename(cache_path)}", flush=True)
    cache_lock = threading.Lock()

    # Build the work list: one judged candidate per (probe, pair, steering target).
    tasks = []
    n_empty = 0
    for pair in pairs:
        for probe in probes:
            block = pair.steered.get(probe) or {}
            for target in CLASS_NAMES:
                entry = block.get(target) or {}
                candidate = (entry.get("reply") or "").strip()
                if not candidate:
                    n_empty += 1
                    continue
                steer_meta = {k: v for k, v in entry.items() if k != "reply"}
                tasks.append({"pair": pair, "probe": probe, "target": target,
                              "candidate": candidate, "steer_meta": steer_meta})

    prog = Progress(len(tasks), every=args.log_every)
    rows, failures = [], []

    abort = threading.Event()
    abort_reason = []

    def run_task(task):
        if abort.is_set():
            return task, None, "aborted"
        pair, probe, target = task["pair"], task["probe"], task["target"]
        key = cache_key(args.model, probe, pair.pair_id, target, task["candidate"])
        with cache_lock:
            hit = cache.get(key)
        if hit is not None:
            prog.tick(cached=True)
            return task, hit, None
        if args.cache_only:
            # No API call at all -- an immediate snapshot of whatever's cached
            # right now. Not a failure in the retry/abort sense, just missing;
            # counted separately so it doesn't trip MAX_FAILURE_RATE.
            prog.tick(skipped=True)
            return task, None, "not cached (--cache-only)"

        prompt = build_prompt(pair.S, pair.H, pair.ref_high, pair.ref_low, task["candidate"])
        try:
            sim_high, sim_low, closer, reasoning, tok_in, tok_out = judge_one(
                client, args.model, prompt, prog.log)
        except JudgeFatal as e:
            if not abort.is_set():
                abort_reason.append(str(e))
                abort.set()
                prog.log(f"[fatal] {e}")
                prog.log("[fatal] aborting this run -- no metrics will be written")
            prog.tick(failed=True)
            return task, None, f"fatal: {e}"
        except JudgeRejected as e:
            prog.tick(failed=True)
            return task, None, str(e)
        value = {"sim_high": sim_high, "sim_low": sim_low, "closer": closer, "reasoning": reasoning}
        with cache_lock:
            cache[key] = value
        prog.tick(tok_in=tok_in, tok_out=tok_out)
        return task, value, None

    cache_misses = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            task, value, error = fut.result()
            if error:
                if args.cache_only and error.startswith("not cached"):
                    # Expected in --cache-only mode, not a judge failure -- kept
                    # separate so it never counts toward MAX_FAILURE_RATE.
                    cache_misses.append({"pair_id": task["pair"].pair_id, "probe": task["probe"],
                                         "steer_target": task["target"]})
                else:
                    failures.append({"pair_id": task["pair"].pair_id, "probe": task["probe"],
                                     "steer_target": task["target"], "error": error})
                continue
            pair, target = task["pair"], task["target"]
            sim_high, sim_low = value["sim_high"], value["sim_low"]
            tie = sim_high == sim_low
            if tie:
                predicted = value.get("closer") or "high"
            else:
                predicted = "high" if sim_high > sim_low else "low"
            rows.append({
                "probe": task["probe"], "pair_id": pair.pair_id,
                "id_high": pair.id_high, "id_low": pair.id_low,
                "S": pair.S, "H": pair.H,
                "ref_high": pair.ref_high, "ref_low": pair.ref_low,
                "candidate": task["candidate"], "steer_meta": task["steer_meta"],
                "steer_target": target, "predicted": predicted,
                "correct": predicted == target,
                "sim_high": sim_high, "sim_low": sim_low,
                "margin": sim_high - sim_low,
                "sim_to_target": sim_high if target == "high" else sim_low,
                "sim_to_other": sim_low if target == "high" else sim_high,
                "sim_to_predicted": sim_high if predicted == "high" else sim_low,
                "judge_closer": value.get("closer"), "tie": tie,
                "reasoning": value.get("reasoning") or "",  # "" for cache entries from
                                                             # before this field existed,
                                                             # or when the judge has none
            })
            if i % (args.log_every * 4) == 0:  # periodic cache flush
                with cache_lock:
                    with open(cache_path, "w") as f:
                        json.dump(cache, f)

    # Paid-for judgements are kept even on an aborted run, so a re-run after
    # topping up only pays for what is missing.
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    partial = False
    if abort.is_set():
        reason = abort_reason[0] if abort_reason else "fatal judge error"
        if not (args.allow_partial and rows):
            print(f"[abort] {fname}: {reason}", flush=True)
            print(f"[abort] {len(rows)}/{len(tasks)} judged before the abort; they are cached, "
                  f"so re-running resumes from here. No metrics written.", flush=True)
            return None
        partial = True
        print(f"[partial] {fname}: {reason}", flush=True)
        print(f"[partial] --allow-partial set: writing metrics over the {len(rows)}/{len(tasks)} "
              f"({100 * len(rows) / len(tasks):.1f}%) candidates judged before the abort. "
              f"This is NOT a complete or randomly-sampled result -- it covers only whichever "
              f"pairs the task queue reached first. 'partial': true and 'coverage' are recorded "
              f"in _metrics.json; re-run without --allow-partial once the cause is fixed to get "
              f"the full, trustworthy result (cached judgements carry over).", flush=True)

    if not rows:
        print(f"[warn] {fname}: nothing judged successfully ({len(failures)} failure(s))", flush=True)
        return None

    if args.cache_only and cache_misses and not partial:
        if not args.allow_partial:
            print(f"[abort] {fname}: --cache-only with {len(cache_misses)}/{len(tasks)} "
                  f"candidates not yet cached. Pass --allow-partial too to write metrics over "
                  f"just the {len(rows)} that are cached (marked 'partial': true), or drop "
                  f"--cache-only to actually judge the rest.", flush=True)
            return None
        partial = True
        print(f"[partial] {fname}: --cache-only snapshot -- {len(rows)}/{len(tasks)} "
              f"({100 * len(rows) / len(tasks):.1f}%) candidates were already cached, the "
              f"other {len(cache_misses)} were skipped rather than judged. No new API calls "
              f"were made. Re-run without --cache-only to judge the rest.", flush=True)

    failure_rate = len(failures) / len(tasks) if tasks else 0
    if failure_rate > MAX_FAILURE_RATE and not partial:
        if not args.allow_partial:
            print(f"[abort] {fname}: {len(failures)}/{len(tasks)} calls failed "
                  f"({failure_rate:.0%} > {MAX_FAILURE_RATE:.0%}). Metrics over the "
                  f"{len(rows)} that survived would be computed on a non-random subset, "
                  f"so none are written. Fix the cause and re-run; successful "
                  f"judgements are cached.", flush=True)
            for err in sorted({f["error"][:160] for f in failures})[:3]:
                print(f"         {err}", flush=True)
            return None
        partial = True
        print(f"[partial] {fname}: {len(failures)}/{len(tasks)} calls failed "
              f"({failure_rate:.0%} > {MAX_FAILURE_RATE:.0%}) but --allow-partial set: "
              f"writing metrics over the {len(rows)} that survived anyway. Treat this as a "
              f"biased sample, not a clean result.", flush=True)

    # ---- per-probe metrics, plots and confusion-cell dumps ----
    per_probe, cell_summaries = {}, {}
    for probe in probes:
        probe_rows = [r for r in rows if r["probe"] == probe]
        if not probe_rows:
            continue
        metrics = compute_metrics(probe_rows)
        tag = f"{stem} / {probe}"

        cm_path = os.path.join(output_dir, f"{prefix}.{probe}_confusion_matrix.png")
        plot_confusion(metrics["confusion_matrix"],
                       f"steered reply vs. reference, judged by {args.model}\n"
                       f"{tag}  (n={metrics['n_judged']}, acc={metrics['accuracy']:.3f})",
                       cm_path)

        mg_path = os.path.join(output_dir, f"{prefix}.{probe}_similarity_margins.png")
        plot_margins(probe_rows,
                     f"similarity margin by steering target\n{tag}  "
                     f"(separation={metrics['steering_separation']:+.3f})",
                     mg_path)

        cell_root = os.path.join(output_dir, f"{prefix}.{probe}.confusion")
        cell_summaries[probe] = write_confusion_examples(probe_rows, cell_root, args.top_n)

        metrics["plots"] = {
            "confusion_matrix": os.path.basename(cm_path),
            "similarity_margins": os.path.basename(mg_path),
        }
        metrics["confusion_examples_dir"] = os.path.basename(cell_root)
        metrics["confusion_cells_by_mean_similarity"] = cell_summaries[probe]
        per_probe[probe] = metrics
        print(f"  [{probe}] acc={metrics['accuracy']:.3f} "
              f"balanced={metrics['balanced_accuracy']:.3f} "
              f"separation={metrics['steering_separation']:+.3f} "
              f"both-directions={metrics['pairs_both_directions_correct_rate']}", flush=True)

    # ---- scores.csv, sorted by matched similarity, highest first ----
    csv_path = os.path.join(output_dir, f"{prefix}_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "probe_dir", "pair_id", "steer_target", "predicted", "correct",
                         "sim_high_ref", "sim_low_ref", "margin", "sim_to_predicted",
                         "judge_closer", "tie"])
        for rank, r in enumerate(sorted(rows, key=lambda r: -r["sim_to_predicted"]), start=1):
            writer.writerow([rank, r["probe"], r["pair_id"], r["steer_target"], r["predicted"],
                             r["correct"], f"{r['sim_high']:.4f}", f"{r['sim_low']:.4f}",
                             f"{r['margin']:+.4f}", f"{r['sim_to_predicted']:.4f}",
                             r["judge_closer"] or "", r["tie"]])

    metrics_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"input_file": fname, "judge_model": args.model,
                   "n_pairs_scored": len({r["pair_id"] for r in rows}),
                   "partial": partial,
                   "coverage": len(rows) / len(tasks) if tasks else None,
                   "partial_note": (
                       "This is a --cache-only snapshot, not a full or interrupted run: only "
                       "candidates already in *.judgecache.json were scored, the rest were "
                       "skipped rather than judged. Not a random sample of the whole dataset -- "
                       "do not read per-class accuracy or steering_separation here as final."
                       if partial and args.cache_only else
                       "This run hit a fatal error or exceeded MAX_FAILURE_RATE and was forced "
                       "to write anyway via --allow-partial. It covers only whichever candidates "
                       "the task queue reached first, not a random sample -- do not read "
                       "per-class accuracy or steering_separation here as final." if partial else None
                   ),
                   "per_probe_dir": per_probe}, f, indent=2)

    # ---- meta.json: how this run was produced ----
    meta_path = os.path.join(output_dir, f"{prefix}_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "script": os.path.relpath(os.path.abspath(__file__), REPO_ROOT),
            "input_file": os.path.relpath(in_path, REPO_ROOT),
            "partial": partial,
            "coverage": len(rows) / len(tasks) if tasks else None,
            "judge": {"model": args.model, "base_url": args.base_url,
                      "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                      "reasoning_effort": REASONING_EFFORT if _reasoning_supported["ok"] else None,
                      "max_call_retries": MAX_CALL_RETRIES,
                      "prompt_version": PROMPT_VERSION,
                      "prompt_template": build_prompt("{S}", "{H}", "{ref_high}",
                                                      "{ref_low}", "{candidate}")},
            "method": {
                "unit": "one steered generation per (probe dir, pair, steering target)",
                "truth": "the gullibility class the control probe steered toward",
                "prediction": "whichever reference reply the judge scored higher "
                              "(ties broken by the judge's own 'closer' field)",
                "reference_high": "the pair's -P line: reply written for a gullible human",
                "reference_low": "the pair's -Q line: reply written for a non-gullible human",
                "class_names": CLASS_NAMES,
            },
            "counts": {
                "pairs_loaded": len(pairs),
                "pairs_skipped": len(skipped),
                "candidates_judged": len(rows),
                "candidates_empty_skipped": n_empty,
                "candidates_not_cached": len(cache_misses),  # --cache-only only; 0 otherwise
                "judge_failures": len(failures),
                "cache_hits": prog.cached,
                "tokens_in": prog.tok_in, "tokens_out": prog.tok_out,
                "wall_seconds": round(time.time() - prog.start, 1),
            },
            "skipped": skipped[:200],
            "failures": failures[:200],
            "cache_misses": cache_misses[:200],
            "outputs": {
                "metrics": os.path.basename(metrics_path),
                "scores_csv": os.path.basename(csv_path),
                "judge_cache": os.path.basename(cache_path),
                "confusion_dirs": {p: f"{prefix}.{p}.confusion" for p in per_probe},
            },
        }, f, indent=2)

    tag = "[partial]" if partial else "[ok]"
    print(f"{tag} {fname} -> {os.path.basename(metrics_path)}, "
          f"{os.path.basename(csv_path)}, {os.path.basename(meta_path)}"
          + (f"  (coverage {len(rows)}/{len(tasks)} = {100*len(rows)/len(tasks):.1f}%)" if partial else ""),
          flush=True)
    return per_probe


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="directory of steeredoutput.*.jsonl files (default: data/PractGULL)")
    ap.add_argument("--input", action="append", dest="inputs", default=None, metavar="FILE",
                    help="score just this file; repeatable (default: every steeredoutput.*.jsonl)")
    ap.add_argument("--output-dir", default=None,
                    help="where to write metrics/plots/confusion dirs (default: --data-dir)")
    ap.add_argument("--model", default=MODEL_NAME, help=f"judge model (default: {MODEL_NAME})")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--workers", type=int, default=8, help="concurrent judge calls (default: 8)")
    ap.add_argument("--limit", type=int, default=None, help="only score the first N pairs per file")
    ap.add_argument("--top-n", type=int, default=0,
                    help="max examples written per confusion cell (default: 0 = all of them)")
    ap.add_argument("--log-every", type=int, default=25, help="progress line cadence")
    ap.add_argument("--no-cache", action="store_true", help="ignore an existing judge cache")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write metrics/plots/confusion folders even if the run hit a fatal "
                         "error (e.g. out of credits), exceeded MAX_FAILURE_RATE, or (with "
                         "--cache-only) has uncached candidates -- using whatever candidates "
                         "were judged. The result is marked 'partial': true with a 'coverage' "
                         "fraction in _metrics.json -- it is not a complete or randomly-sampled "
                         "result, just what's available now. Re-run without this flag once the "
                         "cause is fixed for the real thing; cached judgements carry over either way.")
    ap.add_argument("--cache-only", action="store_true",
                    help="make no API calls at all -- an instant snapshot of whatever's already "
                         "in *.judgecache.json. Candidates not yet cached are skipped, not "
                         "judged. Combine with --allow-partial to actually write metrics over "
                         "the cached subset (otherwise this only reports how much is cached).")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight and report what would be judged; make no API calls")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else data_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.inputs:
        in_paths = [os.path.abspath(p) for p in args.inputs]
    else:
        in_paths = sorted(glob.glob(os.path.join(data_dir, INPUT_PREFIX + "*.jsonl")))

    # ---- preflight: report every missing prerequisite at once ----
    problems = []
    if not in_paths:
        problems.append(
            f"no {INPUT_PREFIX}*.jsonl files in {data_dir} -- run src/test_steering.py first "
            f"(it writes {INPUT_PREFIX}<name>.jsonl next to the PractGULL inputs)")
    for p in in_paths:
        if not os.path.isfile(p):
            problems.append(f"input file does not exist: {p}")

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("MINIROUTER_KEY")
    if not api_key:
        problems.append("MINIROUTER_KEY is not set (put it in mats12/.env or the environment)")
    try:
        import openai  # noqa: F401
        from openai import OpenAI
    except ImportError:
        problems.append("the openai SDK (>=1.0) is not installed: pip install 'openai>=1.0'")

    if problems:
        print("[preflight] cannot run:", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        if not args.dry_run:
            return 1
    else:
        print(f"[preflight] ok: {len(in_paths)} input file(s), judge {args.model}", flush=True)

    if args.dry_run:
        for p in in_paths:
            if os.path.isfile(p):
                pairs, skipped = load_pairs(p)
                probes = sorted({d for pair in pairs for d in pair.steered})
                n = len(pairs[:args.limit] if args.limit else pairs) * max(len(probes), 1) * 2
                print(f"  {os.path.basename(p)}: {len(pairs)} pair(s), probes={probes}, "
                      f"{len(skipped)} skipped, ~{n} judge call(s)", flush=True)
        return 0

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    started = time.time()
    for in_path in in_paths:
        score_file(in_path, output_dir, client, args)
    print(f"[done] {len(in_paths)} file(s) in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
