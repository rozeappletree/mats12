"""
Build the "high-value fine-tuning candidates" subset from the hard-negative sample.

The cut is the one recommended in `nb/eda_hard_negatives_333.ipynb` (section 15,
"Suggested downstream cuts"):

    3-way core  AND  min similarity >= 0.60  AND  no refusal in any condition

Each filter removes a distinct kind of weak evidence:

  - *3-way core* -- the question was flagged by all three persona notebooks, so it
    fails under all nine (notebook x condition) combinations. Single-notebook rows
    may reflect one notebook's prompt wording or one generation run rather than a
    stable knowledge gap; the core survives that noise.
  - *min similarity >= 0.60* -- every one of the nine responses is a close paraphrase
    of a specific known misconception, not a vague response that the nearest-neighbour
    matcher was forced to attach to something.
  - *no refusal* -- a response that declines to answer still gets force-matched to its
    nearest reference answer, which inflates the hard-negative count with abstentions
    rather than confident errors. Refusals are detected with the same opener regex the
    EDA notebook uses.

What survives is: questions the model gets confidently, consistently and
misconception-shaped wrong regardless of persona framing.

Run from the repo root: `python scripts/create_finetune_dataset.py`.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data/sample"
HARD_NEGS_PATH = SAMPLE_DIR / "hard_negatives_333.json"
HARD_NEGS_META_PATH = SAMPLE_DIR / "hard_negatives_333.metadata.json"
OUT_DIR = REPO_ROOT / "data/finetune"

MIN_SCORE = 0.60
REQUIRED_NOTEBOOKS = 3

CORE_FIELDS = [
    "type", "category", "question", "best_answer",
    "correct_answers", "incorrect_answers", "source",
]

# (notebook, condition, role, response_field, outcome_field_prefix) -- the same
# condition map the EDA notebook uses to build its tidy long-form frame.
CONDITIONS = {
    "skeptic_credulous": [
        ("Skeptic", "truth-leaning", "SystemPromptSkeptic", "Skeptic"),
        ("Credulous", "falsehood-leaning", "SystemPromptCredulous", "Credulous"),
        ("UserPromptOnly", "plain", "UserPromptOnly", "UserPromptOnly"),
    ],
    "truthful_untruthful": [
        ("Truthful", "truth-leaning", "SystemPromptTruthful", "Truthful"),
        ("Untruthful", "falsehood-leaning", "SystemPromptUntruthful", "Untruthful"),
        ("General", "plain", "SystemPromptGeneral", "General"),
    ],
    "truthful_untruthful_boi": [
        ("Truthful", "truth-leaning", "SystemPromptTruthful", "Truthful"),
        ("Untruthful", "falsehood-leaning", "SystemPromptUntruthful", "Untruthful"),
        ("General", "plain", "SystemPromptGeneral", "General"),
    ],
}

# Verbatim from nb/eda_hard_negatives_333.ipynb section 10 -- kept identical so the
# selection here reproduces the notebook's refusal counts exactly.
REFUSAL_RE = re.compile(
    r"^\s*(i cannot|i can't|i can not|i'm sorry|i am sorry|i apologize|as an ai|"
    r"i must (clarify|point out|respectfully)|it('s| is) not appropriate|"
    r"i don't think it('s| is) appropriate)",
    re.I)


def observations(record):
    """Flatten a record into one entry per (notebook, condition) that is present."""
    obs = []
    for notebook, conds in CONDITIONS.items():
        block = record.get(f"{notebook}_outcomes")
        if block is None:
            continue
        for condition, role, resp_key, prefix in conds:
            response = block[resp_key] or ""
            obs.append({
                "notebook": notebook,
                "condition": condition,
                "role": role,
                "response": response,
                "matched_answer": block[f"{prefix}BestAnswer"],
                "score": block[f"{prefix}BestScore"],
                "is_refusal": bool(REFUSAL_RE.match(response)),
            })
    return obs


def summarise(record, obs):
    """Per-question selection evidence, attached to every emitted row."""
    scores = [o["score"] for o in obs]
    matched = [o["matched_answer"] for o in obs]
    top_attractor = max(set(matched), key=matched.count)
    return {
        "n_notebooks": len(record["source_notebooks"]),
        "n_observations": len(obs),
        "min_score": min(scores),
        "mean_score": round(sum(scores) / len(scores), 6),
        "max_score": max(scores),
        "n_refusals": sum(o["is_refusal"] for o in obs),
        "n_distinct_matched_answers": len(set(matched)),
        # The misconception most conditions converge on -- the thing to train against.
        "dominant_incorrect_answer": top_attractor,
        "dominant_incorrect_answer_share": round(matched.count(top_attractor) / len(matched), 6),
    }


def select(records):
    """Apply the three filters in order, counting what each one drops."""
    funnel = {"input_hard_negatives": len(records)}

    core = [r for r in records if len(r["source_notebooks"]) == REQUIRED_NOTEBOOKS]
    funnel["after_3way_core"] = len(core)

    scored = [(r, observations(r)) for r in core]
    high_sim = [(r, o) for r, o in scored if min(x["score"] for x in o) >= MIN_SCORE]
    funnel["after_min_score"] = len(high_sim)

    clean = [(r, o) for r, o in high_sim if not any(x["is_refusal"] for x in o)]
    funnel["after_no_refusal"] = len(clean)

    selected = []
    for record, obs in clean:
        row = {f: record[f] for f in CORE_FIELDS}
        row["source_notebooks"] = record["source_notebooks"]
        row["selection"] = summarise(record, obs)
        row["observations"] = obs
        for notebook in CONDITIONS:
            row[f"{notebook}_outcomes"] = record[f"{notebook}_outcomes"]
        selected.append(row)

    selected.sort(key=lambda r: (-r["selection"]["min_score"], r["question"]))
    return selected, funnel


def sft_rows(selected):
    """Chat-format supervised rows: the failing question -> its reference-correct answer."""
    rows = []
    for r in selected:
        rows.append({
            "messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["best_answer"]},
            ],
            "metadata": {
                "type": r["type"],
                "category": r["category"],
                "correct_answers": r["correct_answers"],
                "incorrect_answers": r["incorrect_answers"],
                "rejected": r["selection"]["dominant_incorrect_answer"],
                "min_score": r["selection"]["min_score"],
                "source": r["source"],
            },
        })
    return rows


def build_metadata(selected, funnel, upstream_meta, data_file, sft_file, generated_at):
    by_category, by_type = {}, {}
    for r in selected:
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    min_scores = [r["selection"]["min_score"] for r in selected]

    return {
        "output_file": data_file,
        "sft_file": sft_file,
        "generated_at": generated_at,
        "generated_by": "scripts/create_finetune_dataset.py",
        "description": (
            "High-value fine-tuning candidates: TruthfulQA questions the model answers "
            "confidently, consistently and wrongly regardless of persona framing. This is the "
            "cut recommended in nb/eda_hard_negatives_333.ipynb section 15 ('Suggested "
            "downstream cuts') -- 3-way core AND min similarity >= 0.60 AND no refusal in any "
            "condition -- applied to data/sample/hard_negatives_333.json."
        ),
        "how_this_sample_set_was_generated": {
            "summary": (
                "Four hops: the pristine TruthfulQA benchmark -> per-persona generation -> "
                "embedding-similarity scoring against the reference answers -> the union of the "
                "three confusion matrices' hard-negative cells -> this filtered subset. Only the "
                "last hop is performed by this script; the first three are inherited unchanged "
                "from the upstream files named below."
            ),
            "chain": [
                {
                    "step": 1,
                    "stage": "benchmark",
                    "produced_by": "scripts/truthfulqa_download.py",
                    "output": "data/truthfulqa/truthful_qa.json",
                    "detail": (
                        "817 TruthfulQA questions with their seven core fields (type, category, "
                        "question, best_answer, correct_answers, incorrect_answers, source). No "
                        "model responses, no scores."
                    ),
                },
                {
                    "step": 2,
                    "stage": "generation",
                    "produced_by": [
                        "scripts/truthfulqa_personas_oneliner.py",
                        "scripts/truthfulqa_personas_truthful.py",
                        "scripts/truthfulqa_personas_boi.py",
                    ],
                    "output": "one response per (question x condition) for three notebooks",
                    "detail": (
                        "Each of the three persona notebooks answers all 817 questions under "
                        "three conditions: a truth-leaning persona, a falsehood-leaning persona, "
                        "and a plain/no-persona prompt. The plain condition was re-generated "
                        "independently per notebook, so cross-notebook disagreement includes "
                        "run-to-run sampling noise -- which is precisely why the 3-way core "
                        "filter below matters."
                    ),
                },
                {
                    "step": 3,
                    "stage": "scoring",
                    "produced_by": "scripts/truthfulqa_persona_similarity.py",
                    "output": [
                        "data/truthfulqa/truthful_qa.personas.oneliner.similarity.json",
                        "data/truthfulqa/truthful_qa.truthful.personas.similarity.json",
                        "data/truthfulqa/truthful_qa.truthful.boi.similarity.json",
                    ],
                    "detail": (
                        "Every response is embedded and matched to its single nearest reference "
                        "answer across that question's correct_answers + incorrect_answers. The "
                        "match's cosine similarity becomes *BestScore and whether it landed in "
                        "the correct set becomes *BestIsCorrect. Note the scorer always picks "
                        "some nearest neighbour, even for a response that answered nothing."
                    ),
                },
                {
                    "step": 4,
                    "stage": "hard-negative union",
                    "produced_by": "scripts/create_hard_negs_dataset.py",
                    "output": "data/sample/hard_negatives_333.json",
                    "detail": (
                        "Each notebook's confusion matrix bottom-right cell (all three of its "
                        "conditions incorrect) is selected, and the three cells are unioned and "
                        "deduplicated on 'question': 232 + 223 + 240 rows -> 333 unique "
                        "questions. A row carries one *_outcomes block per notebook that "
                        "flagged it."
                    ),
                },
                {
                    "step": 5,
                    "stage": "this filter",
                    "produced_by": "scripts/create_finetune_dataset.py",
                    "output": data_file,
                    "detail": (
                        "The three filters below are applied to those 333 rows, in order, and "
                        "each surviving row is re-emitted with its core TruthfulQA fields, every "
                        "attached *_outcomes block, a flattened 'observations' list (one entry "
                        "per notebook x condition) and a 'selection' block holding the evidence "
                        "the filters were computed from."
                    ),
                },
            ],
            "provenance_note": (
                "The EDA notebook verifies both hops of this chain: all 817 rows of each scored "
                "dataset and all 333 sample rows round-trip exactly to truthful_qa.json on every "
                "core field, including list order. Reference answers were never rewritten "
                "downstream, so the correct/incorrect flags this selection depends on are sound."
            ),
        },
        "source": {
            "data_file": "data/sample/hard_negatives_333.json",
            "metadata_file": "data/sample/hard_negatives_333.metadata.json",
            "upstream_generated_by": upstream_meta.get("generated_by"),
            "upstream_generated_at": upstream_meta.get("generated_at"),
            "analysis_notebook": "nb/eda_hard_negatives_333.ipynb",
            "records_in": funnel["input_hard_negatives"],
        },
        "filters": [
            {
                "name": "3-way core",
                "expression": "len(source_notebooks) == 3",
                "rationale": (
                    "Flagged by all three persona notebooks, so the question fails under all "
                    "nine (notebook x condition) combinations. Difficulty is then a property of "
                    "the question rather than of one notebook's prompt wording or one sampling "
                    "run."
                ),
                "kept": funnel["after_3way_core"],
                "dropped": funnel["input_hard_negatives"] - funnel["after_3way_core"],
            },
            {
                "name": "min similarity >= 0.60",
                "expression": f"min(BestScore over all 9 observations) >= {MIN_SCORE}",
                "rationale": (
                    "Every response is a close paraphrase of a specific known misconception. "
                    "Weak matches are the nearest-neighbour matcher being forced to pick an "
                    "answer for a response that did not really give one."
                ),
                "kept": funnel["after_min_score"],
                "dropped": funnel["after_3way_core"] - funnel["after_min_score"],
            },
            {
                "name": "no refusal in any condition",
                "expression": "not any(REFUSAL_RE.match(response)) over all 9 observations",
                "rationale": (
                    "A refusal still gets force-matched to a nearest reference answer, which "
                    "inflates the hard-negative count with abstentions instead of confident "
                    "errors. Refusals concentrate in the falsehood-leaning personas."
                ),
                "refusal_regex": REFUSAL_RE.pattern,
                "regex_source": "nb/eda_hard_negatives_333.ipynb, section 10 (verbatim)",
                "kept": funnel["after_no_refusal"],
                "dropped": funnel["after_min_score"] - funnel["after_no_refusal"],
            },
        ],
        "counts": {
            **funnel,
            "selected": len(selected),
            "observations_per_question": 9,
            "total_observations": len(selected) * 9,
            "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
            "by_category": dict(sorted(by_category.items(), key=lambda kv: (-kv[1], kv[0]))),
            "min_score_range": [min(min_scores), max(min_scores)],
        },
        "schema": {
            "record": (
                "The seven core TruthfulQA fields, plus source_notebooks, a 'selection' block, an "
                "'observations' list, and the three *_outcomes blocks carried over verbatim from "
                "the hard-negative file."
            ),
            "selection": (
                "Per-question filter evidence: n_notebooks, n_observations, min/mean/max_score, "
                "n_refusals (always 0 by construction), n_distinct_matched_answers, and the "
                "dominant_incorrect_answer with the share of observations that landed on it."
            ),
            "observations": (
                "One entry per (notebook, condition): notebook, condition, role, response, "
                "matched_answer, score, is_refusal."
            ),
            "sft_file": (
                "Chat-format JSONL, one row per selected question: user = the question, "
                "assistant = best_answer, plus a metadata object carrying the full "
                "correct/incorrect reference sets and the dominant incorrect answer under "
                "'rejected' for preference-style training."
            ),
        },
        "caveats": [
            "Hard negatives carry fewer correct reference answers than the complement while "
            "holding the same number of incorrect ones, so part of the difficulty is a scoring "
            "confound rather than a model failure. Verify the reference sets before training.",
            "The labels come from embedding nearest-neighbour matching, not human judgement. A "
            "response can be substantively fine and still match a misconception most closely.",
            "best_answer is the TruthfulQA reference string -- typically terse. Use it as the "
            "target fact, not necessarily as the target response style.",
        ],
    }


def main():
    with open(HARD_NEGS_PATH) as f:
        records = json.load(f)
    with open(HARD_NEGS_META_PATH) as f:
        upstream_meta = json.load(f)

    selected, funnel = select(records)
    n = len(selected)
    data_file = f"finetune_candidates_{n}.json"
    sft_file = f"finetune_candidates_{n}.sft.jsonl"
    meta_file = f"finetune_candidates_{n}.metadata.json"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / data_file, "w") as f:
        json.dump(selected, f, indent=2)
    with open(OUT_DIR / sft_file, "w") as f:
        for row in sft_rows(selected):
            f.write(json.dumps(row) + "\n")
    with open(OUT_DIR / meta_file, "w") as f:
        json.dump(build_metadata(selected, funnel, upstream_meta, data_file, sft_file,
                                 generated_at), f, indent=2)

    print(f"hard negatives in                 : {funnel['input_hard_negatives']}")
    print(f"  after 3-way core                : {funnel['after_3way_core']}")
    print(f"  after min score >= {MIN_SCORE:.2f}         : {funnel['after_min_score']}")
    print(f"  after no refusal                : {funnel['after_no_refusal']}")
    print()
    print(f"selected {n} -> data/finetune/{data_file}")
    print(f"            data/finetune/{sft_file}")
    print(f"            data/finetune/{meta_file}")


if __name__ == "__main__":
    main()
