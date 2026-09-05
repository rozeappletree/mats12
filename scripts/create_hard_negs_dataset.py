"""
Build deduplicated hard-negative and persona-sanity-check subsets from the two
TruthfulQA persona-framing confusion matrices defined in:

  - nb/vis_skeptic_credulous_persona.ipynb   (Skeptic / Credulous / UserPromptOnly)
  - nb/vis_truthful_untruthful_personal.ipynb (Truthful / Untruthful / General)

Each notebook builds an 8-cell confusion matrix over a triplet of
(persona_A_correct, persona_B_correct, persona_C_correct) booleans, with rows =
(persona_A, persona_B) and columns = persona_C:

    row (False, False), col False -> bottom-right cell: "hard negatives"
        (every condition lands on an incorrect answer)
    row (False, False), col True  -> bottom-left cell: "persona sanity check"
        (the plain/neutral condition is correct, both explicit personas are wrong)

This script selects those two cells from each notebook's underlying data, takes
the union of the two notebooks' subsets deduplicated on the shared `question`
field, and writes each union plus a metadata file describing how it was built.
Run from the repo root: `python scripts/create_hard_negs_dataset.py`.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SC_PATH = REPO_ROOT / "data/truthfulqa/truthful_qa.personas.oneliner.similarity.json"
TU_PATH = REPO_ROOT / "data/truthfulqa/truthful_qa.truthful.personas.similarity.json"
OUT_DIR = REPO_ROOT / "data/sample"

CORE_FIELDS = [
    "type", "category", "question", "best_answer",
    "correct_answers", "incorrect_answers", "source",
]

SC_OUTCOME_FIELDS = [
    "SystemPromptSkeptic", "SkepticBestAnswer", "SkepticBestScore", "SkepticBestIsCorrect",
    "SystemPromptCredulous", "CredulousBestAnswer", "CredulousBestScore", "CredulousBestIsCorrect",
    "UserPromptOnly", "UserPromptOnlyBestAnswer", "UserPromptOnlyBestScore", "UserPromptOnlyBestIsCorrect",
]

TU_OUTCOME_FIELDS = [
    "SystemPromptTruthful", "TruthfulBestAnswer", "TruthfulBestScore", "TruthfulBestIsCorrect",
    "SystemPromptUntruthful", "UntruthfulBestAnswer", "UntruthfulBestScore", "UntruthfulBestIsCorrect",
    "SystemPromptGeneral", "GeneralBestAnswer", "GeneralBestScore", "GeneralBestIsCorrect",
]


def sc_pattern(r):
    return (r["SkepticBestIsCorrect"], r["CredulousBestIsCorrect"], r["UserPromptOnlyBestIsCorrect"])


def tu_pattern(r):
    return (r["TruthfulBestIsCorrect"], r["UntruthfulBestIsCorrect"], r["GeneralBestIsCorrect"])


def merge_union(sc_subset, tu_subset):
    """Union two per-notebook subsets, deduplicated on `question`, merging outcome fields."""
    by_question = {}
    for r in sc_subset:
        q = r["question"]
        entry = by_question.setdefault(q, {f: r[f] for f in CORE_FIELDS})
        entry["source_notebooks"] = sorted(set(entry.get("source_notebooks", [])) | {"skeptic_credulous"})
        entry["skeptic_credulous_outcomes"] = {f: r[f] for f in SC_OUTCOME_FIELDS}
    for r in tu_subset:
        q = r["question"]
        entry = by_question.setdefault(q, {f: r[f] for f in CORE_FIELDS})
        entry["source_notebooks"] = sorted(set(entry.get("source_notebooks", [])) | {"truthful_untruthful"})
        entry["truthful_untruthful_outcomes"] = {f: r[f] for f in TU_OUTCOME_FIELDS}
    return [by_question[q] for q in sorted(by_question)]


def build_metadata(cell_name, description, sc_selector, tu_selector, sc_matched, tu_matched, union_count, output_file, generated_at):
    return {
        "output_file": output_file,
        "generated_at": generated_at,
        "description": description,
        "process": [
            "1. Load both similarity-scored TruthfulQA persona datasets (817 questions each).",
            "2. Each notebook builds an 8-cell confusion matrix over the triplet of "
            "(persona_A_correct, persona_B_correct, persona_C_correct) booleans, with rows = "
            "(persona_A, persona_B) and columns = persona_C.",
            f"3. Select the {cell_name} cell from each notebook's matrix.",
            "4. Union the two per-notebook subsets, deduplicating on the 'question' field (both "
            "notebooks share the same underlying 817 TruthfulQA questions).",
            "5. For a question appearing in both notebooks' selected cells, merge the core "
            "TruthfulQA fields once and attach both notebooks' outcome fields; for a question from "
            "only one notebook, attach only that notebook's outcome fields.",
        ],
        "sources": [
            {
                "notebook": "nb/vis_skeptic_credulous_persona.ipynb",
                "data_file": "data/truthfulqa/truthful_qa.personas.oneliner.similarity.json",
                "persona_label": "skeptic_credulous",
                "cell_selector": sc_selector,
                "rows_matched": sc_matched,
            },
            {
                "notebook": "nb/vis_truthful_untruthful_personal.ipynb",
                "data_file": "data/truthfulqa/truthful_qa.truthful.personas.similarity.json",
                "persona_label": "truthful_untruthful",
                "cell_selector": tu_selector,
                "rows_matched": tu_matched,
            },
        ],
        "dedup_key": "question",
        "counts": {
            "skeptic_credulous_matched": sc_matched,
            "truthful_untruthful_matched": tu_matched,
            "overlap_between_notebooks": sc_matched + tu_matched - union_count,
            "deduplicated_union_total": union_count,
        },
        "generated_by": "scripts/create_hard_negs_dataset.py",
    }


def main():
    with open(SC_PATH) as f:
        sc_records = json.load(f)
    with open(TU_PATH) as f:
        tu_records = json.load(f)

    # bottom-right cell: row (False, False), col False -- every condition wrong
    sc_hard_negs = [r for r in sc_records if sc_pattern(r) == (False, False, False)]
    tu_hard_negs = [r for r in tu_records if tu_pattern(r) == (False, False, False)]

    # bottom-left cell: row (False, False), col True -- only the plain/neutral condition right
    sc_sanity = [r for r in sc_records if sc_pattern(r) == (False, False, True)]
    tu_sanity = [r for r in tu_records if tu_pattern(r) == (False, False, True)]

    hard_negs_union = merge_union(sc_hard_negs, tu_hard_negs)
    sanity_union = merge_union(sc_sanity, tu_sanity)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    hard_negs_count = len(hard_negs_union)
    sanity_count = len(sanity_union)

    hard_negs_file = f"hard_negatives_{hard_negs_count}.json"
    sanity_file = f"persona_sanitycheck_{sanity_count}.json"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / hard_negs_file, "w") as f:
        json.dump(hard_negs_union, f, indent=2)
    with open(OUT_DIR / sanity_file, "w") as f:
        json.dump(sanity_union, f, indent=2)

    hard_negs_meta = build_metadata(
        cell_name="bottom-right",
        description=(
            "Deduplicated union of the 'hard negatives' cell (bottom-right of the confusion "
            "matrix: every persona/condition lands on an incorrect answer) from two TruthfulQA "
            "persona-framing confusion matrices. These are TruthfulQA misconceptions that no "
            "tested prompting strategy in either notebook could get right -- candidates for "
            "further fine-tuning data or for flagging as 'the model doesn't know this regardless "
            "of framing'."
        ),
        sc_selector=(
            "SkepticBestIsCorrect == False and CredulousBestIsCorrect == False "
            "and UserPromptOnlyBestIsCorrect == False"
        ),
        tu_selector=(
            "TruthfulBestIsCorrect == False and UntruthfulBestIsCorrect == False "
            "and GeneralBestIsCorrect == False"
        ),
        sc_matched=len(sc_hard_negs),
        tu_matched=len(tu_hard_negs),
        union_count=hard_negs_count,
        output_file=hard_negs_file,
        generated_at=generated_at,
    )

    sanity_meta = build_metadata(
        cell_name="bottom-left",
        description=(
            "Deduplicated union of the bottom-left cell of the same two confusion matrices: the "
            "plain/neutral condition (UserPromptOnly / General) answers correctly while BOTH "
            "explicit persona conditions in that notebook answer incorrectly. This is the "
            "'persona injection actively hurts' bucket -- used to sanity-check whether adding any "
            "persona framing at all is worth the risk, since the unframed prompt alone would have "
            "gotten these right."
        ),
        sc_selector=(
            "SkepticBestIsCorrect == False and CredulousBestIsCorrect == False "
            "and UserPromptOnlyBestIsCorrect == True"
        ),
        tu_selector=(
            "TruthfulBestIsCorrect == False and UntruthfulBestIsCorrect == False "
            "and GeneralBestIsCorrect == True"
        ),
        sc_matched=len(sc_sanity),
        tu_matched=len(tu_sanity),
        union_count=sanity_count,
        output_file=sanity_file,
        generated_at=generated_at,
    )

    with open(OUT_DIR / f"hard_negatives_{hard_negs_count}.metadata.json", "w") as f:
        json.dump(hard_negs_meta, f, indent=2)
    with open(OUT_DIR / f"persona_sanitycheck_{sanity_count}.metadata.json", "w") as f:
        json.dump(sanity_meta, f, indent=2)

    print(f"skeptic/credulous hard negs (S- C- U-): {len(sc_hard_negs)}")
    print(f"truthful/untruthful hard negs (T- U- G-): {len(tu_hard_negs)}")
    print(f"deduped union hard negs: {hard_negs_count} -> {hard_negs_file}")
    print()
    print(f"skeptic/credulous sanity (S- C- U+): {len(sc_sanity)}")
    print(f"truthful/untruthful sanity (T- U- G+): {len(tu_sanity)}")
    print(f"deduped union sanity: {sanity_count} -> {sanity_file}")


if __name__ == "__main__":
    main()
