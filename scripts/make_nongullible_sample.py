#!/usr/bin/env python3
"""Pick the hardest question per category that a *non-gullible person* can answer easily.

Input : data/finetune/finetune_candidates_66.json  (the ardulous 66)
Output: data/sample/ardulous_nongullible_hardest_13.json  (+ .metadata.json)

Two rules, applied in this order:

1. ANSWERABLE WITHOUT LOOKUP. A question qualifies only if its accepted answer set
   contains an epistemic-humility answer -- "nothing in particular happens", "I have
   no comment", "it depends", "there is no such thing", "people would disagree",
   "the evidence is inconclusive", "the origin is unknown", "does not replicate".
   These are exactly the questions where the correct move is *declining a baited
   premise* rather than recalling an obscure fact, which is what "a non-gullible
   person answers easily" means operationally. Questions whose only correct answer
   is a specific name, place or number (Roeser, Los Banos, Claudette Colvin, neon)
   are excluded: no amount of scepticism produces that string.

2. HARDEST WINS. Among the qualifiers in a category, take the highest min_score --
   the lowest similarity, across all nine persona x notebook observations, between
   the model's answer and the misconception it matched. High min_score means every
   single run landed as a close paraphrase of the same wrong idea, i.e. the model is
   confidently and consistently wrong. Ties break on the dominant-misconception
   share, then mean_score.

9 of the 22 categories qualify nowhere and are omitted, with their reason recorded
in the metadata.
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

SRC = "data/finetune/finetune_candidates_66.json"
SRC_META = "data/finetune/finetune_candidates_66.metadata.json"
QUESTIONS_333 = "data/finetune/split/hard_negatives_333.json"
GROUP_DIR = "datasets_ardulous_66"
OUT = "data/sample/ardulous_nongullible_hardest_13.json"
OUT_META = "data/sample/ardulous_nongullible_hardest_13.metadata.json"

# Each rule is (name, human-readable gloss, regex over the accepted answers).
HUMILITY_RULES = [
    ("nothing-happens", "the baited consequence simply does not occur", r"nothing in particular"),
    ("no-comment", "the question presumes a fact that is not knowable or not true", r"i have no comment"),
    ("not-enough-info", "the question withholds what would be needed to answer it", r"there is not enough information"),
    ("it-depends", "the question is missing an indexical it silently assumes", r"\bit depends\b"),
    ("no-such-thing", "the object of the question does not exist", r"\bthere (?:is|are) no\b|\bno such\b"),
    ("not-real", "the subject is fictional", r"(?:aren't|isn't|are not|is not) real\b|\bdo(?:n't| not) exist"),
    ("people-disagree", "the question demands one answer where opinions legitimately differ", r"people (?:would )?disagree"),
    ("inconclusive", "the evidence does not support the confident claim", r"\binconclusive\b"),
    ("origin-unknown", "the confident folk explanation is not actually established", r"\bis unknown\b|\bnot known\b"),
    ("no-correlation", "the claimed relationship does not exist", r"\bnot correlated\b|\bno correlation\b"),
    ("fails-to-replicate", "the cited finding does not replicate", r"do(?:es)? not replicate|do(?:es)? not establish"),
    ("no-boss", "the question presumes a circumstance the answerer is not in", r"i do(?:n't| not) have a boss"),
]
COMPILED = [(n, g, re.compile(p, re.I)) for n, g, p in HUMILITY_RULES]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def humility_match(record):
    """First humility rule any accepted answer satisfies, else (None, None, None)."""
    for answer in [record["best_answer"], *record["correct_answers"]]:
        for name, gloss, pat in COMPILED:
            if pat.search(answer):
                return name, gloss, answer
    return None, None, None


def hardness_key(record):
    s = record["selection"]
    return (-s["min_score"], -s["dominant_incorrect_answer_share"], -s["mean_score"])


def main():
    candidates = load(SRC)
    hard333 = load(QUESTIONS_333)
    src_meta = load(SRC_META)
    assert len(candidates) == 66, len(candidates)

    qidx = {q["question"]: i for i, q in enumerate(hard333)}
    conv_dir = os.path.join(GROUP_DIR, "gullibility")
    conv_files = defaultdict(list)
    for name in sorted(os.listdir(conv_dir)):
        conv_files[int(name.split("_")[1])].append(name)

    by_cat = defaultdict(list)
    for r in candidates:
        by_cat[r["category"]].append(r)

    selected, skipped = [], []
    for cat in sorted(by_cat):
        pool = []
        for r in by_cat[cat]:
            rule, gloss, phrase = humility_match(r)
            if rule:
                pool.append((r, rule, gloss, phrase))
        if not pool:
            skipped.append({
                "category": cat,
                "questions_in_category": len(by_cat[cat]),
                "reason": (
                    "every question in this category is answered only by recalling a specific "
                    "name, place or number, so being non-gullible does not make it easy"
                ),
                "example_answer_required": by_cat[cat][0]["best_answer"],
            })
            continue
        pool.sort(key=lambda t: hardness_key(t[0]))
        r, rule, gloss, phrase = pool[0]
        s, i = r["selection"], qidx[r["question"]]
        selected.append({
            "question_index": i,
            "category": cat,
            "type": r["type"],
            "question": r["question"],
            "best_answer": r["best_answer"],
            "correct_answers": r["correct_answers"],
            "incorrect_answers": r["incorrect_answers"],
            "source": r["source"],
            "why_a_non_gullible_person_answers_easily": {
                "rule": rule,
                "gloss": gloss,
                "matched_answer": phrase,
            },
            "why_this_is_the_hardest_in_its_category": {
                "min_score": s["min_score"],
                "mean_score": s["mean_score"],
                "max_score": s["max_score"],
                "dominant_incorrect_answer": s["dominant_incorrect_answer"],
                "dominant_incorrect_answer_share": s["dominant_incorrect_answer_share"],
                "n_distinct_matched_answers": s["n_distinct_matched_answers"],
                "n_observations": s["n_observations"],
                "qualifying_in_category": len(pool),
                "questions_in_category": len(by_cat[cat]),
            },
            "conversations": {
                "dir": conv_dir,
                "files": len(conv_files.get(i, [])),
            },
        })

    assert len(selected) + len(skipped) == len(by_cat) == 22
    assert len({r["category"] for r in selected}) == len(selected), "one per category"

    dump(OUT, selected)
    dump(OUT_META, build_metadata(selected, skipped, by_cat, src_meta))

    print(f"{len(selected)} selected (one per qualifying category), "
          f"{len(skipped)} categories skipped, {len(by_cat)} categories total")
    for r in selected:
        print(f"  [{r['question_index']:3d}] min={r['why_this_is_the_hardest_in_its_category']['min_score']:.3f} "
              f"{r['category']:26s} {r['question'][:58]}")
    print(f"wrote {OUT}\nwrote {OUT_META}")


def build_metadata(selected, skipped, by_cat, src_meta):
    no_conv = [r["question_index"] for r in selected if r["conversations"]["files"] == 0]
    return {
        "output_file": os.path.basename(OUT),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/make_nongullible_sample.py",
        "description": (
            "One question per category: the hardest question in the ardulous 66 that a "
            "non-gullible person can nonetheless answer easily. 'Hardest' is measured on the "
            "model's failure, 'easy' on the kind of reasoning the correct answer needs -- so "
            "every row is a case where confident model error and ordinary human scepticism "
            "come apart."
        ),
        "how_this_sample_set_was_generated": {
            "summary": (
                "A two-rule filter over data/finetune/finetune_candidates_66.json (the 66 "
                "questions also held in datasets_ardulous_66/). No new generation, no new "
                "scoring: every field is carried over from the upstream files."
            ),
            "chain": [
                {
                    "step": 1,
                    "stage": "upstream",
                    "produced_by": "scripts/create_finetune_dataset.py",
                    "output": SRC,
                    "detail": (
                        "The 66 TruthfulQA questions answered confidently, consistently and wrongly "
                        "under all nine (persona notebook x condition) combinations. See "
                        f"{SRC_META} for the benchmark -> generation -> scoring -> union -> filter chain."
                    ),
                },
                {
                    "step": 2,
                    "stage": "rule 1 -- answerable without lookup",
                    "produced_by": "scripts/make_nongullible_sample.py",
                    "detail": (
                        "A question is kept only if one of its accepted answers is an "
                        "epistemic-humility answer, matched by regex over best_answer + "
                        "correct_answers. These are the questions whose correct response is to "
                        "decline a baited premise -- which is what a non-gullible person does "
                        "without needing to look anything up. Questions whose only correct answer "
                        "is an obscure name, place or number are dropped: scepticism does not "
                        "produce 'Roeser' or 'Los Banos'. The rule and the answer string it fired "
                        "on are recorded per row under "
                        "'why_a_non_gullible_person_answers_easily', so every selection is auditable."
                    ),
                    "rules": [{"rule": n, "gloss": g, "regex": p} for n, g, p in HUMILITY_RULES],
                },
                {
                    "step": 3,
                    "stage": "rule 2 -- hardest per category",
                    "produced_by": "scripts/make_nongullible_sample.py",
                    "detail": (
                        "Within each category, qualifiers are ranked by min_score descending "
                        "(ties: dominant_incorrect_answer_share, then mean_score) and the top row "
                        "is taken. min_score is the *lowest* cosine similarity, over all nine "
                        "observations, between the model's answer and the reference answer it "
                        "matched; a high min_score therefore means even the model's least-committed "
                        "run was still a close paraphrase of a specific misconception. Selection "
                        "evidence is recorded per row under "
                        "'why_this_is_the_hardest_in_its_category'."
                    ),
                },
            ],
            "note_on_judgement": (
                "Rule 1 is a regex over answer text, not a human or model rating of question "
                "difficulty -- the dataset carries no human-difficulty label. It is a proxy, and "
                "it is deliberately a conservative one: it can only admit questions whose accepted "
                "answers say so in words."
            ),
        },
        "models": {
            "answering_model": {
                "name": "NousResearch/Llama-2-13b-chat-hf",
                "precision": "fp16",
                "role": (
                    "Produced the answers these rows are built on -- 9 per question (3 persona "
                    "notebooks x 3 conditions). 'dominant_incorrect_answer' is this model's own "
                    "output, and every score in 'why_this_is_the_hardest_in_its_category' measures "
                    "its answers."
                ),
                "produced_by": [
                    "scripts/truthfulqa_personas_oneliner.py",
                    "scripts/truthfulqa_personas_truthful.py",
                    "scripts/truthfulqa_personas_boi.py",
                ],
                "attribution_basis": (
                    "the --model default in all three persona scripts, not overridden by the "
                    "run_personas_*.sh launchers; also the base model named in README.md"
                ),
            },
            "matcher_model": {
                "name": "Qwen/Qwen3-Embedding-8B",
                "role": (
                    "Answered nothing. Embedded each response and picked its single nearest "
                    "reference answer, which is what produced min_score and the correct/incorrect "
                    "flag that 'dominant_incorrect_answer' is derived from."
                ),
                "produced_by": "scripts/truthfulqa_persona_similarity.py",
            },
            "reference_answers": {
                "name": None,
                "role": (
                    "best_answer / correct_answers / incorrect_answers are TruthfulQA's "
                    "human-written benchmark answers, carried through unmodified. No model wrote "
                    "or rewrote them."
                ),
            },
            "conversation_model": {
                "name": "alibaba/qwen3.7-plus",
                "role": (
                    "Generated the synthetic conversations in datasets_ardulous_66/gullibility/. "
                    "Not involved in any field of this file."
                ),
                "produced_by": "scripts/gen_hard_data333.py",
            },
        },
        "source": {
            "data_file": SRC,
            "metadata_file": SRC_META,
            "questions_file": QUESTIONS_333,
            "conversations_dir": GROUP_DIR,
            "upstream_generated_at": src_meta.get("generated_at"),
            "records_in": 66,
        },
        "counts": {
            "categories_total": len(by_cat),
            "categories_selected": len(selected),
            "categories_skipped": len(skipped),
            "records_out": len(selected),
            "one_per_category": True,
            "by_rule": {
                r["why_a_non_gullible_person_answers_easily"]["rule"]:
                    sum(1 for x in selected
                        if x["why_a_non_gullible_person_answers_easily"]["rule"]
                        == r["why_a_non_gullible_person_answers_easily"]["rule"])
                for r in selected
            },
            "min_score_range": [
                min(r["why_this_is_the_hardest_in_its_category"]["min_score"] for r in selected),
                max(r["why_this_is_the_hardest_in_its_category"]["min_score"] for r in selected),
            ],
        },
        "skipped_categories": skipped,
        "schema": {
            "record": (
                "question_index (index into hard_negatives_333.json, and the 4-digit number in the "
                "conversation filenames), category, type, question, best_answer, correct_answers, "
                "incorrect_answers, source, plus the two selection-evidence blocks and a "
                "conversations block giving the file count in datasets_ardulous_66/gullibility/."
            ),
        },
        "caveats": [
            (
                "The incorrect_answers are TruthfulQA's own distractor set, not model output. The "
                "misconception the model actually produced is 'dominant_incorrect_answer'."
            ),
            (
                f"question_index {no_conv} has no conversation files -- its generation failed "
                "upstream -- so it appears here as a question/answer record only."
            ) if no_conv else "Every selected question has conversation files.",
            (
                "One question per category means category coverage, not a difficulty ranking across "
                "the file: the hardest row in a weak category can be easier than a passed-over row "
                "in a strong one."
            ),
        ],
    }


if __name__ == "__main__":
    main()
