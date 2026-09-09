#!/usr/bin/env python3
"""Split datasets_hard_333 into two disjoint conversation datasets.

Input : datasets_hard_333/                       (333 hard-negative questions)
        data/finetune/finetune_candidates_66.json (the 66-question cut)
Output: datasets_ardulous_66/                    (the 66 cut)
        datasets_justhard_267/                   (the remaining 267)

Membership is decided on the question string: every record of
finetune_candidates_66.json is looked up in data/finetune/split/hard_negatives_333.json
(the exact file gen_hard_data333.py consumed) to recover its question_index, and
that index set selects the conversation files. Everything else goes to the
justhard group. 66 + 267 == 333, with no question in both.

Files are copied, not moved: datasets_hard_333 stays intact as the parent.
"""

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone

SRC_DIR = "datasets_hard_333"
QUESTIONS_333 = "data/finetune/split/hard_negatives_333.json"
CANDIDATES_66 = "data/finetune/finetune_candidates_66.json"
CANDIDATES_66_META = "data/finetune/finetune_candidates_66.metadata.json"
ARD_DIR = "datasets_ardulous_66"
HARD_DIR = "datasets_justhard_267"
ATTRIBUTE = "gullibility"

NAME_RE = re.compile(r"^conversation_(\d{4})_(\d+)_(.+)_gullibility_(high|low)\.(json|txt)$")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    questions = load_json(QUESTIONS_333)
    candidates = load_json(CANDIDATES_66)
    assert len(questions) == 333, len(questions)
    assert len(candidates) == 66, len(candidates)

    by_question = {}
    for i, q in enumerate(questions):
        by_question.setdefault(q["question"], i)
    assert len(by_question) == 333, "question strings are not unique in the 333 file"

    sel = sorted(by_question[r["question"]] for r in candidates)
    assert len(set(sel)) == 66, f"expected 66 unique indices, got {len(set(sel))}"
    rest = sorted(set(range(333)) - set(sel))
    assert len(rest) == 267, f"expected 267 remaining, got {len(rest)}"
    assert len(sel) + len(rest) == 333
    assert not (set(sel) & set(rest))

    processed = set(load_json(os.path.join(SRC_DIR, "processed_questions.json")))
    failed = set(load_json(os.path.join(SRC_DIR, "failed_questions.json")))

    # Bucket the conversation files by question_index parsed out of the filename.
    src_conv = os.path.join(SRC_DIR, ATTRIBUTE)
    files_by_idx = {}
    for name in sorted(os.listdir(src_conv)):
        m = NAME_RE.match(name)
        assert m, f"unexpected filename: {name}"
        files_by_idx.setdefault(int(m.group(1)), []).append(name)
    total_files = sum(len(v) for v in files_by_idx.values())

    groups = [
        (ARD_DIR, sel, "ardulous_66"),
        (HARD_DIR, rest, "justhard_267"),
    ]

    stats = {}
    for out_dir, idxs, label in groups:
        idxset = set(idxs)
        names = [n for i in idxs for n in files_by_idx.get(i, [])]
        with_files = sorted(i for i in idxs if files_by_idx.get(i))
        stats[label] = {
            "questions": len(idxs),
            "questions_with_conversations": len(with_files),
            "questions_without_conversations": sorted(idxset - set(with_files)),
            "files": len(names),
            "json_files": sum(1 for n in names if n.endswith(".json")),
            "txt_files": sum(1 for n in names if n.endswith(".txt")),
        }
        if args.dry_run:
            continue
        dst_conv = os.path.join(out_dir, ATTRIBUTE)
        os.makedirs(dst_conv, exist_ok=True)
        for n in names:
            shutil.copy2(os.path.join(src_conv, n), os.path.join(dst_conv, n))
        write_json(os.path.join(out_dir, "question_indices.json"), idxs)
        write_json(os.path.join(out_dir, "processed_questions.json"), sorted(processed & idxset))
        write_json(os.path.join(out_dir, "failed_questions.json"), sorted(failed & idxset))
        write_json(os.path.join(out_dir, "questions.json"), [questions[i] for i in idxs])

    # Cross-check: the two groups partition the source directory exactly.
    moved = stats["ardulous_66"]["files"] + stats["justhard_267"]["files"]
    assert moved == total_files, f"{moved} != {total_files}"

    if not args.dry_run:
        upstream_meta = load_json(CANDIDATES_66_META)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for out_dir, idxs, label in groups:
            write_json(
                os.path.join(out_dir, "metadata.json"),
                build_metadata(label, out_dir, idxs, stats, now, upstream_meta, failed),
            )

    for label, s in stats.items():
        print(f"{label}: {s['questions']} questions, "
              f"{s['questions_with_conversations']} with conversations, "
              f"{s['files']} files ({s['json_files']} json + {s['txt_files']} txt)")
    print(f"total: {stats['ardulous_66']['questions']} + {stats['justhard_267']['questions']} "
          f"= {stats['ardulous_66']['questions'] + stats['justhard_267']['questions']} questions, "
          f"{moved}/{total_files} files accounted for")


def build_metadata(label, out_dir, idxs, stats, now, upstream_meta, failed):
    s = stats[label]
    is_ard = label == "ardulous_66"
    return {
        "dataset": out_dir,
        "attribute": ATTRIBUTE,
        "generated_at": now,
        "generated_by": "scripts/split_hard333.py",
        "description": (
            "The 66 hardest of the 333 hard-negative TruthfulQA questions "
            "('ardulous' cut): questions the model answers confidently, consistently "
            "and wrongly under every persona framing."
            if is_ard else
            "The 267 hard-negative TruthfulQA questions that are NOT in the "
            "ardulous 66 cut -- still hard negatives, but they fail at least one of "
            "the three finetune-candidate filters."
        ),
        "how_this_sample_set_was_generated": {
            "summary": (
                "No new generation happened here. This directory is one half of a "
                "two-way split of datasets_hard_333 (conversations already generated "
                "by scripts/gen_hard_data333.py). The split is a pure file partition: "
                "conversation files were copied, never regenerated or edited."
            ),
            "chain": [
                {
                    "step": 1,
                    "stage": "questions",
                    "produced_by": "scripts/create_hard_negs_dataset.py",
                    "output": "data/finetune/split/hard_negatives_333.json",
                    "detail": (
                        "333 unique TruthfulQA questions that all three persona notebooks "
                        "placed in their confusion matrix's all-conditions-incorrect cell. "
                        "See data/sample/hard_negatives_333.metadata.json for the full "
                        "benchmark -> generation -> scoring -> union chain."
                    ),
                },
                {
                    "step": 2,
                    "stage": "conversations",
                    "produced_by": "scripts/gen_hard_data333.py",
                    "output": "datasets_hard_333/gullibility/",
                    "detail": (
                        "For each question, multi-turn human/assistant conversations were "
                        "generated in two variants -- gullibility_high (the human defends a "
                        "plausible falsehood) and gullibility_low (the human reasons through "
                        "to the true answer) -- as a .json (conversation + question fields + "
                        "generation params) and a matching .txt transcript. Filenames encode "
                        "the question's index into hard_negatives_333.json: "
                        "conversation_<4-digit question_index>_<n>_<category>_gullibility_<high|low>.<ext>. "
                        "13 of the 333 questions failed generation and produced no files."
                    ),
                },
                {
                    "step": 3,
                    "stage": "the 66-question cut",
                    "produced_by": "scripts/create_finetune_dataset.py",
                    "output": "data/finetune/finetune_candidates_66.json",
                    "detail": (
                        "Three filters over the 333: 3-way core (flagged by all three "
                        "notebooks) 333->126, min BestScore >= 0.60 across all 9 observations "
                        "126->73, and no refusal in any condition 73->66. Filter details and "
                        "rationales live in data/finetune/finetune_candidates_66.metadata.json."
                    ),
                },
                {
                    "step": 4,
                    "stage": "this split",
                    "produced_by": "scripts/split_hard333.py",
                    "output": [ARD_DIR, HARD_DIR],
                    "detail": (
                        "Each of the 66 candidate records was matched back to "
                        "hard_negatives_333.json on its exact question string (all 66 matched; "
                        "the 333 question strings are unique) to recover its question_index. "
                        "Those 66 indices define datasets_ardulous_66; the complementary 267 "
                        "indices define datasets_justhard_267. Conversation files were assigned "
                        "by the index in their filename. The script asserts 66 + 267 == 333, "
                        "that the index sets are disjoint, and that every one of the "
                        f"{stats['ardulous_66']['files'] + stats['justhard_267']['files']} "
                        "source conversation files landed in exactly one group."
                    ),
                },
            ],
            "selection_rule": (
                "question_index in finetune_candidates_66 index set"
                if is_ard else
                "question_index in range(333) MINUS the finetune_candidates_66 index set"
            ),
            "copied_not_moved": (
                "datasets_hard_333 is left untouched; this directory holds copies."
            ),
        },
        "source": {
            "conversations_dir": SRC_DIR,
            "questions_file": QUESTIONS_333,
            "selection_file": CANDIDATES_66,
            "selection_metadata_file": CANDIDATES_66_META,
            "selection_generated_at": upstream_meta.get("generated_at"),
            "conversation_generation_model": "alibaba/qwen3.7-plus",
        },
        "counts": {
            "questions": s["questions"],
            "questions_with_conversations": s["questions_with_conversations"],
            "questions_failed_generation": len(s["questions_without_conversations"]),
            "files": s["files"],
            "json_files": s["json_files"],
            "txt_files": s["txt_files"],
            "split_check": {
                "ardulous_66": stats["ardulous_66"]["questions"],
                "justhard_267": stats["justhard_267"]["questions"],
                "total": stats["ardulous_66"]["questions"] + stats["justhard_267"]["questions"],
                "expected_total": 333,
            },
        },
        "files_in_this_directory": {
            "gullibility/": "the copied conversation .json / .txt pairs",
            "questions.json": "the source question records for this group, in question_index order",
            "question_indices.json": "the group's question_index values into hard_negatives_333.json",
            "processed_questions.json": "this group's slice of datasets_hard_333/processed_questions.json",
            "failed_questions.json": (
                "this group's slice of datasets_hard_333/failed_questions.json -- "
                "question indices whose generation failed, so they have no conversation files: "
                f"{s['questions_without_conversations']}"
            ),
        },
        "caveats": [
            (
                "Question counts and file counts do not line up: "
                f"{len(s['questions_without_conversations'])} of this group's {s['questions']} "
                "questions failed generation upstream and contribute no conversation files, so "
                f"only {s['questions_with_conversations']} questions are represented in gullibility/."
            ),
            (
                "The 4-digit number in a filename is the question's index into the 333-question "
                "file, not a position within this group -- indices here are sparse and "
                "non-contiguous by construction."
            ),
            (
                "Group membership is inherited from the finetune_candidates_66 filters, which are "
                "based on embedding nearest-neighbour scoring rather than human judgement; the "
                "caveats in finetune_candidates_66.metadata.json carry over."
            ),
        ],
    }


if __name__ == "__main__":
    main()
