# SeeGULL — reading and steering *gullibility* in a chat LLM

Can a linear probe on a chat model's residual stream tell whether the person
it is talking to is **gullible** — and can steering that same direction change
how the model answers them?

The method is TalkTuner (Chen et al. 2024) — reading probes + control probes on
`NousResearch/Llama-2-13b-chat-hf` — ported from the vendored upstream repo
(`TalkTuner-chatbot-llm-dashboard/`, a git submodule) to a new set of persona
attributes: `gullibility`, `rationality`, `seriousness`, `certainty_seeking`.
Later phases narrow to `gullibility` alone (low/high) and to whether the probe
survives **hard** questions — ones where the model itself gets the answer wrong.

> **Status.** The pipeline (data generation → probe training → held-out
> evaluation → activation steering → chat front ends) runs end to end. The
> phase-2B results are trained and evaluated but **not yet interpreted**; see
> [Caveats](#caveats) — in particular there is known data leakage in the 2B
> validation split, and the causality/steering side is still qualitative.

---

## Phases

`.gitignore` is the running log of this project: each phase's datasets and
checkpoint directories are listed there with a comment saying what they are.
The table below is the same story in one place.

| Phase | Question | Data | Checkpoints |
|---|---|---|---|
| **1A** | Can TalkTuner's strategy detect & steer a *human trait* at all? 4 attributes × 3 classes. | `datasets_llama2` (LLaMA-generated, has duplicates), `datasets_llama2_sample2` (deduplicated), `datasets_claudeopus_sample2` (Claude Opus) | `probe_checkpoints.withLLaMaDuplicates`, `probe_checkpoints.withLLaMaOpus` |
| **1B** | Train a better toy model on the same 4 attributes. | `datasets_sol_100` (GPT-5.6-Sol, 100/level/attribute) | `probe_checkpoints.withLLaMaOpusSol` |
| **2A** | **SeeGULL v0.1** — train on *general* gullibility (low/high), test on the TruthfulQA questions that are **not** hard. | `datasets_regular_gullibility_170` (the gullibility slice of 1B), `datasets_defense_484` (the 484 non-hard-negative questions) | `probe_checkpoints.withRegularGullibility` |
| **2B** | **SeeGULL v0.2** — which training mix generalises to hard questions? | `datasets_hard_333` → split disjointly into `datasets_justhard_267` + `datasets_ardulous_66` | `probe_checkpoints.withDefense484Only`, `...withDefense484andRegular170`, `...withDefense484Regular170JustHard267` |

The three 2B runs are exactly what [train.sh](train.sh) executes, in order:
train on defense-only → test on hard; train on defense+regular → test on hard;
train on defense+regular+justhard → test on the held-out ardulous 66.

**Hard vs. non-hard.** `data/finetune/split/` divides TruthfulQA into
`hard_negatives_333.json` (every persona framing lands on an incorrect answer —
built by [scripts/create_hard_negs_dataset.py](scripts/create_hard_negs_dataset.py)
from the three persona confusion-matrix notebooks) and
`non_hard_negatives_484.json` (the rest). "Ardulous" is the 66-question
high-value cut of the hard negatives
([scripts/create_finetune_dataset.py](scripts/create_finetune_dataset.py):
3-way core ∧ min similarity ≥ 0.60 ∧ no refusal), split out by
[scripts/split_hard333.py](scripts/split_hard333.py) and hand-narrowed further
into `data/sample/ardulous_{gullible,nongullible}_hardest_13.json`.

## What is committed, and what is not

Conversation datasets and probe checkpoints are **gitignored as directories**
and committed as **`.zip` snapshots at the repo root** — `datasets_*.zip`,
`probe_checkpoints.*.zip`. Each archive expands to the directory of the same
name, so the working tree is restored with:

```bash
unzip datasets_defense_484.zip          # -> datasets_defense_484/
unzip probe_checkpoints.withDefense484Only.zip
```

A dataset root has one subdirectory per attribute holding
`conversation_{i}_{j}_{category}_{attribute}_{level}.{json,txt}` files;
`src/probe_common.py` reads the label straight off the filename (the last
`_{attribute}_` segment through the extension), plus run bookkeeping
(`metadata.json`, `processed_questions.json`, `failed_questions.json`,
`run.log`) written by the generation scripts.

Also ignored: `logs/`, `__pycache__/`, `.env` (API keys for the data-generation
scripts), and the small smoke-test placeholder datasets
(`datasets_claudeopus_sample`, `datasets_hard_333.sampleWithSol`) — those exist
only to validate code paths and are **not** evidence of anything.

Tracked in git: all code, all notebooks, and the small curated JSON under
`data/` (TruthfulQA + persona generations, the finetune splits, hand-picked
samples, and `data/manual.conversations/` — annotated transcripts from steered
chat sessions, named for what they show).

## Layout

```
src/          probe training / evaluation / steering library
  probe_common.py          dataset loading, LinearProbeClassification, train loop, plots, CLI
  train_reading_probe.py   probe on " I think the {attribute} of this user is" continuation
  train_control_probe.py   probe at the user's last token (what steering acts on)
  test_{reading,control}_probe.py   score a checkpoint on an unseen dataset
  steering.py              reusable `Steering(...).context()` hook for model.generate
scripts/      data generation, TruthfulQA persona sweeps, chat front ends
nb/           EDA + training-curve + confusion-matrix notebooks
webui/        Flask front end for the steered chat session
data/         committed TruthfulQA data, splits, samples, manual transcripts
train.sh      the phase-2B experiment sweep
TalkTuner-chatbot-llm-dashboard/   upstream submodule (Chen et al. 2024)
```

## Setup

```bash
git submodule update --init --recursive
conda activate talktuner-gpu     # torch + transformers + baukit, CUDA GPU
```

`NousResearch/Llama-2-13b-chat-hf` (~26 GB) must be cached under
`~/.cache/huggingface` or downloadable. Embedding-based scoring
(`scripts/truthfulqa_persona_similarity.py`, Qwen3-Embedding-8B) runs in a
separate `embed` env. `installmcp.sh` registers the Jupyter MCP server for
driving the notebooks; data-generation scripts read API keys from `.env`.

## Probes

Both scripts share every flag (`--help` for the full list); they differ only in
what the cached activation is taken from — the reading probe appends
`" I think the {attribute} of this user is"` and reads the model's
completion-primed belief, the control probe appends nothing and reads the
boundary right before the assistant would generate.

```bash
# train (one probe per layer, layers 0..40)
python src/train_reading_probe.py --dataset_dirs datasets_defense_484 \
    --output_dir probe_checkpoints.withDefense484Only --run_name reading_probe

# score that checkpoint on a dataset it has never seen
python src/test_reading_probe.py \
    --checkpoint_dir probe_checkpoints.withDefense484Only/reading_probe \
    --test_dirs datasets_hard_333
```

| flag | default | meaning |
|---|---|---|
| `--dataset_dirs` | `datasets_llama2_sample datasets_claudeopus_sample` (stale phase-1A default — always pass this explicitly) | dataset roots **pooled** per attribute, then auto-split (ignored if `--train_dirs` given) |
| `--train_dirs` / `--val_dirs` / `--test_dirs` | none | explicit split; `--test_dirs` is evaluated once at the end and reported separately |
| `--attributes` | all four | which attributes to train |
| `--layers` | `0 … 40` | residual-stream layers to probe |
| `--max_epochs` / `--batch_size` / `--test_size` | `50` / `32` / `0.2` | training loop and stratified holdout |
| `--output_dir` / `--run_name` | `probe_checkpoints/{reading,control}_probe` / dataset-derived tag | run folder — set `--run_name` to keep several runs over the same data side by side |
| `--ignore_missing_labels` | on | drop labels with no examples so the probe is never given an unlearnable class |

**Training output** lands in `<output_dir>/<run_name>/`:
`{attribute}_probe_layer{N}_{best,final}.pth`, `{attribute}_metrics.json`
(best layer, per-layer train/test accuracy, `class_names`),
`{attribute}_history.pkl`, `plots/` (accuracy-vs-layer, loss curve, confusion
matrix), and `summary.json`.

**Evaluation output** lands in `<checkpoint_dir>/eval/`: per-attribute test
metrics + confusion matrix, `{attribute}_test_scores.csv` (per-conversation
P(high)), `test_summary.json`, `meta.json`, and — for spot-checking — the raw
text of the 10 highest/lowest/most-borderline conversations and the top 10 of
each confusion-matrix cell under `eval/examples/`.

## Steering and chat

`src/steering.py` implements TalkTuner's recipe unchanged: add
`n_scale * (target_one_hot @ control_probe.weight)` to the last-token residual
stream over a window of layers, at every generation step (default `n_scale=7`,
a 13-layer window centred on the attribute's best probe layer). Three front
ends drive the same session code:

```bash
python scripts/chat.py                                        # plain Llama-2 REPL, no probes
python scripts/chat_steered.py --steer gullibility=high       # REPL + live reading-probe scores
python scripts/chat_steered_tui.py --steer gullibility=low    # full-screen TUI with a score sidebar
python webui/app.py --port 5050                               # Flask UI, single session
```

Every turn also *reads* the user's four attribute scores off the conversation
(reading probes, independent of whatever steering is applied). `/save <name>`
writes the transcript to `data/manual.conversations/` — that is where the
committed transcripts came from.

## Data generation

| script | what it makes |
|---|---|
| `gen_llama_dataset.py` | phase-1A conversations from Llama-2 itself (ported from TalkTuner's notebook, with real stopping criteria + validation-before-write) |
| `gen_opus_data100.py`, `gen_sol_data100.py` | phase-1A/1B conversations via Claude Opus / GPT-5.6-Sol, breadth-first across attributes so an interrupted run stays balanced |
| `gen_defense_data484.py` | phase-2A: 6 conversations per non-hard question (3 low, 3 high) — a human defending the true vs. the false answer |
| `gen_hard_data333.py` | same, over the 333 hard negatives, via Qwen 3.7 Plus |
| `find_llama2_duplicates.py`, `build_llama2_sample2.py` | dedup the phase-1A LLaMA data |
| `truthfulqa_*.py`, `run_personas_*.sh` | TruthfulQA persona sweeps + embedding-similarity scoring, the input to the hard/non-hard split |
| `make_{gullible,nongullible}_sample.py` | hand-cut the 13 hardest ardulous questions per side |

All generation scripts are resumable: a question is sent once, existing outputs
are skipped, and progress is tracked in `processed_questions.json` /
`failed_questions.json`. The `*.sh` / `*.watch.sh` siblings run and tail them.

## Notebooks

`nb/` holds the EDA and visualisation, not the pipeline: dataset distributions
per phase (`eda_*`), training curves per checkpoint set
(`vis_training_curves_all_attributes*.ipynb`), confusion matrices
(`vis_confusion_matrices_excl_sampled.ipynb`), the persona-framing matrices the
hard-negative split is derived from (`vis_*_persona.ipynb`), and
`training_checkpoints_analysys.ipynb`.

## Caveats

- **Phase 2B has known data leakage.** LLM response optimisation makes the
  generated conversations sequentially dependent, and the validation split does
  not account for it (see the note in [train.sh](train.sh); the stopgap is a
  k-fold split). Treat 2B validation accuracy as optimistic.
- **Results are not yet interpreted.** The last commit's own TODO is "see
  results empirically / causality" — the checkpoints exist, the empirical read
  of them does not.
- **The steering method is left exactly as TalkTuner wrote it** — fixed
  `n_scale`, no adaptation to layer depth or probe scale. At some layer windows
  the steered generations degrade; that is expected at this stage, not a bug to
  fix before the probes are trusted.
- **Causality scoring is a stand-in.** Correctness is judged by embedding a
  generation and matching it to the nearest TruthfulQA answer, not by the GPT-4
  pairwise judge the upstream notebooks use. First-pass signal only.
- **Anything under a `*_sample*` name is a smoke test.** Those datasets exist to
  prove the code runs; their accuracies are noise.

## Reference

Chen et al. 2024, *Designing a Dashboard for Transparency and Control of
Conversational AI* — vendored at `TalkTuner-chatbot-llm-dashboard/`.
