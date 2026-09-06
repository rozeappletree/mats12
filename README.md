# mats12

> **Status: pipeline smoke test, not a result.** Everything below has only
> ever been run against `datasets_llama2_sample` / `datasets_claudeopus_sample`
> — placeholder datasets (9-33 conversations per attribute) that exist to
> validate the training/steering/scoring code end-to-end, not to support any
> conclusion about these attributes. The checkpoints, metrics, plots, and
> causality-test transcripts currently committed are smoke-test output. Once
> full-scale datasets exist (see `docs/llama_dataset_synthesis.md`), rerun
> everything against those before treating any number here as a finding.

## Probes

Two scripts train TalkTuner-style probes (Chen et al. 2024) for four persona
attributes — `gullibility`, `rationality`, `seriousness`,
`certainty_seeking` — on top of `NousResearch/Llama-2-13b-chat-hf`
residual-stream activations:

- `src/train_reading_probe.py` — **reading probe**. Drops the final assistant
  turn and appends `" I think the {attribute} of this user is"`; probes the
  model's completion-primed belief about the user.
- `src/train_control_probe.py` — **control probe**. Drops the final assistant
  turn and appends nothing; probes the residual stream at the boundary right
  before the assistant would start generating (the representation TalkTuner
  actually steers at inference time).

Both share their dataset loading, probe, training loop, and plotting code in
`src/probe_common.py`; see either script's module docstring for the full
method. See `docs/llama_dataset_synthesis.md` for how the datasets were
generated.

For each attribute, conversations from **every** `--dataset_dirs` root are
pooled into one combined dataset before the train/test split — one probe per
layer is trained on data from all listed sources together, not one probe per
source. Pass a single directory to train on just that source instead.

### Setup

Uses the `talktuner-gpu` conda env (torch + transformers + a CUDA GPU;
`NousResearch/Llama-2-13b-chat-hf` needs ~26 GB either already cached under
`~/.cache/huggingface` or downloadable):

```bash
conda activate talktuner-gpu
```

### Run

From the repo root, training on both sample datasets combined (the default):

```bash
python src/train_reading_probe.py
python src/train_control_probe.py
```

Train on a single source instead of pooling:

```bash
python src/train_reading_probe.py --dataset_dirs datasets_llama2_sample
```

Quick smoke test (few layers, few epochs) before a full run:

```bash
python src/train_reading_probe.py --attributes gullibility --layers 0 20 40 --max_epochs 5
```

Both scripts accept the same flags (see `--help` for the full list):

| flag | default | meaning |
|---|---|---|
| `--dataset_dirs` | `datasets_llama2_sample datasets_claudeopus_sample` | dataset roots to pool together |
| `--attributes` | all four | which attributes to train |
| `--layers` | `0 1 ... 40` | which residual-stream layers to probe |
| `--max_epochs` | `50` | epochs per layer |
| `--batch_size` | `32` | train batch size (test uses the full test split in one batch) |
| `--test_size` | `0.2` | held-out fraction (stratified; falls back to a plain split if a class is too small to stratify) |
| `--output_dir` | `probe_checkpoints/{reading,control}_probe` | where checkpoints/plots/metrics are written |

### Output layout

Everything lands under `<output_dir>/<sources joined with "+">/`, e.g.
`probe_checkpoints/reading_probe/llama2_sample+claudeopus_sample/` and
`probe_checkpoints/control_probe/llama2_sample+claudeopus_sample/`:

- `{attribute}_probe_layer{N}_best.pth` / `..._final.pth` — probe weights at
  each layer (best test-accuracy epoch, and the last epoch)
- `{attribute}_metrics.json` — best layer, best accuracy, per-layer train/test
  accuracy arrays (layer depth vs. accuracy), and relative paths to the plots
  below
- `{attribute}_history.pkl` — full per-epoch loss/accuracy history for every
  layer plus the raw predictions used for the confusion matrix
- `plots/{attribute}_accuracy_vs_layer.png` — accuracy vs. layer depth
- `plots/{attribute}_loss_curve_layer{N}.png` — train/test loss and accuracy
  vs. epoch, for the best layer
- `plots/{attribute}_confusion_matrix_layer{N}.png` — test-set confusion
  matrix, for the best layer
- `summary.json` — one-line-per-attribute rollup (best layer/accuracy, example
  counts)

### Caveat on the sample datasets

This is a smoke test, not an experiment. `datasets_llama2_sample` has ~30
conversations per attribute; `datasets_claudeopus_sample` has only 9. Pooling
them still leaves a held-out set of 8-9 examples, so per-layer accuracy is
almost pure noise at this scale — the pipeline runs correctly and produces
the right shapes/plots/checkpoints, but the actual accuracy numbers and
"best layer" picks should not be read as saying anything about the
attributes themselves. Rerun against a real dataset before drawing any
conclusion from these numbers.

## Causality tests

`nb/causality_tests/` ports TalkTuner's activation-steering causality tests
(`TalkTuner-chatbot-llm-dashboard/notebooks/causality_tests/`) to the four new
attributes, unchanged from their method: for a window of residual-stream
layers, it adds `n_scale * (target_one_hot @ control_probe.weight)` — a fixed
magnitude, same as TalkTuner's own `N=7` for their gender probe over layers
19-29 — to the last-token activation at every generation step, then lets the
model generate a response. Steering toward class C pushes the residual stream
in the direction the control probe uses to detect class C; checking whether
the response shifts accordingly is the causality test. Shared code lives in
`nb/causality_tests/intervention_common.py`.

One notebook per attribute:

- `causality_test_on_gullibility.ipynb`, `..._rationality.ipynb`,
  `..._certainty_seeking.ipynb` — steer over TruthfulQA questions (from
  `data/truthfulqa/truthful_qa.json`, filtered to categories chosen per
  attribute: Misconceptions / Logical Falsehood+Superstitions /
  Paranormal+Subjective) and check whether steering shifts how often the
  response's content matches TruthfulQA's correct vs. incorrect answer pool.
- `causality_test_on_seriousness.ipynb` — steers over everyday advice
  questions (`questions/seriousness.txt`); tone has no ground truth, so this
  one is read qualitatively rather than auto-scored.

Each notebook's executed copy is saved alongside it as `*.run.ipynb` (matching
TalkTuner's own convention), and raw responses + per-question transcripts are
saved under `nb/causality_tests/intervention_results/{attribute}/`.

### Run

```bash
conda activate talktuner-gpu
cd nb/causality_tests
jupyter nbconvert --to notebook --execute --output causality_test_on_gullibility.run.ipynb causality_test_on_gullibility.ipynb
```

(or just open the `.ipynb` in Jupyter and run all cells).

### Scoring (TruthfulQA-sourced notebooks only)

Correctness scoring needs sentence embeddings, which live in a separate
`embed` conda env (Qwen3-Embedding-8B — no matplotlib there, so the bar plot
is a second step in `talktuner-gpu`):

```bash
conda run -n embed python nb/causality_tests/score_truthfulqa_responses.py --attribute gullibility
python nb/causality_tests/plot_scored_accuracy.py --attribute gullibility
```

This embeds each condition's response and matches it to the closest
TruthfulQA answer (correct or incorrect) — the same method
`scripts/truthfulqa_persona_similarity.py` already uses in this repo, just
applied to full generations instead of one-liners. It's a fully local
stand-in for the GPT-4 pairwise judge TalkTuner's own notebooks use (no
OpenAI/Anthropic API key is available in this environment) — treat it as a
first-pass signal, not ground truth.

### Caveats

- **This is a smoke test of the steering/scoring code, not a result.** The
  control probes it steers with were themselves trained on the ~30-40-example
  sample datasets (see the Probes caveat above), and only 10 TruthfulQA
  questions were used per attribute. Every number and transcript under
  `intervention_results/` demonstrates that the pipeline runs end-to-end, not
  that steering "gullibility" or "certainty-seeking" actually does anything
  in particular.
- The steering method is left exactly as TalkTuner wrote it (fixed
  `n_scale`, no adaptation to layer depth or probe scale) on purpose: this
  round is about validating the pipeline, not tuning the method. At the
  current (early/mid) layer windows and smoke-test probe quality, some
  conditions may generate incoherent or repetitive text — don't read that as
  "the method needs fixing" yet. Once probes are trained on a real dataset,
  look at what the outputs actually look like and decide from there whether
  `n_scale`, the layer window, or anything else needs to change.
