"""Train reading probes for the persona-attribute datasets.

Ports the reading-probe recipe from TalkTuner (Chen et al. 2024,
notebooks/train_probes/train_read_and_controlling_probes.run.ipynb) to the
four new attributes (gullibility, rationality, seriousness,
certainty_seeking) described in docs/llama_dataset_synthesis.md. Shared
machinery (dataset loading, probe, training loop, plotting) lives in
probe_common.py; see also train_control_probe.py for the sibling script.

--dataset_dirs takes a *list* of dataset roots (default: both
datasets_llama2_sample and datasets_claudeopus_sample). For each attribute,
conversations from every listed root are pooled into a single combined
dataset before the train/test split, so one probe per layer is trained on
data from all sources together (not one probe per source).

Alternatively, pass --train_dirs to fully control the split yourself: it is
pooled as the training set (--dataset_dirs is then ignored). --val_dirs, if
given, is pooled as the held-out split used for model selection and plots
(otherwise falls back to an automatic stratified split of --train_dirs via
--test_size). --test_dirs, if given, is pooled as an additional truly
held-out set evaluated once after training and reported separately in
*_metrics.json (held_out_test_acc_*), without affecting model selection.

--ignore_missing_labels (default: on) checks whether every label defined for
the attribute (e.g. "medium") actually has examples in the data, and drops the
ones that don't so the probe is never given a class it can't learn or be
scored on. This applies in both dataset modes: with --train_dirs the label must
be present in every split given (train, and --val_dirs/--test_dirs if used); with
pooled --dataset_dirs it must be present in the pooled sources. An attribute left
with fewer than two labels is skipped. A notice is printed before training
starts. Pass --no-ignore_missing_labels to keep the full label set regardless
(the missing label is then a dead class: never predicted, and it deflates the
macro averages with a meaningless 0).

Output goes to <--output_dir>/<run_name>/ (default --output_dir:
probe_checkpoints/reading_probe; default run_name: the '+'-joined dataset
directory names minus their 'datasets_' prefix, plus '_custom_split' in
--train_dirs mode). Pass --run_name to name the run folder yourself and keep
several runs over the same data side by side rather than overwriting one
another; --output_dir accepts an absolute path if the checkpoints belong
outside the repo.

For each (attribute, layer) this:
  1. Feeds each conversation, formatted as a Llama-2 chat prompt with the
     final assistant turn removed and " I think the {attribute} of this
     user is" appended, through the base LM and caches the last-token
     residual-stream activation at every layer.
  2. Trains a one-vs-rest logistic probe (linear + sigmoid, BCE loss against
     one-hot labels) per layer on an 80/20 stratified split.
  3. Saves the best- and final-epoch checkpoints, plus accuracy-vs-layer,
     loss-curve, and confusion-matrix plots for the best layer, and a
     per-attribute *_metrics.json with accuracy at every layer.

Usage:
    python src/train_reading_probe.py --dataset_dirs datasets_defense_484 datasets_regular_gullibility_170
    python src/train_reading_probe.py
    python src/train_reading_probe.py --dataset_dirs datasets_llama2_sample2 datasets_claudeopus_sample2 datasets_sol_100
    python src/train_reading_probe.py --attributes gullibility --max_epochs 10 --layers 0 20 40
    python src/train_reading_probe.py --train_dirs datasets_train --val_dirs datasets_val --test_dirs datasets_test
    python src/train_reading_probe.py --dataset_dirs datasets_defense_484

See README.md for a full walkthrough and the output layout.
"""

from probe_common import build_arg_parser, run


def main():
    parser = build_arg_parser(__doc__, default_output_dir="probe_checkpoints/reading_probe")
    args = parser.parse_args()
    run(args, probe_type="reading")


if __name__ == "__main__":
    main()
