"""Train control probes for the persona-attribute datasets.

Ports the control-probe recipe from TalkTuner (Chen et al. 2024,
notebooks/train_probes/train_read_and_controlling_probes.run.ipynb) to the
four new attributes (gullibility, rationality, seriousness,
certainty_seeking) described in docs/llama_dataset_synthesis.md. Shared
machinery (dataset loading, probe, training loop, plotting) lives in
probe_common.py; see also train_reading_probe.py for the sibling script.

Unlike the reading probe, no "I think the {attribute} of this user is"
prompt is appended — the final assistant turn is dropped and nothing else is
added, so the cached activation is the last token of the user's own final
message (the boundary right before the assistant would start generating).
This is the representation TalkTuner steers at inference time, so it is also
what a downstream activation-steering probe should be trained on.

--dataset_dirs takes a *list* of dataset roots (default: both
datasets_llama2_sample and datasets_claudeopus_sample). For each attribute,
conversations from every listed root are pooled into a single combined
dataset before the train/test split, so one probe per layer is trained on
data from all sources together (not one probe per source).

For each (attribute, layer) this:
  1. Feeds each conversation, formatted as a Llama-2 chat prompt with the
     final assistant turn removed (nothing appended), through the base LM
     and caches the last-token residual-stream activation at every layer.
  2. Trains a one-vs-rest logistic probe (linear + sigmoid, BCE loss against
     one-hot labels) per layer on an 80/20 stratified split.
  3. Saves the best- and final-epoch checkpoints, plus accuracy-vs-layer,
     loss-curve, and confusion-matrix plots for the best layer, and a
     per-attribute *_metrics.json with accuracy at every layer.

Usage:
    python src/train_control_probe.py
    python src/train_control_probe.py --dataset_dirs datasets_llama2_sample
    python src/train_control_probe.py --attributes gullibility --max_epochs 10 --layers 0 20 40

See README.md for a full walkthrough and the output layout.
"""

from probe_common import build_arg_parser, run


def main():
    parser = build_arg_parser(__doc__, default_output_dir="probe_checkpoints/control_probe")
    args = parser.parse_args()
    run(args, probe_type="control")


if __name__ == "__main__":
    main()
