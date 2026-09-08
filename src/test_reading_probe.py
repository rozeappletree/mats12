"""Score trained reading probes on a novel, unseen held-out dataset.

Loads the checkpoint(s) written by train_reading_probe.py under --checkpoint_dir
(<attribute>_metrics.json plus <attribute>_probe_layer<L>_{best,final}.pth) and evaluates them on
--test_dirs, a dataset the probe was never trained or validated on. The label set scored is read
back from each attribute's <attribute>_metrics.json ("class_names"), so this automatically matches
whatever --ignore_missing_labels decided when the checkpoint was trained -- it does not re-derive
the full ATTRIBUTE_LABELS schema.

--ignore_missing_labels (default: on) then narrows that set once more to what --test_dirs can
actually exercise: a trained label with zero test examples is excluded from scoring, its column
masked out of the argmax so it can never be predicted, and left out of the confusion matrix and
the macro averages it would otherwise deflate with a meaningless 0. The probe is unchanged -- the
head width is fixed by the checkpoint. Pass --no-ignore_missing_labels to score the full trained
label set instead, reporting 0 precision/recall for the unscoreable classes.

For each attribute this reports accuracy, per-class precision/recall/F1/support, macro- and
weighted-average precision/recall/F1, the full confusion matrix, and a confusion-matrix plot. It
also scores every test conversation individually by the probe's P(high) and writes
<attribute>_test_scores.csv (file, source dir, true/predicted label, score) plus copies of the 10
highest-, 10 lowest-, and 10 closest-to-0.5-scoring conversations' raw text under
eval/examples/<attribute>/{high,low,normal}/, and copies of each confusion-matrix cell's top-10
scoring conversations under eval/examples/<attribute>/confusion/true_<label>-pred_<label>/, for
spot-checking.
Everything is written under <checkpoint_dir>/eval/ (override with --output_dir): meta.json
(model/checkpoint/strategy/test data used), <attribute>_test_metrics.json + confusion-matrix PNG +
score CSV + example conversations per attribute, and test_summary.json across attributes.

Usage:
    python src/test_reading_probe.py --checkpoint_dir probe_checkpoints/reading_probe/defense_484 --test_dirs datasets_hard_333
    python src/test_reading_probe.py --checkpoint_dir probe_checkpoints/reading_probe/regular_gullibility_170_custom_split --test_dirs datasets_defense_484 --attributes gullibility --layers 20

See README.md for a full walkthrough and the output layout.
"""

from probe_common import build_test_arg_parser, evaluate


def main():
    parser = build_test_arg_parser(__doc__, default_probe_dir="probe_checkpoints/reading_probe")
    args = parser.parse_args()
    evaluate(args, probe_type="reading")


if __name__ == "__main__":
    main()
