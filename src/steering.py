"""Reusable activation-steering helpers for the TalkTuner-style control probes
trained by src/train_control_probe.py (Chen et al. 2024 recipe): add
`n_scale * (target_one_hot @ probe.weight)` to the residual stream at the last
token position, for a window of decoder-block layers, on every generation
step. Steering toward class c pushes the residual stream in the direction the
control probe uses to detect class c.

Same recipe as scripts/chat_steered.py and (the now-removed)
nb/causality_tests/intervention_common.py, factored out so any script can
apply a fixed steering configuration to `model.generate(...)` with a single
`with` block instead of re-deriving the hook machinery:

    from steering import Steering

    steer = Steering(model, "rationality", "high", n_scale=9.0, device=device)
    with steer.context():
        out = model.generate(**enc, **gen_kwargs)

Construct one `Steering` per script run (it loads that attribute's control
probes once) and reuse the same `.context()` for every batch.
"""

import json
import os

import torch
from baukit import TraceDict

from probe_common import ATTRIBUTE_LABELS, HIDDEN_DIM, NUM_LAYERS, LinearProbeClassification

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROBE_DIR = os.path.join(REPO_ROOT, "probe_checkpoints", "control_probe", "regular_gullibility_170_custom_split")
DEFAULT_N_SCALE = 7.0  # TalkTuner's own fixed-magnitude default
WINDOW_RADIUS_BEFORE = 6  # matches the causality-test notebooks: [best-6, best+7)
WINDOW_RADIUS_AFTER = 7


def class_names(attribute, probe_dir=None):
    """Class list, in probe-output order, for `attribute` under `probe_dir`.

    Reads it back from the training run's {attribute}_metrics.json
    ("class_names"), the same way scripts/chat_steered.py and
    probe_common.evaluate() do, so a checkpoint dir trained on a subset of a
    attribute's labels (e.g. gullibility with only low/high, because the
    dataset had no "medium" examples) steers with the right number of classes
    instead of failing to load a 2-class checkpoint into a 3-class probe.
    Falls back to probe_common's static table when there's no metrics.json.
    """
    if probe_dir:
        path = os.path.join(probe_dir, f"{attribute}_metrics.json")
        if os.path.isfile(path):
            with open(path) as f:
                names = json.load(f).get("class_names")
            if names:
                return list(names)
    return [name for name, _ in sorted(ATTRIBUTE_LABELS[attribute].items(), key=lambda kv: kv[1])]


def one_hot(class_idx, num_classes):
    vec = [0.0] * num_classes
    vec[class_idx] = 1.0
    return torch.tensor([vec])


def load_summary(probe_dir):
    """{attribute: {"best_layer": int, "best_acc": float, ...}}, or {} if the
    probe directory has no summary.json (falls back to the model's middle layer)."""
    path = os.path.join(probe_dir, "summary.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def default_window(attribute, summary):
    """[from_idx, to_idx) decoder-block window centered on the attribute's
    best probe layer (from summary.json), falling back to the middle of the
    model if there's no entry for it. Returns (from_idx, to_idx, best_layer)."""
    info = summary.get(attribute)
    best_layer = info["best_layer"] if info else NUM_LAYERS // 2
    return best_layer - WINDOW_RADIUS_BEFORE, best_layer + WINDOW_RADIUS_AFTER, best_layer


def load_control_probes(attribute, probe_dir=DEFAULT_PROBE_DIR, checkpoint="best", device="cuda"):
    """Returns {hidden_states_layer_index: LinearProbeClassification}, where
    layer index 0 is the embedding output and i>=1 is the output of decoder
    block i-1 -- matching src/probe_common.py's TextDataset activation indexing."""
    num_classes = len(class_names(attribute, probe_dir))
    probes = {}
    for layer in range(NUM_LAYERS):
        path = os.path.join(probe_dir, f"{attribute}_probe_layer{layer}_{checkpoint}.pth")
        if not os.path.isfile(path):
            continue
        probe = LinearProbeClassification(device, num_classes, HIDDEN_DIM)
        probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        probe.eval()
        probes[layer] = probe
    if not probes:
        raise FileNotFoundError(f"No control-probe checkpoints found for '{attribute}' under {probe_dir}")
    return probes


def which_layers(model, from_idx, to_idx):
    """Decoder-block module names ('model.layers.{i}') for from_idx <= i < to_idx."""
    names = []
    for name, _ in model.named_modules():
        if name.startswith("model.layers.") and name.count(".") == 2 and name.rsplit(".", 1)[-1].isdigit():
            layer_num = int(name.rsplit(".", 1)[-1])
            if from_idx <= layer_num < to_idx:
                names.append(name)
    return sorted(names, key=lambda n: int(n.rsplit(".", 1)[-1]))


def make_steering_hook(probes, cf_target, n_scale):
    """baukit.TraceDict edit_output: adds n_scale * (cf_target @ probe.weight)
    to the last-token residual stream at every traced layer that has a loaded
    probe at layer_num + 1 (probes dict is keyed by hidden_states index, one
    ahead of the decoder-block index TraceDict names layers by)."""

    def edit_output(output, layer_name):
        layer_num = int(layer_name.rsplit(".", 1)[-1])
        probe = probes.get(layer_num + 1)
        if probe is None:
            return output
        weight = probe.proj[0].weight.detach().to(torch.float)
        direction = (cf_target.to(weight.device) @ weight) * n_scale
        hidden = output[0]
        hidden[:, -1] = (hidden[:, -1].to(torch.float) + direction).to(hidden.dtype)
        return output

    return edit_output


class Steering:
    """One fixed steering configuration -- attribute, target class, scale,
    and layer window -- ready to wrap any `model.generate(...)` call.

    Loads the attribute's control-probe checkpoints once at construction and
    defaults the layer window to [best_layer - 6, best_layer + 7) from the
    probe directory's summary.json (override with from_idx/to_idx).
    """

    def __init__(self, model, attribute, label, n_scale=DEFAULT_N_SCALE, probe_dir=DEFAULT_PROBE_DIR,
                 from_idx=None, to_idx=None, device="cuda"):
        if attribute not in ATTRIBUTE_LABELS:
            raise ValueError(f"unknown attribute {attribute!r}; choices: {list(ATTRIBUTE_LABELS)}")
        labels = class_names(attribute, probe_dir)
        if label not in labels:
            raise ValueError(f"'{label}' is not a class of {attribute!r}; choices: {labels}")

        self.model = model
        self.attribute = attribute
        self.label = label
        self.n_scale = n_scale
        self.probe_dir = probe_dir
        self.summary = load_summary(probe_dir)
        self.probes = load_control_probes(attribute, probe_dir, device=device)

        default_from, default_to, self.best_layer = default_window(attribute, self.summary)
        self.from_idx = default_from if from_idx is None else from_idx
        self.to_idx = default_to if to_idx is None else to_idx
        self.layer_names = which_layers(model, self.from_idx, self.to_idx)
        if not self.layer_names:
            raise ValueError(f"no decoder blocks fall in layer window [{self.from_idx}, {self.to_idx})")

        self.labels = labels
        self.cf_target = one_hot(labels.index(label), len(labels))
        self._hook = make_steering_hook(self.probes, self.cf_target, self.n_scale)

    def context(self):
        """Context manager to wrap a `model.generate(...)` call in."""
        return TraceDict(self.model, self.layer_names, edit_output=self._hook)

    def describe(self):
        info = self.summary.get(self.attribute, {})
        best = f", best layer {info['best_layer']} (val acc {info['best_acc']:.3f})" if info else ""
        return (f"{self.attribute}={self.label} (classes {self.labels}) n_scale={self.n_scale} "
                f"layers=[{self.from_idx}, {self.to_idx}){best}")
