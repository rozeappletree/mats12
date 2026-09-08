#!/usr/bin/env python3
"""
chat_steered.py -- chat with Llama-2-13b-chat while steering the residual
stream toward a persona-attribute class, using the control probes trained by
src/train_control_probe.py. Same activation-steering recipe as
nb/causality_tests.withoutLLaMaDuplicates/intervention_common.py (TalkTuner /
Chen et al. 2024: add `n_scale * (target_one_hot @ probe.weight)` to the
residual stream at the last token position, for a window of layers, on every
generation step), wrapped in an interactive REPL instead of a fixed notebook
run, so you can hand-tune the layer window and steering strength per
attribute and see the effect turn by turn.

Defaults to the control probes trained on the deduplicated llama2_sample2 +
claudeopus_sample2 data (probe_checkpoints/control_probe/llama2_sample2+claudeopus_sample2),
and to each attribute's empirically-best probe layer (from that directory's
summary.json), centered in a 13-layer steering window -- the same window
size used to pick FROM_IDX/TO_IDX in the causality-test notebooks.

Also loads the *reading* probes (src/train_reading_probe.py) for all four
attributes and, after every turn, reads the user's gullibility/rationality/
seriousness/certainty-seeking scores off the conversation so far -- same
recipe as TalkTuner's dashboard: drop the last assistant turn, append " I
think the {attribute} of this user is" to the prompt, and classify the last
token's hidden state at that attribute's best reading-probe layer. This is a
read, not an intervention -- it runs independently of whatever steering is
currently active, and never changes what the model says.

SETUP
  conda activate talktuner-gpu

USAGE
  python scripts/chat_steered.py
  python scripts/chat_steered.py --steer gullibility=high --steer certainty_seeking=low \\
      --probe-dir probe_checkpoints/control_probe/llama2+claudeopus_sample2

Multiple attributes can be steered at once -- each active attribute keeps its
own independent layer window and scale; their steering deltas are simply
summed at every shared layer (the recipe is linear, so this composes cleanly).

REPL COMMANDS
  /status                    show every loaded attribute's steering state, layers, scale
  /probes                    list attributes with their classes and best layer/accuracy
  /steer <attr> <label>      steer toward this class of <attr> (loads its probes on first use)
  /steer <attr> off          disable steering for just <attr>
  /steer off                 disable steering for every active attribute
  /layers <attr> <from> <to> steer <attr>'s decoder blocks [from, to) instead of its default window
  /layers <attr> best        reset <attr>'s window to its best-layer default
  /scale <attr> <n>          set <attr>'s steering magnitude n_scale (float, default 7.0)
  /scores                    read the user's attribute scores from the conversation now
  /scores on|off             toggle automatic score readout after every turn (default: on)
  /system                    show the current system prompt
  /system <text>             set/edit the system prompt and clear history
  /reset                     clear conversation history (keeps system prompt + steering)
  /save <name>               save the conversation to data/manual.conversations/<name>
  /help                      show this list of commands
  /quit                      exit
"""

import argparse
import datetime
import json
import os
import sys
import textwrap
from contextlib import nullcontext

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from probe_common import (ATTRIBUTE_LABELS as STATIC_ATTRIBUTE_LABELS, ATTRIBUTE_PROMPT_NAME,  # noqa: E402
                           HIDDEN_DIM, NUM_LAYERS, LinearProbeClassification, llama_v2_prompt)

from baukit import TraceDict  # noqa: E402

SAVE_DIR = os.path.join(REPO_ROOT, "data", "manual.conversations")
DEFAULT_MODEL_NAME = "NousResearch/Llama-2-13b-chat-hf"
DEFAULT_PROBE_DIR = os.path.join(REPO_ROOT, "probe_checkpoints", "control_probe", "regular_gullibility_170_custom_split")
DEFAULT_READING_PROBE_DIR = os.path.join(REPO_ROOT, "probe_checkpoints", "reading_probe", "regular_gullibility_170_custom_split")
WINDOW_RADIUS_BEFORE = 6  # matches the causality-test notebooks: [best-6, best+7)
WINDOW_RADIUS_AFTER = 7
DEFAULT_N_SCALE = 7.0  # TalkTuner's own fixed-magnitude default

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe. Please ensure that your responses "
    "are socially unbiased and positive in nature. If a question does not make "
    "any sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don't know the answer to a question, please "
    "don't share false information."
)


# Which attributes/classes are actually available, discovered per-run from
# whatever probe checkpoints are on disk (see init_attribute_labels) rather
# than assumed to be probe_common's fixed four-attribute/three-class layout
# -- a checkpoint dir trained for just one attribute (e.g. gullibility with
# only "low"/"high") should just work, with no code change needed.
ATTRIBUTE_LABELS = dict(STATIC_ATTRIBUTE_LABELS)


def discover_attributes(*probe_dirs):
    """{attribute: {label: idx}} built from every {attribute}_metrics.json
    found across the given probe directories (later directories don't
    override attributes already found in an earlier one). class_names is
    written by every training run, and is authoritative -- probe_common's
    ATTRIBUTE_LABELS is only consulted for the (not expected) case of a
    metrics.json without it, the same way probe_common.evaluate() reads its
    label set back from class_names rather than assuming the static table.

    Returns {} for a directory with no probes in it -- deliberately, rather
    than falling back to the static table, so a wrong --probe-dir shows up
    as an empty attribute panel instead of four attributes that look
    steerable but have no checkpoints behind them."""
    attrs = {}
    for probe_dir in probe_dirs:
        if not probe_dir or not os.path.isdir(probe_dir):
            print(f"[warn] probe dir does not exist: {probe_dir}")
            continue
        for fname in sorted(os.listdir(probe_dir)):
            if not fname.endswith("_metrics.json") or fname.endswith("_test_metrics.json"):
                continue
            with open(os.path.join(probe_dir, fname)) as f:
                metrics = json.load(f)
            attribute = metrics.get("attribute", fname[: -len("_metrics.json")])
            if attribute in attrs:
                continue
            class_names = metrics.get("class_names")
            if class_names:
                attrs[attribute] = {name: i for i, name in enumerate(class_names)}
            elif attribute in STATIC_ATTRIBUTE_LABELS:
                attrs[attribute] = STATIC_ATTRIBUTE_LABELS[attribute]
    if not attrs:
        print(f"[warn] no {{attribute}}_metrics.json found under {list(probe_dirs)} -- "
              f"no attributes will be offered for steering")
    return attrs


def init_attribute_labels(*probe_dirs):
    """Repoints the module-level ATTRIBUTE_LABELS at whatever attributes/
    classes discover_attributes finds under probe_dirs. Mutates in place
    (rather than rebinding the name) so every function below that reads the
    module global -- and any caller holding a reference to it, like the
    webui -- sees the update without needing its own plumbing."""
    ATTRIBUTE_LABELS.clear()
    ATTRIBUTE_LABELS.update(discover_attributes(*probe_dirs))


def class_names(attribute):
    return [name for name, _ in sorted(ATTRIBUTE_LABELS[attribute].items(), key=lambda kv: kv[1])]


def one_hot(class_idx, num_classes):
    vec = [0.0] * num_classes
    vec[class_idx] = 1.0
    return torch.tensor([vec])


def load_summary(probe_dir):
    path = os.path.join(probe_dir, "summary.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def window_around(layer):
    """[from_idx, to_idx) decoder-block window centered on a given layer,
    same radius used to pick FROM_IDX/TO_IDX in the causality-test notebooks."""
    return layer - WINDOW_RADIUS_BEFORE, layer + WINDOW_RADIUS_AFTER


def default_window(attribute, summary):
    """[from_idx, to_idx) decoder-block window centered on the attribute's
    best probe layer, falling back to the middle of the model if summary.json
    has no entry for it."""
    info = summary.get(attribute)
    best_layer = info["best_layer"] if info else NUM_LAYERS // 2
    from_idx, to_idx = window_around(best_layer)
    return from_idx, to_idx, best_layer


def load_layer_accuracies(probe_dir):
    """{attribute: [(layer, acc), ...]} sorted by test accuracy descending,
    read from each attribute's own {attribute}_metrics.json -- the per-layer
    breakdown that summary.json only reduces to a single best_layer/best_acc."""
    result = {}
    for attribute in ATTRIBUTE_LABELS:
        path = os.path.join(probe_dir, f"{attribute}_metrics.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            metrics = json.load(f)
        pairs = list(zip(metrics["layers"], metrics["test_acc_best_epoch_per_layer"]))
        result[attribute] = sorted(pairs, key=lambda p: p[1], reverse=True)
    return result


def load_control_probes(attribute, probe_dir, device):
    num_classes = len(ATTRIBUTE_LABELS[attribute])
    probes = {}
    for layer in range(NUM_LAYERS):
        path = os.path.join(probe_dir, f"{attribute}_probe_layer{layer}_best.pth")
        if not os.path.isfile(path):
            continue
        probe = LinearProbeClassification(device, num_classes, HIDDEN_DIM)
        probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        probe.eval()
        probes[layer] = probe
    if not probes:
        raise FileNotFoundError(f"No control-probe checkpoints found for '{attribute}' under {probe_dir}")
    return probes


def load_reading_probes(reading_probe_dir, summary, device):
    """Loads one probe per attribute -- the best checkpoint at that
    attribute's own best reading-probe layer -- for live scoring. Returns
    {attribute: (layer, probe)}; attributes with no usable checkpoint are
    omitted (with a warning) rather than failing the whole load.
    """
    probes = {}
    for attribute in ATTRIBUTE_LABELS:
        info = summary.get(attribute)
        layer = info["best_layer"] if info else NUM_LAYERS // 2
        path = os.path.join(reading_probe_dir, f"{attribute}_probe_layer{layer}_best.pth")
        if not os.path.isfile(path):
            print(f"[warn] no reading-probe checkpoint for '{attribute}' at layer {layer} under "
                  f"{reading_probe_dir} -- /scores will skip it")
            continue
        probe = LinearProbeClassification(device, len(ATTRIBUTE_LABELS[attribute]), HIDDEN_DIM)
        probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        probe.eval()
        probes[attribute] = (layer, probe)
    return probes


def _reading_prompt_text(messages, attribute):
    """Same recipe as probe_common.TextDataset with control_probe=False:
    drop the trailing assistant turn (if any) so the prompt ends right where
    the assistant would start responding, then append the completion prompt
    the reading probes were trained to read."""
    trimmed = messages[:-1] if messages and messages[-1]["role"] == "assistant" else list(messages)
    if not trimmed:
        return None
    text = llama_v2_prompt(trimmed)
    text = text[text.find("<s>") + len("<s>"):]
    prompt_name = ATTRIBUTE_PROMPT_NAME.get(attribute, attribute.replace("_", " "))
    return text + f" I think the {prompt_name} of this user is"


@torch.no_grad()
def compute_user_scores(model, tokenizer, messages, reading_probes, device):
    """Returns {attribute: {"labels": [...], "probs": [...], "predicted": label}}
    read off the conversation so far, or None if there's nothing to read yet."""
    results = {}
    for attribute, (layer, probe) in reading_probes.items():
        text = _reading_prompt_text(messages, attribute)
        if text is None:
            return None
        enc = tokenizer(text, truncation=True, max_length=2048, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, return_dict=True)
        act = out.hidden_states[layer][:, -1].to(torch.float)
        probs = probe(act).squeeze(0).tolist()
        labels = class_names(attribute)
        predicted = labels[max(range(len(probs)), key=lambda i: probs[i])]
        results[attribute] = {"labels": labels, "probs": probs, "predicted": predicted}
    return results


def format_scores(results):
    if results is None:
        return "  (nothing to read yet -- send a message first)"
    lines = []
    for attribute, r in results.items():
        breakdown = "  ".join(f"{lbl}={p:.2f}" for lbl, p in zip(r["labels"], r["probs"]))
        lines.append(f"  {attribute:<18} {r['predicted']:<8} ({breakdown})")
    return "\n".join(lines)


def which_layers(model, from_idx, to_idx):
    names = []
    for name, _ in model.named_modules():
        if name.startswith("model.layers.") and name.count(".") == 2 and name.rsplit(".", 1)[-1].isdigit():
            layer_num = int(name.rsplit(".", 1)[-1])
            if from_idx <= layer_num < to_idx:
                names.append(name)
    return sorted(names, key=lambda n: int(n.rsplit(".", 1)[-1]))


def make_multi_steering_hook(specs):
    """specs: list of {probes, cf_target, n_scale, from_idx, to_idx} dicts,
    one per currently-active attribute. Sums each attribute's steering delta
    at every layer that falls inside *its own* window -- the single TraceDict
    this is installed under spans the union of all active windows, so a given
    layer must only pick up deltas from attributes whose window includes it."""

    def edit_output(output, layer_name):
        layer_num = int(layer_name.rsplit(".", 1)[-1])
        total = None
        for spec in specs:
            if not (spec["from_idx"] <= layer_num < spec["to_idx"]):
                continue
            probe = spec["probes"].get(layer_num + 1)
            if probe is None:
                continue
            weight = probe.proj[0].weight.detach().to(torch.float)
            direction = (spec["cf_target"].to(weight.device) @ weight) * spec["n_scale"]
            total = direction if total is None else total + direction
        if total is not None:
            hidden = output[0]
            hidden[:, -1] = (hidden[:, -1].to(torch.float) + total).to(hidden.dtype)
        return output

    return edit_output


def wrap(s, indent="  "):
    return "\n".join(
        textwrap.fill(line, 88, initial_indent=indent, subsequent_indent=indent) or indent
        for line in s.split("\n")
    )


class _AttrConfig:
    """Steering config for one attribute: its probes, classes, current layer
    window/scale, and target label (None == loaded but not currently steering)."""

    def __init__(self, attribute, probe_dir, summary, device):
        self.attribute = attribute
        self.labels = class_names(attribute)
        self.probes = load_control_probes(attribute, probe_dir, device)
        self.from_idx, self.to_idx, self.best_layer = default_window(attribute, summary)
        self.n_scale = DEFAULT_N_SCALE
        self.target_label = None


class SteeringState:
    """Everything that can be hot-swapped from the REPL without reloading the
    model. Holds one _AttrConfig per attribute that's been referenced so far
    (lazily loaded on first use); any number of them can be active
    (target_label != None) at once, and their steering deltas are summed."""

    def __init__(self, probe_dir, summary, device):
        self.probe_dir = probe_dir
        self.summary = summary
        self.device = device
        self.attrs = {}  # attribute -> _AttrConfig, in first-referenced order

    def _ensure_loaded(self, attribute):
        if attribute not in ATTRIBUTE_LABELS:
            raise ValueError(f"unknown attribute {attribute!r}; choices: {list(ATTRIBUTE_LABELS)}")
        if attribute not in self.attrs:
            self.attrs[attribute] = _AttrConfig(attribute, self.probe_dir, self.summary, self.device)
        return self.attrs[attribute]

    def active_attrs(self):
        return [attr for attr, cfg in self.attrs.items() if cfg.target_label is not None]

    def set_target(self, attribute, label):
        cfg = self._ensure_loaded(attribute)
        if label not in cfg.labels:
            raise ValueError(f"'{label}' is not a class of {attribute}; choices: {cfg.labels}")
        cfg.target_label = label

    def clear_target(self, attribute=None):
        if attribute is None:
            for cfg in self.attrs.values():
                cfg.target_label = None
        else:
            self._ensure_loaded(attribute).target_label = None

    def set_layers(self, attribute, from_idx, to_idx):
        cfg = self._ensure_loaded(attribute)
        cfg.from_idx, cfg.to_idx = from_idx, to_idx

    def reset_layers_to_best(self, attribute):
        cfg = self._ensure_loaded(attribute)
        cfg.from_idx, cfg.to_idx, cfg.best_layer = default_window(attribute, self.summary)

    def set_scale(self, attribute, n_scale):
        self._ensure_loaded(attribute).n_scale = n_scale

    def hook_for(self, model):
        """Returns (layer_names, edit_output) for the currently active
        attributes, or ((), None) if none are steering."""
        active = [cfg for cfg in self.attrs.values() if cfg.target_label is not None]
        if not active:
            return (), None
        from_idx = min(cfg.from_idx for cfg in active)
        to_idx = max(cfg.to_idx for cfg in active)
        layer_names = which_layers(model, from_idx, to_idx)
        specs = [
            dict(probes=cfg.probes, cf_target=one_hot(cfg.labels.index(cfg.target_label), len(cfg.labels)),
                 n_scale=cfg.n_scale, from_idx=cfg.from_idx, to_idx=cfg.to_idx)
            for cfg in active
        ]
        return layer_names, make_multi_steering_hook(specs)

    def status(self):
        if not self.attrs:
            return "  no attributes loaded -- /steer <attribute> <label>"
        lines = []
        for attribute, cfg in self.attrs.items():
            info = self.summary.get(attribute, {})
            state = f"ON -> '{cfg.target_label}'" if cfg.target_label is not None else "off"
            is_best = (cfg.from_idx, cfg.to_idx) == default_window(attribute, self.summary)[:2]
            lines.append(f"{attribute:<18} steering: {state:<14} "
                         f"layers: [{cfg.from_idx}, {cfg.to_idx}){'  (best default)' if is_best else ''}  "
                         f"n_scale: {cfg.n_scale}"
                         + (f"  (best layer {info['best_layer']}, val acc {info['best_acc']:.3f})" if info else ""))
        return "\n".join(f"  {l}" for l in lines)

    def config_dict(self):
        return [
            dict(attribute=attribute, target_label=cfg.target_label,
                 from_idx=cfg.from_idx, to_idx=cfg.to_idx, n_scale=cfg.n_scale)
            for attribute, cfg in self.attrs.items() if cfg.target_label is not None
        ]


def save_conversation(name, system_prompt, messages, steering_log):
    if not name:
        print("  usage: /save <filename>")
        return
    if not messages:
        print("  nothing to save yet")
        return
    if os.path.basename(name) != name:
        print("  filename must not contain path separators")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, name if os.path.splitext(name)[1] else name + ".json")
    record = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "system_prompt": system_prompt,
        "messages": messages,
        "steering_log": steering_log,  # steering config in effect for each assistant turn
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  saved conversation to {os.path.relpath(path, REPO_ROOT)}")


@torch.no_grad()
def generate(model, tokenizer, messages, system_prompt, device, args, state):
    text = llama_v2_prompt(messages, system_prompt)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=3500)
    enc = {k: v.to(device) for k, v in enc.items()}
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if args.sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)

    layer_names, edit_output = state.hook_for(model)
    context = TraceDict(model, list(layer_names), edit_output=edit_output) if layer_names else nullcontext()
    with context:
        out = model.generate(**enc, **gen_kwargs)
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def print_help():
    print(__doc__[__doc__.index("REPL COMMANDS"):])


def main():
    ap = argparse.ArgumentParser(description="Chat with Llama-2-13b-chat, steered by a control probe.",
                                  formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--probe-dir", default=DEFAULT_PROBE_DIR,
                     help="directory of {attribute}_probe_layer{N}_best.pth checkpoints "
                          "(default: the deduplicated llama2_sample2+claudeopus_sample2 probes)")
    ap.add_argument("--reading-probe-dir", default=DEFAULT_READING_PROBE_DIR,
                     help="directory of reading-probe checkpoints used for /scores "
                          "(default: the deduplicated llama2_sample2+claudeopus_sample2 reading probes)")
    ap.add_argument("--no-scores", action="store_true", help="don't auto-print user scores after every turn")
    ap.add_argument("--steer", action="append", default=[], metavar="ATTRIBUTE=LABEL",
                     help="start with steering enabled toward this class; repeatable to steer "
                          "multiple attributes at once, e.g. --steer gullibility=high "
                          "--steer certainty_seeking=low. Fine-tune each attribute's layer "
                          "window (/layers) and scale (/scale) from the REPL after startup.")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--sample", action="store_true", help="stochastic sampling instead of greedy decoding")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no CUDA device visible; fp16 on CPU will be extremely slow.")

    print(f"[..] loading {args.model} in fp16 (this takes a few minutes on first run)")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    init_attribute_labels(args.probe_dir, args.reading_probe_dir)

    summary = load_summary(args.probe_dir)
    state = SteeringState(args.probe_dir, summary, device)

    reading_summary = load_summary(args.reading_probe_dir)
    reading_probes = load_reading_probes(args.reading_probe_dir, reading_summary, device)
    show_scores = not args.no_scores
    for spec in args.steer:
        if "=" not in spec:
            ap.error(f"--steer must be ATTRIBUTE=LABEL, got {spec!r}")
        attribute, label = spec.split("=", 1)
        try:
            state.set_target(attribute, label)
        except ValueError as e:
            ap.error(str(e))

    print("[ok] loaded. /help for commands, /quit to exit\n")
    print(state.status())
    print()

    system_prompt = args.system_prompt
    messages = []
    steering_log = []

    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                break

            elif cmd == "/help":
                print_help()

            elif cmd == "/status":
                print(state.status())

            elif cmd == "/probes":
                for attr in ATTRIBUTE_LABELS:
                    info = summary.get(attr)
                    best = f"best layer {info['best_layer']} (acc {info['best_acc']:.3f})" if info else "no summary.json entry"
                    print(f"  {attr:<18} classes={class_names(attr)}  {best}")

            elif cmd == "/steer":
                bits = arg.split()
                if not bits:
                    print("  usage: /steer <attribute> <label>   or   /steer <attribute> off   or   /steer off")
                elif bits == ["off"]:
                    state.clear_target()
                    print("  steering disabled for all attributes")
                elif len(bits) != 2:
                    print("  usage: /steer <attribute> <label>   or   /steer <attribute> off   or   /steer off")
                else:
                    attribute, label = bits
                    try:
                        if label == "off":
                            state.clear_target(attribute)
                            print(f"  steering disabled for '{attribute}'")
                        else:
                            state.set_target(attribute, label)
                            print(f"  steering '{attribute}' toward '{label}'")
                    except (ValueError, FileNotFoundError) as e:
                        print(f"  {e}")

            elif cmd == "/layers":
                bits = arg.split()
                if len(bits) == 2 and bits[1] == "best":
                    attribute = bits[0]
                    try:
                        state.reset_layers_to_best(attribute)
                        cfg = state.attrs[attribute]
                        print(f"  '{attribute}' layers reset to best-default [{cfg.from_idx}, {cfg.to_idx})")
                    except (ValueError, FileNotFoundError) as e:
                        print(f"  {e}")
                elif len(bits) == 3 and all(b.lstrip("-").isdigit() for b in bits[1:]):
                    attribute, from_idx, to_idx = bits[0], int(bits[1]), int(bits[2])
                    if from_idx >= to_idx:
                        print("  <from> must be less than <to>")
                    else:
                        try:
                            state.set_layers(attribute, from_idx, to_idx)
                        except (ValueError, FileNotFoundError) as e:
                            print(f"  {e}")
                        else:
                            n_hit = len(which_layers(model, from_idx, to_idx))
                            print(f"  '{attribute}' layers set to [{from_idx}, {to_idx})  ({n_hit} decoder blocks in range)")
                            if n_hit == 0:
                                print("  [warn] no decoder blocks fall in this range -- steering will be a no-op")
                else:
                    print("  usage: /layers <attribute> <from> <to>   or   /layers <attribute> best")

            elif cmd == "/scores":
                if arg in ("on", "off"):
                    show_scores = arg == "on"
                    print(f"  auto score readout {'enabled' if show_scores else 'disabled'}")
                elif arg:
                    print("  usage: /scores   or   /scores on|off")
                else:
                    if not reading_probes:
                        print("  no reading probes loaded -- see the [warn] lines at startup")
                    else:
                        print(format_scores(compute_user_scores(model, tokenizer, messages, reading_probes, device)))

            elif cmd == "/scale":
                bits = arg.split()
                if len(bits) != 2:
                    print("  usage: /scale <attribute> <number>")
                else:
                    attribute, value = bits
                    try:
                        state.set_scale(attribute, float(value))
                        print(f"  '{attribute}' n_scale set to {state.attrs[attribute].n_scale}")
                    except ValueError as e:
                        print(f"  {e}")

            elif cmd == "/reset":
                messages = []
                print("  conversation cleared")

            elif cmd == "/system":
                if not arg:
                    print(f"  current system prompt:\n{wrap(system_prompt)}")
                else:
                    system_prompt = arg
                    messages = []
                    print("  system prompt updated, conversation cleared")

            elif cmd == "/save":
                save_conversation(arg or None, system_prompt, messages, steering_log)

            else:
                print("  unrecognized; /help")
            continue

        messages.append({"role": "user", "content": line})
        ans = generate(model, tokenizer, messages, system_prompt, device, args, state)
        active = state.active_attrs()
        if active:
            tag = " | ".join(
                f"{attr}={state.attrs[attr].target_label} layers={state.attrs[attr].from_idx}-{state.attrs[attr].to_idx} "
                f"scale={state.attrs[attr].n_scale}"
                for attr in active
            )
        else:
            tag = "off"
        print(f"\nbot [{tag}] >\n{wrap(ans)}\n")
        messages.append({"role": "assistant", "content": ans})
        steering_log.append(state.config_dict())

        if show_scores and reading_probes:
            print("scores:")
            print(format_scores(compute_user_scores(model, tokenizer, messages, reading_probes, device)))
            print()


if __name__ == "__main__":
    main()
