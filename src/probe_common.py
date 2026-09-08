"""Shared machinery for training TalkTuner-style probes on persona-attribute
datasets (Chen et al. 2024). Used by train_reading_probe.py and
train_control_probe.py — the two differ only in what text the last cached
activation is taken from:

  * reading probe:  the final assistant turn is dropped and " I think the
    {attribute} of this user is" is appended; the probe reads the model's
    completion-primed belief about the user.
  * control probe:  the final assistant turn is dropped and nothing is
    appended, so the cached token is the boundary right after the user's
    last message (i.e. right where the assistant would start generating);
    the probe reads what the model "knows" going into its own response.

See docs/llama_dataset_synthesis.md and
TalkTuner-chatbot-llm-dashboard/notebooks/train_probes/train_read_and_controlling_probes.run.ipynb
for the source recipe.
"""

import argparse
import csv
import datetime
import json
import os
import pickle
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

NUM_LAYERS = 41  # embedding output + 40 transformer blocks, Llama-2-13B
HIDDEN_DIM = 5120

ATTRIBUTE_LABELS = {
    "gullibility": {"low": 0, "medium": 1, "high": 2},
    "rationality": {"low": 0, "medium": 1, "high": 2},
    "seriousness": {"low": 0, "medium": 1, "high": 2},
    "certainty_seeking": {"low": 0, "neutral": 1, "high": 2},
}

# Phrase substituted into "I think the {phrase} of this user is" for reading
# probes — must read naturally as a noun phrase, unlike the raw attribute key.
ATTRIBUTE_PROMPT_NAME = {
    "gullibility": "gullibility",
    "rationality": "rationality",
    "seriousness": "seriousness",
    "certainty_seeking": "certainty-seeking tendency",
}


# --------------------------------------------------------------------------
# Conversation / prompt helpers (ported from TalkTuner's src/dataset.py)
# --------------------------------------------------------------------------

def split_conversation(text, user_identifier="HUMAN:", ai_identifier="ASSISTANT:"):
    user_messages = []
    assistant_messages = []
    current_user_message = ""
    current_assistant_message = ""

    for line in text.split("\n"):
        line = line.lstrip(" ")
        if line.startswith(user_identifier):
            if current_assistant_message:
                assistant_messages.append(current_assistant_message.strip())
                current_assistant_message = ""
            current_user_message += line.replace(user_identifier, "").strip() + " "
        elif line.startswith(ai_identifier):
            if current_user_message:
                user_messages.append(current_user_message.strip())
                current_user_message = ""
            current_assistant_message += line.replace(ai_identifier, "").strip() + " "

    if current_user_message:
        user_messages.append(current_user_message.strip())
    if current_assistant_message:
        assistant_messages.append(current_assistant_message.strip())

    return user_messages, assistant_messages


def llama_v2_prompt(messages, system_prompt=None):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    default_system_prompt = system_prompt or (
        "You are a helpful, respectful and honest assistant. Always answer as "
        "helpfully as possible, while being safe. Please ensure that your "
        "responses are socially unbiased and positive in nature. If a question "
        "does not make any sense, or is not factually coherent, explain why "
        "instead of answering something not correct. If you don't know the "
        "answer to a question, please don't share false information."
    )

    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": default_system_prompt}] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {prompt['content'].strip()} {E_INST} {answer['content'].strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    if messages[-1]["role"] == "user":
        messages_list.append(f"{BOS}{B_INST} {messages[-1]['content'].strip()} {E_INST}")

    return "".join(messages_list)


class TextDataset(Dataset):
    """Probe dataset for one attribute, pooled across one or more source
    directories: caches the last-token residual-stream activation at every
    layer for each conversation, labeled from its
    `conversation_{i}_{attribute}_{label}.txt` filename.

    With `control_probe=False` (reading probe), " I think the {attribute} of
    this user is" is appended before the forward pass, so the cached token is
    that appended prompt's last token. With `control_probe=True`, nothing is
    appended, so the cached token is the last token of the user's final
    message (the point right before the assistant would respond).
    """

    def __init__(self, directories, tokenizer, model, attribute, device="cuda", control_probe=False,
                 label_to_id=None):
        self.attribute = attribute
        self.control_probe = control_probe
        self.label_idf = f"_{attribute}_"
        self.label_to_id = label_to_id if label_to_id is not None else ATTRIBUTE_LABELS[attribute]
        self.prompt_name = ATTRIBUTE_PROMPT_NAME[attribute]
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.file_paths = sorted(
            os.path.join(directory, f)
            for directory in directories
            for f in os.listdir(directory)
            if f.endswith(".txt") and os.path.isfile(os.path.join(directory, f))
        )
        self.texts = []
        self.labels = []
        self.acts = []
        self.used_file_paths = []  # file_paths[i] that actually produced texts/labels/acts[i]
        self._load()

    def _load(self):
        skipped = 0
        for file_path in tqdm(self.file_paths, desc=f"  activations [{self.attribute}]"):
            label = file_path[file_path.rfind(self.label_idf) + len(self.label_idf):file_path.rfind(".txt")]
            if label not in self.label_to_id:
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            if "### Human:" in raw_text:
                user_msgs, ai_msgs = split_conversation(raw_text, "### Human:", "### Assistant:")
            elif "### User:" in raw_text:
                user_msgs, ai_msgs = split_conversation(raw_text, "### User:", "### Assistant:")
            else:
                user_msgs, ai_msgs = split_conversation(raw_text)

            messages = []
            for user_msg, ai_msg in zip(user_msgs, ai_msgs):
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": ai_msg})

            if not messages:
                skipped += 1
                continue
            if messages[-1]["role"] == "assistant":
                messages = messages[:-1]  # drop final assistant turn (reading/control probe recipe)

            try:
                text = llama_v2_prompt(messages)
            except Exception:
                skipped += 1
                continue

            text = text[text.find("<s>") + len("<s>"):]
            if not self.control_probe:
                text += f" I think the {self.prompt_name} of this user is"

            with torch.no_grad():
                encoding = self.tokenizer(
                    text, truncation=True, max_length=2048,
                    return_attention_mask=True, return_tensors="pt",
                )
                output = self.model(
                    input_ids=encoding["input_ids"].to(self.device),
                    attention_mask=encoding["attention_mask"].to(self.device),
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_acts = torch.cat([
                    output["hidden_states"][layer][:, -1].detach().cpu().to(torch.float)
                    for layer in range(NUM_LAYERS)
                ])  # [NUM_LAYERS, HIDDEN_DIM]

            self.texts.append(text)
            self.labels.append(self.label_to_id[label])
            self.acts.append(last_acts)
            self.used_file_paths.append(file_path)

        if skipped:
            print(f"  skipped {skipped} malformed file(s)")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"hidden_states": self.acts[idx], "label": self.labels[idx]}


# --------------------------------------------------------------------------
# Probe (ported from TalkTuner's src/probes.py: LinearProbeClassification
# with logistic=True, i.e. one-vs-rest logistic regression per layer)
# --------------------------------------------------------------------------

class TrainerConfig:
    learning_rate = 1e-3
    betas = (0.9, 0.95)
    weight_decay = 0.1


class LinearProbeClassification(nn.Module):
    def __init__(self, device, probe_class, input_dim=HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, probe_class), nn.Sigmoid())
        self.apply(self._init_weights)
        self.to(device)

    def forward(self, act):
        return self.proj(act)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=train_config.learning_rate,
            betas=train_config.betas, weight_decay=train_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=0)
        return optimizer, scheduler


def run_epoch(probe, loader, device, layer, num_classes, optimizer=None):
    training = optimizer is not None
    probe.train(training)
    loss_fn = nn.BCELoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets, all_probs = [], [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            act = batch["hidden_states"][:, layer].to(device)
            target = batch["label"].to(device).long()
            target_one_hot = F.one_hot(target, num_classes).float()

            probs = probe(act)
            loss = loss_fn(probs, target_one_hot)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = torch.argmax(probs, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            total_loss += loss.item() * target.size(0)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_probs)


def stratified_split(labels, test_size, seed):
    idx = list(range(len(labels)))
    try:
        return train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True, stratify=labels)
    except ValueError:
        # too few examples in some class to stratify -- fall back to a plain shuffle split
        return train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_accuracy_vs_layer(layers, train_final_acc, best_test_acc, final_test_acc,
                            attribute, dataset_tag, probe_type, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, train_final_acc, label="train acc (final epoch)", marker="o", markersize=3)
    ax.plot(layers, best_test_acc, label="test acc (best epoch)", marker="o", markersize=3)
    ax.plot(layers, final_test_acc, label="test acc (final epoch)", marker="o", markersize=3)
    ax.set_xlabel("layer")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{attribute} {probe_type} probe accuracy vs. layer ({dataset_tag})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curve(history, attribute, dataset_tag, probe_type, layer, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["test_loss"], label="test")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("BCE loss")
    ax_loss.set_title("loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["test_acc"], label="test")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle(f"{attribute} {probe_type} probe, layer {layer} ({dataset_tag})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(truths, preds, class_names, attribute, dataset_tag, probe_type, layer, path):
    cm = confusion_matrix(truths, preds, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6.5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{attribute} {probe_type} probe confusion matrix, layer {layer}\n({dataset_tag})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Per-attribute training
# --------------------------------------------------------------------------

def train_attribute(dataset, attribute, dataset_tag, probe_type, out_dir, plot_dir, layers, max_epochs,
                     batch_size, test_size, seed, device, val_dataset=None, test_dataset=None, label_to_id=None):
    """`dataset` is the training set. If `val_dataset` is given (explicit
    --train_dirs/--val_dirs mode) it is used as the held-out split for model
    selection / plots instead of an automatic stratified split of `dataset`.
    If `test_dataset` is also given, the best checkpoint per layer is
    additionally evaluated once on it (--test_dirs), reported separately and
    not used for model selection.

    `label_to_id` overrides ATTRIBUTE_LABELS[attribute] (used by
    --ignore_missing_labels to train/evaluate on a reduced, contiguous label
    set when some label has no examples in the data it was given).
    """
    label_to_id = label_to_id if label_to_id is not None else ATTRIBUTE_LABELS[attribute]
    num_classes = len(label_to_id)
    class_names = [name for name, _ in sorted(label_to_id.items(), key=lambda kv: kv[1])]

    if val_dataset is not None:
        train_ds, test_ds = dataset, val_dataset
    else:
        train_idx, test_idx = stratified_split(dataset.labels, test_size, seed)
        train_ds, test_ds = Subset(dataset, train_idx), Subset(dataset, test_idx)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=max(len(test_ds), 1), shuffle=False)

    history_by_layer = {}
    best_acc_per_layer = []
    final_acc_per_layer = []
    train_final_acc_per_layer = []
    best_payload_by_layer = {}

    for layer in layers:
        probe = LinearProbeClassification(device, num_classes)
        optimizer, scheduler = probe.configure_optimizers(TrainerConfig())
        history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
        best_acc, best_payload = 0.0, None

        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc, _, _, _ = run_epoch(probe, train_loader, device, layer, num_classes, optimizer)
            test_loss, test_acc, test_preds, test_truths, _ = run_epoch(probe, test_loader, device, layer, num_classes)
            scheduler.step(test_loss)

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            if test_acc >= best_acc:
                best_acc = test_acc
                best_payload = {
                    "state_dict": {k: v.detach().clone() for k, v in probe.state_dict().items()},
                    "preds": test_preds,
                    "truths": test_truths,
                    "epoch": epoch,
                }

        torch.save(best_payload["state_dict"], os.path.join(out_dir, f"{attribute}_probe_layer{layer}_best.pth"))
        torch.save(probe.state_dict(), os.path.join(out_dir, f"{attribute}_probe_layer{layer}_final.pth"))

        history_by_layer[layer] = history
        best_acc_per_layer.append(best_acc)
        final_acc_per_layer.append(history["test_acc"][-1])
        train_final_acc_per_layer.append(history["train_acc"][-1])
        best_payload_by_layer[layer] = best_payload

        print(f"  [{attribute}] layer {layer:2d}: best test acc {best_acc:.3f} "
              f"(epoch {best_payload['epoch']}), final test acc {history['test_acc'][-1]:.3f}")

    best_layer = layers[int(np.argmax(best_acc_per_layer))]

    plot_accuracy_vs_layer(
        layers, train_final_acc_per_layer, best_acc_per_layer, final_acc_per_layer,
        attribute, dataset_tag, probe_type, os.path.join(plot_dir, f"{attribute}_accuracy_vs_layer.png"),
    )
    plot_loss_curve(
        history_by_layer[best_layer], attribute, dataset_tag, probe_type, best_layer,
        os.path.join(plot_dir, f"{attribute}_loss_curve_layer{best_layer}.png"),
    )
    plot_confusion(
        best_payload_by_layer[best_layer]["truths"], best_payload_by_layer[best_layer]["preds"],
        class_names, attribute, dataset_tag, probe_type, best_layer,
        os.path.join(plot_dir, f"{attribute}_confusion_matrix_layer{best_layer}.png"),
    )

    held_out_acc_per_layer = None
    if test_dataset is not None:
        held_out_loader = DataLoader(test_dataset, batch_size=max(len(test_dataset), 1), shuffle=False)
        held_out_acc_per_layer = {}
        for layer in layers:
            probe = LinearProbeClassification(device, num_classes)
            probe.load_state_dict(best_payload_by_layer[layer]["state_dict"])
            _, held_out_acc, _, _, _ = run_epoch(probe, held_out_loader, device, layer, num_classes)
            held_out_acc_per_layer[layer] = held_out_acc
        print(f"  [{attribute}] held-out test acc (best-val checkpoint), best layer {best_layer}: "
              f"{held_out_acc_per_layer[best_layer]:.3f}")

    return {
        "best_layer": best_layer,
        "best_acc_per_layer": dict(zip(layers, best_acc_per_layer)),
        "final_acc_per_layer": dict(zip(layers, final_acc_per_layer)),
        "held_out_acc_per_layer": held_out_acc_per_layer,
        "history_by_layer": history_by_layer,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "n_held_out_test": len(test_dataset) if test_dataset is not None else None,
    }


# --------------------------------------------------------------------------
# CLI / orchestration shared by both entry-point scripts
# --------------------------------------------------------------------------

def build_arg_parser(doc, default_output_dir):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_root", default=repo_root)
    parser.add_argument("--dataset_dirs", nargs="+",
                         default=["datasets_llama2_sample", "datasets_claudeopus_sample"],
                         help="Dataset roots to pool together into one training set per attribute. "
                              "Ignored if --train_dirs is given.")
    parser.add_argument("--train_dirs", nargs="+", default=None,
                         help="Explicit training dataset roots (same layout as --dataset_dirs, i.e. "
                              "each contains one subdirectory per attribute). If given, --dataset_dirs "
                              "is ignored and this is used as the training set instead of pooling "
                              "--dataset_dirs and taking an automatic stratified split of it.")
    parser.add_argument("--val_dirs", nargs="+", default=None,
                         help="Explicit validation dataset roots, used for model selection / plots "
                              "the same way the automatic held-out split is used. Only meaningful "
                              "together with --train_dirs; if omitted, falls back to a stratified "
                              "split of --train_dirs using --test_size.")
    parser.add_argument("--test_dirs", nargs="+", default=None,
                         help="Optional held-out test dataset roots, evaluated once per layer with "
                              "the best (val-selected) checkpoint after training. Reported alongside "
                              "but separate from the val metrics; not used for model selection. Only "
                              "meaningful together with --train_dirs.")
    parser.add_argument("--attributes", nargs="+", default=list(ATTRIBUTE_LABELS.keys()),
                         choices=list(ATTRIBUTE_LABELS.keys()))
    parser.add_argument("--model_name", default="NousResearch/Llama-2-13b-chat-hf")
    parser.add_argument("--output_dir", default=default_output_dir,
                         help=f"Parent directory for this run's checkpoint folder (default: "
                              f"{default_output_dir}). Relative paths are resolved against "
                              f"--repo_root; an absolute path is used as given. The run itself is "
                              f"written to <output_dir>/<run_name>.")
    parser.add_argument("--run_name", default=None,
                         help="Name of the per-run subdirectory under --output_dir, holding the "
                              "checkpoints, plots, *_metrics.json and summary.json. Default: derived "
                              "from the dataset directory names ('+'-joined, minus the 'datasets_' "
                              "prefix, with '_custom_split' appended in --train_dirs mode). Set this "
                              "to keep runs over the same data side by side instead of overwriting "
                              "one another. May contain '/' to nest further.")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(NUM_LAYERS)))
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--ignore_missing_labels", action=argparse.BooleanOptionalAction, default=True,
                         help="If a label defined for the attribute (e.g. 'medium') has no examples in "
                              "the data, drop it everywhere and train/evaluate only on the labels that "
                              "are present, printing a notice before training starts. Applies in both "
                              "dataset modes: with --train_dirs the label must appear in every split "
                              "given (train, plus --val_dirs/--test_dirs if used); with pooled "
                              "--dataset_dirs it must appear in the pooled sources. An attribute left "
                              "with fewer than two labels is skipped. Pass --no-ignore_missing_labels "
                              "to keep the full label set regardless (the missing label's class is then "
                              "never trained or validated, and deflates the macro averages with a 0).")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    return parser


def _load_pooled_dataset(paths, attribute, tokenizer, model, device, control_probe, split_name, label_to_id=None):
    """Pool the `attribute` subdirectory of each root in `paths` into one
    TextDataset (mirrors the --dataset_dirs pooling logic, for one split)."""
    attr_dirs = [os.path.join(p, attribute) for p in paths if os.path.isdir(os.path.join(p, attribute))]
    for p in paths:
        if not os.path.isdir(os.path.join(p, attribute)):
            print(f"  note: {os.path.join(p, attribute)} not found, skipping that source "
                  f"for {attribute} ({split_name})")
    if not attr_dirs:
        return None
    print(f"=== {attribute} [{split_name}] (sources: {', '.join(attr_dirs)}) ===")
    return TextDataset(attr_dirs, tokenizer, model, attribute, device=device, control_probe=control_probe,
                        label_to_id=label_to_id)


def _discover_present_labels(paths, attribute):
    """Scan filenames only (no tokenization/model forward pass) to find which
    of ATTRIBUTE_LABELS[attribute] actually have at least one example under
    the `attribute` subdirectory of each root in `paths`."""
    label_idf = f"_{attribute}_"
    valid_labels = set(ATTRIBUTE_LABELS[attribute])
    present = set()
    for p in paths:
        attr_dir = os.path.join(p, attribute)
        if not os.path.isdir(attr_dir):
            continue
        for f in os.listdir(attr_dir):
            if not f.endswith(".txt"):
                continue
            label = f[f.rfind(label_idf) + len(label_idf):f.rfind(".txt")]
            if label in valid_labels:
                present.add(label)
    return present


def run(args, probe_type):
    """probe_type: 'reading' (appends the completion prompt) or 'control'
    (no appended prompt, last user-message token).

    Two dataset modes:
      * default: --dataset_dirs are pooled per attribute and split into
        train/val with an automatic stratified split (--test_size).
      * explicit: --train_dirs (required to trigger this mode) is pooled as
        the training set; --val_dirs, if given, is pooled as the held-out
        split used for model selection instead of an automatic split
        (otherwise falls back to a stratified split of --train_dirs);
        --test_dirs, if given, is pooled as an additional held-out test set
        evaluated once after training and reported separately.

    In both modes --ignore_missing_labels (default: on) first drops any label
    of the attribute that has no examples in the data given -- in explicit
    mode it must be present in every split, in pooled mode in the pooled
    sources -- so the probe never gets a head for a class it cannot learn.

    Everything is written to <--output_dir>/<run_name>, where run_name
    defaults to the dataset-derived tag and can be overridden with --run_name.
    """
    control_probe = probe_type == "control"
    torch.manual_seed(args.seed)

    explicit_split = args.train_dirs is not None
    if args.val_dirs is not None and not explicit_split:
        raise ValueError("--val_dirs requires --train_dirs")
    if args.test_dirs is not None and not explicit_split:
        raise ValueError("--test_dirs requires --train_dirs")

    if explicit_split:
        train_paths = [os.path.join(args.repo_root, d) for d in args.train_dirs]
        val_paths = [os.path.join(args.repo_root, d) for d in args.val_dirs] if args.val_dirs else None
        test_paths = [os.path.join(args.repo_root, d) for d in args.test_dirs] if args.test_dirs else None
        source_tags = [os.path.basename(d.rstrip("/")).removeprefix("datasets_") for d in args.train_dirs]
        combined_tag = "+".join(source_tags) + "_custom_split"
    else:
        dataset_paths = [os.path.join(args.repo_root, d) for d in args.dataset_dirs]
        source_tags = [os.path.basename(d.rstrip("/")).removeprefix("datasets_") for d in args.dataset_dirs]
        combined_tag = "+".join(source_tags)

    if args.run_name is not None:
        if os.path.isabs(args.run_name) or ".." in args.run_name.split(os.sep):
            raise ValueError("--run_name must be a relative name below --output_dir "
                             f"(got {args.run_name!r}); use --output_dir for the parent path")
        combined_tag = args.run_name.strip("/")
        if not combined_tag:
            raise ValueError("--run_name must not be empty")

    out_dir = os.path.join(args.repo_root, args.output_dir, combined_tag)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Loading {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    model.to(args.device)
    model.eval()

    if explicit_split:
        print(f"Using explicit split -> {out_dir}: train={args.train_dirs} "
              f"val={args.val_dirs} test={args.test_dirs}")
    else:
        print(f"Pooling sources {args.dataset_dirs} -> {out_dir}")

    summary = {}
    for attribute in args.attributes:
        val_dataset = None
        test_dataset = None

        label_to_id = ATTRIBUTE_LABELS[attribute]

        if args.ignore_missing_labels:
            # A label is kept only if it has examples in *every* split we were given.
            # In pooled mode there is just the one pooled source set to check.
            if explicit_split:
                split_paths = [p for p in (train_paths, val_paths, test_paths) if p is not None]
                where = "one of train/val/test"
            else:
                split_paths = [dataset_paths]
                where = "the pooled sources"
            present = set(label_to_id)
            for paths in split_paths:
                present &= _discover_present_labels(paths, attribute)
            missing = set(label_to_id) - present
            if missing:
                kept = sorted(present, key=lambda name: label_to_id[name])
                if len(kept) < 2:
                    print(f"skipping {attribute}: only label(s) {kept} have examples in {where}, "
                          f"which is too few to train a classifier (pass "
                          f"--no-ignore_missing_labels to keep the full label set).")
                    continue
                print(f"  [{attribute}] label(s) {sorted(missing)} have no examples in {where} "
                      f"-- ignoring them everywhere and training/evaluating on {kept} only "
                      f"(pass --no-ignore_missing_labels to keep the full label set).")
                label_to_id = {name: i for i, name in enumerate(kept)}

        if explicit_split:
            dataset = _load_pooled_dataset(train_paths, attribute, tokenizer, model, args.device,
                                            control_probe, "train", label_to_id=label_to_id)
            if dataset is None:
                print(f"skipping {attribute}: no source directories found under --train_dirs")
                continue
            if len(dataset) < 2 * len(label_to_id):
                print(f"  too few usable train examples ({len(dataset)}), skipping")
                del dataset
                continue

            if val_paths is not None:
                val_dataset = _load_pooled_dataset(val_paths, attribute, tokenizer, model, args.device,
                                                     control_probe, "val", label_to_id=label_to_id)
                if val_dataset is None or len(val_dataset) == 0:
                    print(f"skipping {attribute}: no usable examples found under --val_dirs")
                    del dataset
                    continue

            if test_paths is not None:
                test_dataset = _load_pooled_dataset(test_paths, attribute, tokenizer, model, args.device,
                                                      control_probe, "test", label_to_id=label_to_id)
                if test_dataset is None or len(test_dataset) == 0:
                    print(f"  note: no usable examples found under --test_dirs for {attribute}, "
                          f"skipping held-out test eval")
                    test_dataset = None
        else:
            attr_dirs = [os.path.join(p, attribute) for p in dataset_paths if os.path.isdir(os.path.join(p, attribute))]
            for p in dataset_paths:
                if not os.path.isdir(os.path.join(p, attribute)):
                    print(f"  note: {os.path.join(p, attribute)} not found, skipping that source for {attribute}")
            if not attr_dirs:
                print(f"skipping {attribute}: no source directories found")
                continue

            print(f"=== {attribute} [{probe_type}] (sources: {', '.join(attr_dirs)}) ===")
            dataset = TextDataset(attr_dirs, tokenizer, model, attribute, device=args.device,
                                   control_probe=control_probe, label_to_id=label_to_id)
            if len(dataset) < 2 * len(label_to_id):
                print(f"  too few usable examples ({len(dataset)}), skipping")
                del dataset
                continue

        result = train_attribute(
            dataset, attribute, combined_tag, probe_type, out_dir, plot_dir, args.layers, args.max_epochs,
            args.batch_size, args.test_size, args.seed, args.device,
            val_dataset=val_dataset, test_dataset=test_dataset, label_to_id=label_to_id,
        )
        with open(os.path.join(out_dir, f"{attribute}_history.pkl"), "wb") as f:
            pickle.dump(result, f)

        best_layer = result["best_layer"]
        metrics = {
            "attribute": attribute,
            "probe_type": probe_type,
            "sources": (
                {"train_dirs": args.train_dirs, "val_dirs": args.val_dirs, "test_dirs": args.test_dirs}
                if explicit_split else args.dataset_dirs
            ),
            "class_names": [name for name, _ in sorted(label_to_id.items(), key=lambda kv: kv[1])],
            "n_examples": len(dataset),
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "best_layer": best_layer,
            "best_acc": max(result["best_acc_per_layer"].values()),
            # layer depth vs. accuracy -- same data plotted in accuracy_vs_layer.png
            "layers": args.layers,
            "train_acc_final_epoch_per_layer": [result["history_by_layer"][l]["train_acc"][-1] for l in args.layers],
            "test_acc_best_epoch_per_layer": [result["best_acc_per_layer"][l] for l in args.layers],
            "test_acc_final_epoch_per_layer": [result["final_acc_per_layer"][l] for l in args.layers],
            "plots": {
                "accuracy_vs_layer": f"plots/{attribute}_accuracy_vs_layer.png",
                "loss_curve": f"plots/{attribute}_loss_curve_layer{best_layer}.png",
                "confusion_matrix": f"plots/{attribute}_confusion_matrix_layer{best_layer}.png",
            },
        }
        if result["held_out_acc_per_layer"] is not None:
            metrics["n_held_out_test"] = result["n_held_out_test"]
            metrics["held_out_test_acc_per_layer"] = [result["held_out_acc_per_layer"][l] for l in args.layers]
            metrics["held_out_test_acc_best_layer"] = result["held_out_acc_per_layer"][best_layer]

        with open(os.path.join(out_dir, f"{attribute}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        summary[attribute] = {
            "best_layer": best_layer,
            "best_acc": metrics["best_acc"],
            "n_examples": metrics["n_examples"],
            "n_train": metrics["n_train"],
            "n_test": metrics["n_test"],
        }
        if "held_out_test_acc_best_layer" in metrics:
            summary[attribute]["held_out_test_acc"] = metrics["held_out_test_acc_best_layer"]

        del dataset
        if val_dataset is not None:
            del val_dataset
        if test_dataset is not None:
            del test_dataset
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------
# CLI / orchestration for scoring a trained probe on novel held-out data
# --------------------------------------------------------------------------

def build_test_arg_parser(doc, default_probe_dir):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_root", default=repo_root)
    parser.add_argument("--checkpoint_dir", required=True,
                         help=f"Directory produced by a train_*.py run under {default_probe_dir} "
                              "(contains <attribute>_metrics.json and "
                              "<attribute>_probe_layer<L>_{best,final}.pth).")
    parser.add_argument("--test_dirs", nargs="+", required=True,
                         help="Novel held-out dataset roots, same layout as training data (one "
                              "subdirectory per attribute, files named ..._<attribute>_<label>.txt).")
    parser.add_argument("--attributes", nargs="+", default=None, choices=list(ATTRIBUTE_LABELS.keys()),
                         help="Attributes to evaluate (default: every attribute with a "
                              "<attribute>_metrics.json under --checkpoint_dir).")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                         help="Which layer's checkpoint to evaluate per attribute: one value to "
                              "apply to every attribute, or one value per --attributes entry. "
                              "Default: the 'best_layer' recorded in <attribute>_metrics.json.")
    parser.add_argument("--checkpoint_suffix", choices=["best", "final"], default="best",
                         help="Score the best-val-epoch checkpoint (default) or the final-epoch one.")
    parser.add_argument("--ignore_missing_labels", action=argparse.BooleanOptionalAction, default=True,
                         help="If a label the checkpoint was trained on (e.g. 'medium') has zero "
                              "examples in --test_dirs, exclude it from scoring: its column is "
                              "masked out of the argmax so it can never be predicted, and it is "
                              "left out of the confusion matrix and the macro averages it would "
                              "otherwise deflate with a meaningless 0. The probe itself is "
                              "unchanged (the head width is fixed by the checkpoint). Pass "
                              "--no-ignore_missing_labels to score the full trained label set, "
                              "reporting 0 precision/recall for the unscoreable classes.")
    parser.add_argument("--output_dir", default=None,
                         help="Where to write <attribute>_test_metrics.json, the confusion-matrix "
                              "plot, meta.json (model/checkpoint/test-data used) and test_summary.json "
                              "(default: <checkpoint_dir>/eval).")
    parser.add_argument("--model_name", default="NousResearch/Llama-2-13b-chat-hf")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser


def _format_test_report(test_metrics):
    """Render a test_metrics dict (see evaluate()) as a human-readable report."""
    m = test_metrics["meta"]
    title = f"{test_metrics['attribute']} {m['probe_type']} probe -- held-out test report"
    lines = [title, "=" * len(title), ""]
    lines.append(f"model:              {m['model_name']}")
    lines.append(f"checkpoint:         {test_metrics['checkpoint']}")
    lines.append(f"checkpoint_suffix:  {m['checkpoint_suffix']}")
    lines.append(f"layer:              {test_metrics['layer']}")
    lines.append(f"test_dirs:          {', '.join(m['test_dirs'])}")
    if test_metrics.get("train_sources") is not None:
        lines.append(f"train_sources:      {test_metrics['train_sources']}")
    lines.append(f"evaluated_at:       {m['evaluated_at']}")
    lines.append(f"n_test:             {test_metrics['n_test']}")
    lines.append("")
    lines.append(f"accuracy:           {test_metrics['accuracy']:.4f}")
    lines.append(f"loss:               {test_metrics['loss']:.4f}")
    lines.append("")

    class_names = test_metrics["class_names"]
    lines.append(f"{'':>12s}  {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>10s}")
    for name in class_names:
        lines.append(
            f"{name:>12s}  {test_metrics['precision_per_class'][name]:>10.4f} "
            f"{test_metrics['recall_per_class'][name]:>10.4f} {test_metrics['f1_per_class'][name]:>10.4f} "
            f"{test_metrics['support_per_class'][name]:>10.0f}"
        )
    lines.append("")
    lines.append(
        f"{'macro avg':>12s}  {test_metrics['macro_precision']:>10.4f} "
        f"{test_metrics['macro_recall']:>10.4f} {test_metrics['macro_f1']:>10.4f} {test_metrics['n_test']:>10d}"
    )
    lines.append(
        f"{'weighted avg':>12s}  {test_metrics['weighted_precision']:>10.4f} "
        f"{test_metrics['weighted_recall']:>10.4f} {test_metrics['weighted_f1']:>10.4f} {test_metrics['n_test']:>10d}"
    )
    lines.append("")

    lines.append("confusion matrix (rows=true, cols=predicted):")
    lines.append(" " * 12 + "".join(f"{name:>10s}" for name in class_names))
    for name, row in zip(class_names, test_metrics["confusion_matrix"]):
        lines.append(f"{name:>12s}" + "".join(f"{v:>10d}" for v in row))

    if "note" in test_metrics:
        lines.append("")
        lines.append(f"note: {test_metrics['note']}")
    lines.append("")
    lines.append(f"per-conversation scores (P[{test_metrics['score_class']}]): {test_metrics['score_csv']}")
    ex = test_metrics["example_dirs"]
    lines.append(f"top/bottom/near-0.5 example conversations: high={ex['high']} low={ex['low']} normal={ex['normal']}")
    lines.append(f"per confusion-matrix-cell example conversations ({len(ex['confusion'])} cells): {ex['confusion']}")
    lines.append("")
    return "\n".join(lines)


def _write_score_report(dataset, truths, preds, probs, class_names, attribute, output_dir, repo_root, top_n=10):
    """Score every held-out test conversation by the probe's probability for the
    "high" class (falls back to the highest-index class if an attribute's label
    set ever lacks "high"), write one row per conversation to
    <attribute>_test_scores.csv (file, source dataset dir, true/predicted label,
    score, sorted by score descending), and copy the `top_n` highest-scoring,
    lowest-scoring, and closest-to-0.5 ("normal") conversations' raw .txt files
    into eval/examples/<attribute>/{high,low,normal}/ so they can be read
    directly. Also buckets every conversation by its confusion-matrix cell
    (true_label, predicted_label) and copies up to `top_n` highest-scoring
    examples of each into eval/examples/<attribute>/confusion/true_X-pred_Y/,
    so misclassifications (and correct calls) can be read directly per cell.
    Re-running overwrites all of this, so it stays current every time the test
    script is run rather than accumulating stale output.
    """
    score_class = "high" if "high" in class_names else class_names[-1]
    score_idx = class_names.index(score_class)
    scores = probs[:, score_idx]

    n = len(scores)
    k = min(top_n, n)
    order_by_score = sorted(range(n), key=lambda i: -scores[i])
    high_idx = order_by_score[:k]
    low_idx = order_by_score[::-1][:k]
    normal_idx = sorted(range(n), key=lambda i: abs(scores[i] - 0.5))[:k]

    bucket_of = {}
    for bucket, idxs in (("high", high_idx), ("low", low_idx), ("normal", normal_idx)):
        for i in idxs:
            bucket_of.setdefault(i, bucket)

    csv_path = os.path.join(output_dir, f"{attribute}_test_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "file", "source_dir", "true_label", "predicted_label",
                          "score", "score_class", "bucket"])
        for rank, i in enumerate(order_by_score, start=1):
            file_path = dataset.used_file_paths[i]
            source_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            writer.writerow([
                rank, os.path.relpath(file_path, repo_root), source_dir,
                class_names[truths[i]], class_names[preds[i]],
                f"{scores[i]:.6f}", score_class, bucket_of.get(i, ""),
            ])

    example_dirs = {}
    for bucket, idxs in (("high", high_idx), ("low", low_idx), ("normal", normal_idx)):
        bucket_dir = os.path.join(output_dir, "examples", attribute, bucket)
        shutil.rmtree(bucket_dir, ignore_errors=True)
        os.makedirs(bucket_dir, exist_ok=True)
        for rank, i in enumerate(idxs, start=1):
            file_path = dataset.used_file_paths[i]
            source_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            dest_name = f"{rank:02d}_score{scores[i]:.3f}_{source_dir}_{os.path.basename(file_path)}"
            shutil.copy2(file_path, os.path.join(bucket_dir, dest_name))
        example_dirs[bucket] = os.path.relpath(bucket_dir, output_dir)

    confusion_root = os.path.join(output_dir, "examples", attribute, "confusion")
    shutil.rmtree(confusion_root, ignore_errors=True)
    cells = {}
    for i in range(n):
        cells.setdefault((truths[i], preds[i]), []).append(i)
    confusion_dirs = {}
    for (t, p), idxs in cells.items():
        cell_name = f"true_{class_names[t]}-pred_{class_names[p]}"
        cell_dir = os.path.join(confusion_root, cell_name)
        os.makedirs(cell_dir, exist_ok=True)
        cell_idxs = sorted(idxs, key=lambda i: -scores[i])[:k]
        for rank, i in enumerate(cell_idxs, start=1):
            file_path = dataset.used_file_paths[i]
            source_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            dest_name = f"{rank:02d}_score{scores[i]:.3f}_{source_dir}_{os.path.basename(file_path)}"
            shutil.copy2(file_path, os.path.join(cell_dir, dest_name))
        confusion_dirs[cell_name] = os.path.relpath(cell_dir, output_dir)
    example_dirs["confusion"] = confusion_dirs

    return os.path.relpath(csv_path, output_dir), score_class, example_dirs


def evaluate(args, probe_type):
    """Load checkpoint(s) written by train_attribute() (via train_reading_probe.py /
    train_control_probe.py) and score them on a novel held-out dataset.

    For each attribute, the label set the checkpoint was trained on is read back from
    <attribute>_metrics.json's "class_names" -- so this automatically matches whatever
    --ignore_missing_labels decided at train time, rather than assuming ATTRIBUTE_LABELS[attribute].
    Test examples whose label isn't in that set are skipped (reported). A trained label with zero
    examples in --test_dirs is unscoreable; with --ignore_missing_labels (default: on) it is excluded
    from the report entirely -- its column is masked out of the argmax so it can never be predicted,
    and it is dropped from the confusion matrix and the macro averages it would otherwise deflate
    with a meaningless 0 (the probe itself is untouched; the head width is fixed by the checkpoint,
    and "trained_class_names" in the output records it). With --no-ignore_missing_labels the full
    trained label set is scored and such classes report 0 precision/recall (zero_division=0).

    Results are written under <checkpoint_dir>/eval/ (or --output_dir): one meta.json recording the
    model/checkpoint/strategy/data used for the whole run, one <attribute>_test_metrics.json per
    attribute (scores plus that same metadata and the original training sources, so each file is
    self-contained) with its confusion-matrix PNG alongside it, and a test_summary.json across
    attributes. Also, per attribute (pooled across every --test_dirs source given): one
    <attribute>_test_scores.csv rating every scored test conversation by the probe's P(high) (file,
    source dataset dir, true/predicted label, score, sorted descending); copies of the 10
    highest-scoring, 10 lowest-scoring, and 10 closest-to-0.5 ("normal"/most-uncertain) conversations'
    raw .txt files under eval/examples/<attribute>/{high,low,normal}/; and, per confusion-matrix cell
    (true_label, predicted_label), copies of its up-to-10 highest-scoring conversations under
    eval/examples/<attribute>/confusion/true_<label>-pred_<label>/, so misclassifications can be read
    directly. All of this is regenerated (old contents replaced) on every run, so it stays current
    with whatever --checkpoint_dir/--test_dirs/--attributes were just passed.
    """
    control_probe = probe_type == "control"

    checkpoint_dir = os.path.join(args.repo_root, args.checkpoint_dir)
    test_paths = [os.path.join(args.repo_root, d) for d in args.test_dirs]
    output_dir = (os.path.join(args.repo_root, args.output_dir) if args.output_dir
                  else os.path.join(checkpoint_dir, "eval"))
    os.makedirs(output_dir, exist_ok=True)

    run_meta = {
        "model_name": args.model_name,
        "probe_type": probe_type,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_suffix": args.checkpoint_suffix,
        "test_dirs": args.test_dirs,
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    if args.attributes is not None:
        attributes = args.attributes
    else:
        attributes = sorted(
            attr for attr in ATTRIBUTE_LABELS
            if os.path.isfile(os.path.join(checkpoint_dir, f"{attr}_metrics.json"))
        )
        if not attributes:
            raise ValueError(f"no <attribute>_metrics.json found under {checkpoint_dir}")

    if args.layers is not None and len(args.layers) not in (1, len(attributes)):
        raise ValueError("--layers must have length 1 or match --attributes")

    print(f"Loading {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    model.to(args.device)
    model.eval()

    print(f"Evaluating checkpoints from {checkpoint_dir} on {args.test_dirs} -> {output_dir}")

    summary = {}
    for i, attribute in enumerate(attributes):
        metrics_path = os.path.join(checkpoint_dir, f"{attribute}_metrics.json")
        if not os.path.isfile(metrics_path):
            print(f"skipping {attribute}: no {metrics_path}")
            continue
        with open(metrics_path) as f:
            train_metrics = json.load(f)

        class_names = train_metrics["class_names"]
        label_to_id = {name: idx for idx, name in enumerate(class_names)}
        num_classes = len(label_to_id)

        if args.layers is None:
            layer = train_metrics["best_layer"]
        else:
            layer = args.layers[i] if len(args.layers) == len(attributes) else args.layers[0]

        ckpt_path = os.path.join(checkpoint_dir, f"{attribute}_probe_layer{layer}_{args.checkpoint_suffix}.pth")
        if not os.path.isfile(ckpt_path):
            print(f"skipping {attribute}: no checkpoint at {ckpt_path}")
            continue

        present = _discover_present_labels(test_paths, attribute)
        extra = present - set(label_to_id)
        missing = set(label_to_id) - present
        if extra:
            print(f"  [{attribute}] note: test data has label(s) {sorted(extra)} the checkpoint "
                  f"wasn't trained on -- those files are skipped")
        # Classes the checkpoint knows but --test_dirs can't exercise are dropped from the
        # *scoring*, not from the probe -- the head width is fixed by the checkpoint.
        scored_names = class_names
        if missing:
            print(f"  [{attribute}] note: checkpoint was trained on label(s) {sorted(missing)} with "
                  f"zero examples in --test_dirs -- those classes can't be scored here")
            if args.ignore_missing_labels:
                scored_names = [name for name in class_names if name not in missing]
                if len(scored_names) < 2:
                    print(f"skipping {attribute}: only {scored_names} scoreable in --test_dirs, too "
                          f"few to report (pass --no-ignore_missing_labels to score the full label "
                          f"set with 0s for the unscoreable classes).")
                    continue
                print(f"  [{attribute}] excluding them from the report and from the macro averages "
                      f"they would deflate; scoring on {scored_names} only (pass "
                      f"--no-ignore_missing_labels to keep them at 0 precision/recall).")

        dataset = _load_pooled_dataset(test_paths, attribute, tokenizer, model, args.device,
                                        control_probe, "test", label_to_id=label_to_id)
        if dataset is None or len(dataset) == 0:
            print(f"skipping {attribute}: no usable test examples found under --test_dirs")
            continue

        probe = LinearProbeClassification(args.device, num_classes)
        probe.load_state_dict(torch.load(ckpt_path, map_location=args.device, weights_only=True))

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        loss, acc, preds, truths, probs = run_epoch(probe, loader, args.device, layer, num_classes)

        n_reassigned = 0
        if scored_names != class_names:
            # Mask the unscoreable columns out of the argmax (a prediction that landed on one
            # is reassigned to the best scoreable class) and reindex truths/preds/probs into
            # the scored label space, so every metric below covers only scoreable classes.
            scored_ids = [label_to_id[name] for name in scored_names]
            n_reassigned = int(np.sum(~np.isin(preds, scored_ids)))
            remap = {old: new for new, old in enumerate(scored_ids)}
            probs = probs[:, scored_ids]
            preds = np.argmax(probs, axis=1)
            truths = np.array([remap[int(t)] for t in truths])
            acc = float(np.mean(preds == truths)) if len(truths) else 0.0
            class_names = scored_names
            if n_reassigned:
                print(f"  [{attribute}] {n_reassigned} prediction(s) had landed on an unscoreable "
                      f"class and were reassigned to the best scoreable one")

        precision, recall, f1, support = precision_recall_fscore_support(
            truths, preds, labels=list(range(len(class_names))), zero_division=0)
        report = classification_report(
            truths, preds, labels=list(range(len(class_names))), target_names=class_names,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(truths, preds, labels=list(range(len(class_names))))

        cm_path = os.path.join(output_dir, f"{attribute}_test_confusion_matrix_layer{layer}.png")
        plot_confusion(truths, preds, class_names, attribute, os.path.basename(checkpoint_dir),
                        f"{probe_type} (held-out test)", layer, cm_path)

        score_csv, score_class, example_dirs = _write_score_report(
            dataset, truths, preds, probs, class_names, attribute, output_dir, args.repo_root)

        test_metrics = {
            "attribute": attribute,
            "meta": run_meta,
            "train_sources": train_metrics.get("sources"),
            "checkpoint": ckpt_path,
            "layer": layer,
            "class_names": class_names,
            "trained_class_names": train_metrics["class_names"],
            "n_test": len(dataset),
            "loss": loss,
            "accuracy": acc,
            "precision_per_class": dict(zip(class_names, precision.tolist())),
            "recall_per_class": dict(zip(class_names, recall.tolist())),
            "f1_per_class": dict(zip(class_names, f1.tolist())),
            "support_per_class": dict(zip(class_names, support.tolist())),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "plots": {"confusion_matrix": os.path.relpath(cm_path, output_dir)},
            "score_class": score_class,
            "score_csv": score_csv,
            "example_dirs": example_dirs,
        }
        if missing:
            if scored_names != train_metrics["class_names"]:
                test_metrics["note"] = (
                    f"label(s) {sorted(missing)} had zero examples in --test_dirs; they were "
                    f"excluded from scoring (--ignore_missing_labels), so every metric here "
                    f"covers {class_names} only. {n_reassigned} prediction(s) that had landed "
                    f"on an excluded class were reassigned to the best scoreable class."
                )
            else:
                test_metrics["note"] = (
                    f"label(s) {sorted(missing)} had zero examples in --test_dirs; their "
                    "precision/recall are reported as 0 (zero_division=0), not a true measurement"
                )

        with open(os.path.join(output_dir, f"{attribute}_test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        with open(os.path.join(output_dir, f"{attribute}_test_metrics.txt"), "w") as f:
            f.write(_format_test_report(test_metrics))

        print(f"  [{attribute}] layer {layer} (n={len(dataset)}): acc={acc:.3f}  "
              f"macro P/R/F1={test_metrics['macro_precision']:.3f}/{test_metrics['macro_recall']:.3f}/"
              f"{test_metrics['macro_f1']:.3f}  "
              f"weighted P/R/F1={test_metrics['weighted_precision']:.3f}/{test_metrics['weighted_recall']:.3f}/"
              f"{test_metrics['weighted_f1']:.3f}")
        for name, p, r, f, s in zip(class_names, precision, recall, f1, support):
            print(f"      {name:>10s}: precision={p:.3f} recall={r:.3f} f1={f:.3f} support={s}")
        score_buckets = ", ".join(f"{b}={d}" for b, d in example_dirs.items() if b != "confusion")
        n_cells = len(example_dirs["confusion"])
        print(f"      scores: {score_csv}  examples: {score_buckets}, "
              f"confusion={os.path.join('examples', attribute, 'confusion')} ({n_cells} cells)  "
              f"(all under {output_dir})")

        summary[attribute] = {
            "layer": layer,
            "n_test": len(dataset),
            "accuracy": acc,
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_precision": test_metrics["weighted_precision"],
            "weighted_recall": test_metrics["weighted_recall"],
            "weighted_f1": test_metrics["weighted_f1"],
            "precision_per_class": test_metrics["precision_per_class"],
            "recall_per_class": test_metrics["recall_per_class"],
            "f1_per_class": test_metrics["f1_per_class"],
            "score_class": score_class,
            "score_csv": score_csv,
            "example_dirs": example_dirs,
        }

        del dataset, probe
        torch.cuda.empty_cache()

    with open(os.path.join(output_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
