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
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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

    def __init__(self, directories, tokenizer, model, attribute, device="cuda", control_probe=False):
        self.attribute = attribute
        self.control_probe = control_probe
        self.label_idf = f"_{attribute}_"
        self.label_to_id = ATTRIBUTE_LABELS[attribute]
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
    all_preds, all_targets = [], []

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

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, np.concatenate(all_preds), np.concatenate(all_targets)


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
                     batch_size, test_size, seed, device):
    num_classes = len(ATTRIBUTE_LABELS[attribute])
    class_names = [name for name, _ in sorted(ATTRIBUTE_LABELS[attribute].items(), key=lambda kv: kv[1])]

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
            train_loss, train_acc, _, _ = run_epoch(probe, train_loader, device, layer, num_classes, optimizer)
            test_loss, test_acc, test_preds, test_truths = run_epoch(probe, test_loader, device, layer, num_classes)
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

    return {
        "best_layer": best_layer,
        "best_acc_per_layer": dict(zip(layers, best_acc_per_layer)),
        "final_acc_per_layer": dict(zip(layers, final_acc_per_layer)),
        "history_by_layer": history_by_layer,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
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
                         help="Dataset roots to pool together into one training set per attribute.")
    parser.add_argument("--attributes", nargs="+", default=list(ATTRIBUTE_LABELS.keys()),
                         choices=list(ATTRIBUTE_LABELS.keys()))
    parser.add_argument("--model_name", default="NousResearch/Llama-2-13b-chat-hf")
    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(NUM_LAYERS)))
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    return parser


def run(args, probe_type):
    """probe_type: 'reading' (appends the completion prompt) or 'control'
    (no appended prompt, last user-message token)."""
    control_probe = probe_type == "control"
    torch.manual_seed(args.seed)

    dataset_paths = [os.path.join(args.repo_root, d) for d in args.dataset_dirs]
    source_tags = [os.path.basename(d.rstrip("/")).removeprefix("datasets_") for d in args.dataset_dirs]
    combined_tag = "+".join(source_tags)

    out_dir = os.path.join(args.repo_root, args.output_dir, combined_tag)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Loading {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    model.to(args.device)
    model.eval()

    print(f"Pooling sources {args.dataset_dirs} -> {out_dir}")

    summary = {}
    for attribute in args.attributes:
        attr_dirs = [os.path.join(p, attribute) for p in dataset_paths if os.path.isdir(os.path.join(p, attribute))]
        for p in dataset_paths:
            if not os.path.isdir(os.path.join(p, attribute)):
                print(f"  note: {os.path.join(p, attribute)} not found, skipping that source for {attribute}")
        if not attr_dirs:
            print(f"skipping {attribute}: no source directories found")
            continue

        print(f"=== {attribute} [{probe_type}] (sources: {', '.join(attr_dirs)}) ===")
        dataset = TextDataset(attr_dirs, tokenizer, model, attribute, device=args.device,
                               control_probe=control_probe)
        if len(dataset) < 2 * len(ATTRIBUTE_LABELS[attribute]):
            print(f"  too few usable examples ({len(dataset)}), skipping")
            del dataset
            continue

        result = train_attribute(
            dataset, attribute, combined_tag, probe_type, out_dir, plot_dir, args.layers, args.max_epochs,
            args.batch_size, args.test_size, args.seed, args.device,
        )
        with open(os.path.join(out_dir, f"{attribute}_history.pkl"), "wb") as f:
            pickle.dump(result, f)

        best_layer = result["best_layer"]
        metrics = {
            "attribute": attribute,
            "probe_type": probe_type,
            "sources": args.dataset_dirs,
            "class_names": [name for name, _ in sorted(ATTRIBUTE_LABELS[attribute].items(), key=lambda kv: kv[1])],
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
        with open(os.path.join(out_dir, f"{attribute}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        summary[attribute] = {
            "best_layer": best_layer,
            "best_acc": metrics["best_acc"],
            "n_examples": metrics["n_examples"],
            "n_train": metrics["n_train"],
            "n_test": metrics["n_test"],
        }

        del dataset
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
