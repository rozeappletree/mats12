"""Shared machinery for causal-intervention ("activation steering") tests on
the persona-attribute control probes trained by src/train_control_probe.py.

Ported from TalkTuner's (Chen et al. 2024)
notebooks/causality_tests/causality_test_on_gender.run.ipynb and
src/intervention_utils.py: for a window of residual-stream layers, add
`n_scale * (target_one_hot @ probe.weight)` to the activation at the last
token position on every generation step, then let the model generate a
response. Steering toward class c pushes the residual stream in the
direction the control probe uses to detect class c, so if the probe (and the
underlying representation) is causally load-bearing, the model's behavior
should shift toward how it treats a user it believes is class c.

Usage (see the four causality_test_on_*.ipynb notebooks for full examples):

    tokenizer, model = load_model()
    probes = load_control_probes("gullibility", device="cuda")
    layer_names = which_layers(model, from_idx=10, to_idx=20)

    baseline = generate_responses(model, tokenizer, questions, layer_names)
    steered = generate_responses(model, tokenizer, questions, layer_names,
                                  edit_output=make_steering_hook(probes, one_hot(2, 3)))
"""

import json
import os
import random
import sys
from contextlib import nullcontext

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from probe_common import ATTRIBUTE_LABELS, HIDDEN_DIM, NUM_LAYERS, LinearProbeClassification, llama_v2_prompt  # noqa: E402

from baukit import TraceDict  # noqa: E402

DEFAULT_MODEL_NAME = "NousResearch/Llama-2-13b-chat-hf"
DEFAULT_PROBE_DIR = os.path.join(REPO_ROOT, "probe_checkpoints", "control_probe", "llama2_sample+claudeopus_sample")
TRUTHFULQA_PATH = os.path.join(REPO_ROOT, "data", "truthfulqa", "truthful_qa.json")
QUESTIONS_DIR = os.path.join(REPO_ROOT, "nb", "causality_tests", "questions")
RESULTS_DIR = os.path.join(REPO_ROOT, "nb", "causality_tests", "intervention_results")


def class_names(attribute):
    return [name for name, _ in sorted(ATTRIBUTE_LABELS[attribute].items(), key=lambda kv: kv[1])]


def one_hot(class_idx, num_classes):
    vec = [0.0] * num_classes
    vec[class_idx] = 1.0
    return torch.tensor([vec])


# --------------------------------------------------------------------------
# Model / probe loading
# --------------------------------------------------------------------------

def load_model(model_name=DEFAULT_MODEL_NAME, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    if "<pad>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def load_control_probes(attribute, probe_dir=DEFAULT_PROBE_DIR, checkpoint="best", device="cuda"):
    """Returns {hidden_states_layer_index: LinearProbeClassification}, where
    layer index 0 is the embedding output and i>=1 is the output of decoder
    block i-1 -- matching src/probe_common.py's TextDataset activation
    indexing.
    """
    num_classes = len(ATTRIBUTE_LABELS[attribute])
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


# --------------------------------------------------------------------------
# Steering
# --------------------------------------------------------------------------

def make_steering_hook(probes, cf_target, n_scale=7.0):
    """cf_target: one-hot tensor [1, num_classes] (see one_hot()). Returns a
    baukit-compatible edit_output(output, layer_name) that nudges the
    residual stream at the last token position toward the probe's direction
    for the target class, at every intervened layer and every generation
    step (i.e. it keeps steering token-by-token during autoregressive
    decoding, not just once on the prompt).

    The nudge is `n_scale * (cf_target @ probe.weight)` -- TalkTuner's own
    fixed-magnitude recipe (src/intervention_utils.py, N=7 for their gender
    probe over layers 19-29). Left as-is rather than adapted: this repo's
    probes are only trained on smoke-test-sized sample data right now, so any
    tuning of the steering magnitude (e.g. to a layer window far from
    TalkTuner's) belongs after a real dataset exists, not before.
    """
    cf_target = cf_target.to(torch.float)

    def edit_output(output, layer_name):
        layer_num = int(layer_name.rsplit(".", 1)[-1])
        probe = probes.get(layer_num + 1)
        if probe is None:
            return output
        weight = probe.proj[0].weight.detach().to(torch.float)  # [num_classes, hidden_dim]
        direction = (cf_target.to(weight.device) @ weight) * n_scale  # [1, hidden_dim]
        hidden = output[0]
        hidden[:, -1] = (hidden[:, -1].to(torch.float) + direction).to(hidden.dtype)
        return output

    return edit_output


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate_responses(model, tokenizer, questions, layer_names=(), edit_output=None,
                        batch_size=5, max_new_tokens=256, device="cuda"):
    responses = []
    use_hook = bool(layer_names) and edit_output is not None
    for i in tqdm(range(0, len(questions), batch_size), desc="generating"):
        batch_questions = questions[i:i + batch_size]
        prompts = [llama_v2_prompt([{"role": "user", "content": q}]) for q in batch_questions]

        context = TraceDict(model, list(layer_names), edit_output=edit_output) if use_hook else nullcontext()
        with context:
            with torch.no_grad():
                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         temperature=1.0, top_p=1.0)

        for seq in tokens:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            responses.append(text.split("[/INST]")[-1].strip())
    return responses


# --------------------------------------------------------------------------
# Question sources
# --------------------------------------------------------------------------

def load_truthfulqa_subset(categories, n=10, seed=0):
    """Returns a deterministic sample of n TruthfulQA records (question,
    correct_answers, incorrect_answers, ...) drawn from the given categories.
    """
    with open(TRUTHFULQA_PATH) as f:
        data = json.load(f)
    pool = [r for r in data if r["category"] in categories]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def load_plain_questions(attribute):
    path = os.path.join(QUESTIONS_DIR, f"{attribute}.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------

def save_intervention_results(attribute, questions, responses_by_condition, config):
    """questions: list[str] or list[dict] (TruthfulQA records).
    responses_by_condition: {condition_name: [response, ...]} same order/length as questions.
    """
    out_dir = os.path.join(RESULTS_DIR, attribute)
    os.makedirs(out_dir, exist_ok=True)

    question_texts = [q["question"] if isinstance(q, dict) else q for q in questions]

    for i, q_text in enumerate(question_texts):
        lines = [f"USER: {q_text}", "-" * 50]
        for condition, responses in responses_by_condition.items():
            lines.append(f"Intervened: {condition}")
            lines.append(f"CHATBOT: {responses[i]}")
            lines.append("-" * 50)
        with open(os.path.join(out_dir, f"{attribute}_question_{i + 1}.txt"), "w") as f:
            f.write("\n".join(lines))

    payload = {
        "attribute": attribute,
        "config": config,
        "questions": questions,
        "responses": responses_by_condition,
    }
    with open(os.path.join(out_dir, f"{attribute}_responses.json"), "w") as f:
        json.dump(payload, f, indent=2)

    return out_dir
