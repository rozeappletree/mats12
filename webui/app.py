#!/usr/bin/env python3
"""
webui/app.py -- Flask front end for the same steered-chat session as
scripts/chat_steered.py: one persistent conversation with any number of
TalkTuner control-probe attributes steered at once, reusing chat_steered.py's
SteeringState / generate / compute_user_scores wholesale instead of
reimplementing the steering recipe.

Single-session, single-process, like the REPL: state lives in module-level
globals, the model is loaded once at startup, and the whole app is meant for
one person driving it from a browser -- not a multi-user service.

SETUP
  conda activate talktuner-gpu

USAGE
  python webui/app.py
  python webui/app.py --steer gullibility=high --steer certainty_seeking=low
  python webui/app.py --port 5050 --no-scores

USAGE (4 CLASSES)
    python  webui/app.py  \
        --probe-dir /root/SeeGULL/mats12/probe_checkpoints.withLLaMaOpusSol/control_probe/llama2_sample2+claudeopus_sample2+sol_100 \
        --reading-probe-dir /root/SeeGULL/mats12/probe_checkpoints.withLLaMaOpusSol/reading_probe/llama2_sample2+claudeopus_sample2+sol_100/

USAGE (SeeGULL v0.1)
    python  webui/app.py  \
        --probe-dir  /root/SeeGULL/mats12/probe_checkpoints.withRegularGullibility/control_probe/regular_gullibility_170_custom_split\
        --reading-probe-dir /root/SeeGULL/mats12/probe_checkpoints.withRegularGullibility/reading_probe/regular_gullibility_170_custom_split
"""

import argparse
import os
import sys

from flask import Flask, jsonify, render_template, request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import chat_steered as cs  # noqa: E402

app = Flask(__name__)

# Populated once by create_session() in main(); a plain namespace-ish dict
# rather than a class since it's just the REPL's local variables lifted to
# module scope so route handlers can reach them.
SESSION = {}


def attribute_status(attribute):
    cfg = SESSION["state"].attrs.get(attribute)
    info = SESSION["summary"].get(attribute, {})
    default_from, default_to, best_layer = cs.default_window(attribute, SESSION["summary"])
    top_layers = [{"layer": layer, "acc": acc}
                  for layer, acc in SESSION["layer_accuracies"].get(attribute, [])[:4]]
    if cfg is None:
        return {
            "attribute": attribute,
            "classes": cs.class_names(attribute),
            "loaded": False,
            "active": False,
            "target_label": None,
            "from_idx": default_from,
            "to_idx": default_to,
            "n_scale": cs.DEFAULT_N_SCALE,
            "best_layer": info.get("best_layer", best_layer),
            "best_acc": info.get("best_acc"),
            "is_best_window": True,
            "top_layers": top_layers,
        }
    return {
        "attribute": attribute,
        "classes": cfg.labels,
        "loaded": True,
        "active": cfg.target_label is not None,
        "target_label": cfg.target_label,
        "from_idx": cfg.from_idx,
        "to_idx": cfg.to_idx,
        "n_scale": cfg.n_scale,
        "best_layer": info.get("best_layer", cfg.best_layer),
        "best_acc": info.get("best_acc"),
        "is_best_window": (cfg.from_idx, cfg.to_idx) == (default_from, default_to),
        "top_layers": top_layers,
    }


def full_status():
    return {
        "attributes": [attribute_status(a) for a in cs.ATTRIBUTE_LABELS],
        "system_prompt": SESSION["system_prompt"],
        "show_scores": SESSION["show_scores"],
        "num_messages": len(SESSION["messages"]),
        "reading_probes_loaded": sorted(SESSION["reading_probes"]),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    return jsonify(full_status())


@app.get("/api/messages")
def api_messages():
    return jsonify(messages=SESSION["messages"])


@app.post("/api/steer")
def api_steer():
    body = request.get_json(force=True)
    attribute = body.get("attribute")
    label = body.get("label")
    if not attribute:
        return jsonify(error="attribute is required"), 400
    try:
        if label in (None, "off"):
            SESSION["state"].clear_target(attribute)
        else:
            SESSION["state"].set_target(attribute, label)
    except (ValueError, FileNotFoundError) as e:
        return jsonify(error=str(e)), 400
    return jsonify(full_status())


@app.post("/api/steer/all_off")
def api_steer_all_off():
    SESSION["state"].clear_target()
    return jsonify(full_status())


@app.post("/api/layers")
def api_layers():
    body = request.get_json(force=True)
    attribute = body.get("attribute")
    if not attribute:
        return jsonify(error="attribute is required"), 400
    try:
        if body.get("best"):
            SESSION["state"].reset_layers_to_best(attribute)
        elif "layer" in body:
            from_idx, to_idx = cs.window_around(int(body["layer"]))
            SESSION["state"].set_layers(attribute, from_idx, to_idx)
        else:
            from_idx, to_idx = int(body["from_idx"]), int(body["to_idx"])
            if from_idx >= to_idx:
                return jsonify(error="from_idx must be less than to_idx"), 400
            SESSION["state"].set_layers(attribute, from_idx, to_idx)
    except (ValueError, FileNotFoundError, KeyError) as e:
        return jsonify(error=str(e)), 400
    return jsonify(full_status())


@app.post("/api/scale")
def api_scale():
    body = request.get_json(force=True)
    attribute = body.get("attribute")
    if not attribute:
        return jsonify(error="attribute is required"), 400
    try:
        SESSION["state"].set_scale(attribute, float(body["n_scale"]))
    except (ValueError, FileNotFoundError, KeyError) as e:
        return jsonify(error=str(e)), 400
    return jsonify(full_status())


@app.get("/api/system")
def api_get_system():
    return jsonify(system_prompt=SESSION["system_prompt"],
                   default_system_prompt=SESSION["default_system_prompt"])


@app.post("/api/system")
def api_set_system():
    body = request.get_json(force=True)
    prompt = body.get("prompt", "")
    if not prompt:
        return jsonify(error="prompt is required"), 400
    SESSION["system_prompt"] = prompt
    SESSION["messages"] = []
    return jsonify(full_status())


@app.post("/api/reset")
def api_reset():
    SESSION["messages"] = []
    return jsonify(full_status())


@app.post("/api/scores/toggle")
def api_scores_toggle():
    body = request.get_json(force=True)
    SESSION["show_scores"] = bool(body.get("on", True))
    return jsonify(full_status())


@app.get("/api/scores")
def api_scores():
    if not SESSION["reading_probes"]:
        return jsonify(error="no reading probes loaded"), 400
    results = cs.compute_user_scores(SESSION["model"], SESSION["tokenizer"], SESSION["messages"],
                                      SESSION["reading_probes"], SESSION["device"])
    return jsonify(scores=results)


@app.post("/api/save")
def api_save():
    body = request.get_json(force=True)
    name = body.get("name")
    if not name:
        return jsonify(error="name is required"), 400
    if os.path.basename(name) != name:
        return jsonify(error="name must not contain path separators"), 400
    if not SESSION["messages"]:
        return jsonify(error="nothing to save yet"), 400
    cs.save_conversation(name, SESSION["system_prompt"], SESSION["messages"], SESSION["steering_log"])
    path = name if os.path.splitext(name)[1] else name + ".json"
    return jsonify(saved_to=os.path.join("data", "manual.conversations", path))


@app.post("/api/chat")
def api_chat():
    body = request.get_json(force=True)
    text = (body.get("message") or "").strip()
    if not text:
        return jsonify(error="message is required"), 400

    SESSION["messages"].append({"role": "user", "content": text})
    reply = cs.generate(SESSION["model"], SESSION["tokenizer"], SESSION["messages"],
                         SESSION["system_prompt"], SESSION["device"], SESSION["gen_args"], SESSION["state"])
    SESSION["messages"].append({"role": "assistant", "content": reply})
    active_config = SESSION["state"].config_dict()
    SESSION["steering_log"].append(active_config)

    scores = None
    if SESSION["show_scores"] and SESSION["reading_probes"]:
        scores = cs.compute_user_scores(SESSION["model"], SESSION["tokenizer"], SESSION["messages"],
                                         SESSION["reading_probes"], SESSION["device"])

    return jsonify(reply=reply, active=active_config, scores=scores)


def create_session(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no CUDA device visible; fp16 on CPU will be extremely slow.")

    print(f"[..] loading {args.model} in fp16 (this takes a few minutes on first run)")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    cs.init_attribute_labels(args.probe_dir, args.reading_probe_dir)


    print("PROBE DIR (labels):", args.probe_dir, args.reading_probe_dir)

    summary = cs.load_summary(args.probe_dir)
    layer_accuracies = cs.load_layer_accuracies(args.probe_dir)
    state = cs.SteeringState(args.probe_dir, summary, device)

    reading_summary = cs.load_summary(args.reading_probe_dir)
    reading_probes = cs.load_reading_probes(args.reading_probe_dir, reading_summary, device)

    for spec in args.steer:
        if "=" not in spec:
            raise SystemExit(f"--steer must be ATTRIBUTE=LABEL, got {spec!r}")
        attribute, label = spec.split("=", 1)
        state.set_target(attribute, label)

    gen_args = argparse.Namespace(max_new_tokens=args.max_new_tokens, sample=args.sample,
                                   temperature=args.temperature, top_p=args.top_p)

    SESSION.update(
        model=model, tokenizer=tokenizer, device=device, state=state, summary=summary,
        layer_accuracies=layer_accuracies,
        reading_probes=reading_probes, show_scores=not args.no_scores,
        system_prompt=args.system_prompt, default_system_prompt=args.system_prompt,
        messages=[], steering_log=[], gen_args=gen_args,
    )
    print("[ok] model loaded, web UI ready")


def main():
    ap = argparse.ArgumentParser(description="Web UI for chatting with Llama-2-13b-chat, steered by control probes.")
    ap.add_argument("--model", default=cs.DEFAULT_MODEL_NAME)
    ap.add_argument("--probe-dir", default=cs.DEFAULT_PROBE_DIR)
    ap.add_argument("--reading-probe-dir", default=cs.DEFAULT_READING_PROBE_DIR)
    ap.add_argument("--no-scores", action="store_true", help="don't compute user scores after every turn")
    ap.add_argument("--steer", action="append", default=[], metavar="ATTRIBUTE=LABEL",
                     help="start with steering enabled toward this class; repeatable")
    ap.add_argument("--system-prompt", default=cs.DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--sample", action="store_true", help="stochastic sampling instead of greedy decoding")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    create_session(args)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
