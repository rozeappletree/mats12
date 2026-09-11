#!/usr/bin/env python3
"""Generate steered replies for the PractGULL (S, H) pairs.

For every conversation starter in a PractGULL-style .jsonl file -- a system
prompt ``S`` and a single human turn ``H`` -- this asks Llama-2-13b-chat for two
replies with the TalkTuner control-probe steering of ``src/steering.py`` applied:

  i.  gullibility steered **high**  (accept the presupposition, low friction)
  ii. gullibility steered **low**   (hedge + a concrete verify step)

"Best settings" == ``src/steering.py``'s own defaults: each probe directory's
empirically-best control-probe layer (from its ``summary.json``), a 13-layer
window centred on it ([best-6, best+7)), and ``n_scale = 7`` -- exactly what
``Steering(model, "gullibility", label)`` uses with no overrides. Override with
``--n-scale`` / ``--layers`` if you want to sweep.

Both probe checkpoint sets from the SeeGULL phase-2 work are run by default and
reported side by side, keyed by the top-level checkpoint directory name:

  * ``probe_checkpoints.withRegularGullibility``  (SeeGULL v0.1, trained on the
    regular-gullibility slice; control probes under
    ``control_probe/regular_gullibility_170_custom_split/``)
  * ``probe_checkpoints.withDefense484Only``      (SeeGULL v0.2, trained on the
    484 non-hard-negative defence conversations; control probes under
    ``control_probe/``)

The resolver walks each ``--probe-dir`` and its ``control_probe/`` subtree to
find the directory that actually holds ``gullibility_probe_layer*_best.pth``.

INPUT   one directory of ``*.jsonl`` files, each line an object with at least
        ``S`` and ``H`` (extra keys are copied through untouched). Files already
        named ``steeredoutput.*`` are skipped.

OUTPUT  ``steeredoutput.<original-name>`` written next to each input file. Every
        output line is the input object plus::

            "steered": {
              "<probe-dir-name>": {
                "high": {"reply": "...", "best_layer": 10, "from_idx": 4, "to_idx": 17, "n_scale": 7.0},
                "low":  {"reply": "...", ...}
              },
              ...
            }

SETUP
  conda activate talktuner-gpu     # torch + transformers + baukit, CUDA GPU

USAGE
  python src/test_steering.py
  python src/test_steering.py --data-dir data/PractGULL \\
      --probe-dir probe_checkpoints.withDefense484Only \\
      --probe-dir probe_checkpoints.withRegularGullibility
  python src/test_steering.py --n-scale 9 --max-new-tokens 200 --limit 20

This script does not train or evaluate anything -- it only samples generations.
"""

import argparse
import glob
import json
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from probe_common import llama_v2_prompt  # noqa: E402
from steering import DEFAULT_N_SCALE, Steering  # noqa: E402

DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "PractGULL")
DEFAULT_PROBE_DIRS = [
    os.path.join(REPO_ROOT, "probe_checkpoints.withRegularGullibility"),
    os.path.join(REPO_ROOT, "probe_checkpoints.withDefense484Only"),
]
DEFAULT_MODEL_NAME = "NousResearch/Llama-2-13b-chat-hf"
ATTRIBUTE = "gullibility"
OUTPUT_PREFIX = "steeredoutput."

# When a probe dir has several candidate control-probe subdirectories, prefer
# the SeeGULL-v0.1 custom split over the generic smoke-test sample.
_SUBDIR_PREFERENCE = ("regular_gullibility", "custom_split", "defense")


def resolve_control_probe_dir(top):
    """Return the directory under ``top`` that holds
    ``gullibility_probe_layer*_best.pth`` checkpoints (plus ``summary.json`` /
    ``gullibility_metrics.json``), searching ``top`` itself, ``top/control_probe``,
    and one level below ``top/control_probe``."""
    top = os.path.abspath(top)
    candidates = [top, os.path.join(top, "control_probe")]
    control_probe = os.path.join(top, "control_probe")
    if os.path.isdir(control_probe):
        subs = [os.path.join(control_probe, d) for d in os.listdir(control_probe)]
        subs = [d for d in subs if os.path.isdir(d)]

        def rank(path):
            name = os.path.basename(path).lower()
            for i, key in enumerate(_SUBDIR_PREFERENCE):
                if key in name:
                    return (i, name)
            return (len(_SUBDIR_PREFERENCE), name)

        candidates.extend(sorted(subs, key=rank))

    for cand in candidates:
        if glob.glob(os.path.join(cand, f"{ATTRIBUTE}_probe_layer*_best.pth")):
            return cand
    raise FileNotFoundError(
        f"no {ATTRIBUTE}_probe_layer*_best.pth checkpoints found under {top} "
        f"(looked in: {', '.join(candidates)})"
    )


def parse_layers(spec):
    """``"12,25"`` -> ``(None, (12, 25))`` (applies to every probe dir);
    ``"withDefense484Only=12,25"`` -> ``("withDefense484Only", (12, 25))``
    (applies only to the probe dir whose name matches)."""
    name, _, rest = spec.rpartition("=")
    parts = rest.replace("-", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--layers must be FROM,TO or NAME=FROM,TO (e.g. 12,25 or withDefense484Only=12,25)")
    try:
        return (name or None), (int(parts[0]), int(parts[1]))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--layers bounds must be integers, got {rest!r}")


def parse_scale(spec):
    """``"7"`` -> ``(None, 7.0)``; ``"withDefense484Only=4"`` -> ``("withDefense484Only", 4.0)``."""
    name, _, rest = spec.rpartition("=")
    try:
        return (name or None), float(rest)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--n-scale must be a number or NAME=number, got {spec!r}")


def lookup_override(overrides, probe_dir, default=None):
    """Most specific match for `probe_dir` in an {name-or-None: value} map: an
    exact basename match, then any name that is a substring of the basename,
    then the global (None) entry."""
    base = os.path.basename(os.path.normpath(probe_dir))
    if base in overrides:
        return overrides[base]
    for name, value in overrides.items():
        if name and name in base:
            return value
    return overrides.get(None, default)


@torch.no_grad()
def generate(model, tokenizer, system_prompt, human, device, gen_kwargs, steer):
    """One greedy (or sampled) generation for (system_prompt, human) with
    ``steer`` applied for the whole decode."""
    text = llama_v2_prompt([{"role": "user", "content": human}], system_prompt)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=3500)
    enc = {k: v.to(device) for k, v in enc.items()}
    with steer.context():
        out = model.generate(**enc, **gen_kwargs)
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_records(path):
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
            if "S" not in obj or "H" not in obj:
                raise ValueError(f"{path}:{lineno}: line has no 'S'/'H' keys")
            records.append(obj)
    return records


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="directory of PractGULL *.jsonl files (default: data/PractGULL)")
    ap.add_argument("--input", action="append", dest="inputs", default=None, metavar="FILE",
                    help="generate for just this file; repeatable (default: every *.jsonl under "
                         "--data-dir). Output is written next to each input file.")
    ap.add_argument("--probe-dir", action="append", dest="probe_dirs", default=None,
                    metavar="DIR", help="top-level probe checkpoint dir; repeatable "
                    "(default: withRegularGullibility + withDefense484Only)")
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--n-scale", type=parse_scale, action="append", default=None,
                    metavar="[NAME=]N",
                    help=f"steering magnitude (default: {DEFAULT_N_SCALE}, TalkTuner's fixed "
                         f"default). Prefix with a probe dir name to set it for just that dir, "
                         f"e.g. --n-scale withDefense484Only=4; repeatable")
    ap.add_argument("--layers", type=parse_layers, action="append", default=None,
                    metavar="[NAME=]FROM,TO",
                    help="override the [from, to) decoder-block steering window (default: each "
                         "dir's best-layer window). Prefix with a probe dir name to set it for "
                         "just that dir, e.g. --layers withDefense484Only=12,25; repeatable. "
                         "NOTE: a checkpoint set whose best layer sits early (withDefense484Only's "
                         "is 10, an arbitrary tie-break in a curve that is flat to layer 40) "
                         "degenerates at n_scale=7 -- steer it at a later window instead")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--sample", action="store_true",
                    help="stochastic sampling instead of greedy decoding")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N lines of each file (smoke test)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the steeredoutput.* file already exists")
    args = ap.parse_args()

    probe_dirs = args.probe_dirs or DEFAULT_PROBE_DIRS
    data_dir = os.path.abspath(args.data_dir)
    if args.inputs:
        jsonl_files = [os.path.abspath(p) for p in args.inputs]
        missing = [p for p in jsonl_files if not os.path.isfile(p)]
        if missing:
            ap.error(f"--input file(s) do not exist: {', '.join(missing)}")
    else:
        if not os.path.isdir(data_dir):
            ap.error(f"--data-dir does not exist: {data_dir}")
        jsonl_files = sorted(
            p for p in glob.glob(os.path.join(data_dir, "*.jsonl"))
            if not os.path.basename(p).startswith(OUTPUT_PREFIX)
        )
        if not jsonl_files:
            ap.error(f"no input *.jsonl files in {data_dir}")

    resolved = {}  # top-level probe dir -> control-probe checkpoint dir
    for top in probe_dirs:
        resolved[top] = resolve_control_probe_dir(top)
        print(f"[probe] {os.path.basename(os.path.normpath(top))} -> "
              f"{os.path.relpath(resolved[top], REPO_ROOT)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no CUDA device visible; fp16 on CPU will be extremely slow.")

    print(f"[..] loading {args.model} in fp16 (this takes a few minutes on first run)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    layer_overrides = dict(args.layers or [])
    scale_overrides = dict(args.n_scale or [])

    # One Steering per (probe dir, target label); each loads its control probes once.
    steerers = {}  # top-level probe dir -> {"high": Steering, "low": Steering}
    for top, ckpt_dir in resolved.items():
        window = lookup_override(layer_overrides, top)
        from_idx, to_idx = window if window else (None, None)
        n_scale = lookup_override(scale_overrides, top, DEFAULT_N_SCALE)
        steerers[top] = {}
        for label in ("high", "low"):
            steer = Steering(model, ATTRIBUTE, label, n_scale=n_scale,
                             probe_dir=ckpt_dir, from_idx=from_idx, to_idx=to_idx,
                             device=device)
            steerers[top][label] = steer
            print(f"[steer] {os.path.basename(os.path.normpath(top))} {label}: {steer.describe()}")

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if args.sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)

    def steer_meta(steer):
        return dict(best_layer=steer.best_layer, from_idx=steer.from_idx,
                    to_idx=steer.to_idx, n_scale=steer.n_scale)

    run_started = time.time()
    for in_path in jsonl_files:
        fname = os.path.basename(in_path)
        out_path = os.path.join(os.path.dirname(in_path), OUTPUT_PREFIX + fname)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {os.path.basename(out_path)} exists (use --overwrite)")
            continue

        records = load_records(in_path)
        if args.limit is not None:
            records = records[: args.limit]

        # Generations go to a .partial file that is renamed into place only once
        # the file is complete, so an interrupted run (this is a many-hour job)
        # resumes at the record it stopped on instead of starting over. Only
        # whole, parseable lines count as done -- a line truncated by a kill
        # mid-write is dropped and regenerated.
        partial_path = out_path + ".partial"
        done = 0
        if os.path.exists(partial_path):
            if args.overwrite:
                os.remove(partial_path)
            else:
                kept = []
                with open(partial_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            break  # truncated tail; stop here
                        kept.append(line)
                with open(partial_path, "w") as f:
                    for line in kept:
                        f.write(line + "\n")
                done = len(kept)
                if done:
                    print(f"[resume] {os.path.basename(partial_path)} has {done} complete "
                          f"record(s); continuing from record {done + 1}")

        file_started = time.time()
        print(f"[file] {fname}: {len(records)} record(s) -> {os.path.basename(out_path)}", flush=True)
        if done >= len(records):
            os.replace(partial_path, out_path)
            print(f"[ok] {os.path.basename(out_path)} already complete")
            continue

        # (top, label, S, H) -> reply, so the -P / -Q lines of a pair (identical
        # S and H) are only generated once.
        cache = {}
        with open(partial_path, "a" if done else "w") as out_f:
            for i, obj in enumerate(records, 1):
                if i <= done:
                    continue
                system_prompt, human = obj["S"], obj["H"]
                steered = {}
                for top, by_label in steerers.items():
                    key_name = os.path.basename(os.path.normpath(top))
                    steered[key_name] = {}
                    for label, steer in by_label.items():
                        cache_key = (top, label, system_prompt, human)
                        if cache_key not in cache:
                            cache[cache_key] = generate(
                                model, tokenizer, system_prompt, human,
                                device, gen_kwargs, steer,
                            )
                        steered[key_name][label] = dict(
                            reply=cache[cache_key], **steer_meta(steer)
                        )
                out_f.write(json.dumps({**obj, "steered": steered}, ensure_ascii=False) + "\n")
                out_f.flush()
                if i % 10 == 0 or i == len(records):
                    elapsed = time.time() - file_started
                    rate = (i - done) / elapsed if elapsed else 0
                    eta = (len(records) - i) / rate if rate else 0
                    print(f"  {i}/{len(records)} ({100 * i / len(records):.1f}%) "
                          f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m", flush=True)

        os.replace(partial_path, out_path)
        print(f"[ok] wrote {os.path.basename(out_path)} "
              f"({len(records)} record(s), {time.time() - file_started:.0f}s)", flush=True)

    print(f"[done] {len(jsonl_files)} file(s) in {(time.time() - run_started) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
