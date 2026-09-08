#!/usr/bin/env python3
"""
chat_steered_tui.py -- full-screen terminal UI for the same steered-chat
session as chat_steered.py / webui/app.py: a chat pane plus a live sidebar
showing every attribute's steering state and the user's reading-probe
scores. Reuses chat_steered.py's SteeringState / generate /
compute_user_scores / save_conversation wholesale instead of reimplementing
the steering recipe -- this is a different front end on the same session,
not a different implementation of it.

Same slash commands as the chat_steered.py REPL, typed into the input bar
at the bottom; anything not starting with "/" is sent as a chat message.
Generation runs in a background thread so the UI stays responsive (you can
still scroll, and the status bar shows "generating...") while the model
works.

SETUP
  conda activate talktuner-gpu

USAGE
  python scripts/chat_steered_tui.py
  python scripts/chat_steered_tui.py --steer gullibility=high --steer certainty_seeking=low

KEYS
  Enter               send message / run command (lines starting with /)
  Tab / Shift-Tab      switch focus between input and chat pane
  Up/Down/PageUp/PageDown, mouse wheel   scroll the chat pane
  Ctrl-C               quit

COMMANDS (same as chat_steered.py's REPL)
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
import asyncio
import os
import sys

from prompt_toolkit import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.widgets import Frame

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import chat_steered as cs  # noqa: E402


class Session:
    """Everything the UI reads/mutates -- the full-screen-app equivalent of
    the REPL's local variables in chat_steered.main(), lifted onto an object
    so callbacks can reach them."""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.device = None
        self.state = None
        self.summary = {}
        self.reading_probes = {}
        self.show_scores = not args.no_scores
        self.system_prompt = args.system_prompt
        self.messages = []
        self.steering_log = []
        self.latest_scores = None
        self.busy = False
        self.gen_args = argparse.Namespace(max_new_tokens=args.max_new_tokens, sample=args.sample,
                                            temperature=args.temperature, top_p=args.top_p)

    def load_model(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("[warn] no CUDA device visible; fp16 on CPU will be extremely slow.")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model, torch_dtype=torch.float16, device_map="auto")
        self.model.eval()

        self.summary = cs.load_summary(self.args.probe_dir)
        self.state = cs.SteeringState(self.args.probe_dir, self.summary, self.device)
        reading_summary = cs.load_summary(self.args.reading_probe_dir)
        self.reading_probes = cs.load_reading_probes(self.args.reading_probe_dir, reading_summary, self.device)

        for spec in self.args.steer:
            attribute, label = spec.split("=", 1)
            self.state.set_target(attribute, label)


SESSION: Session = None
chat_lines = []
chat_buffer = Buffer(read_only=True)
input_buffer = Buffer(multiline=False)


def append_chat(text):
    chat_lines.append(text)
    doc_text = "\n".join(chat_lines) + "\n"
    chat_buffer.set_document(Document(doc_text, cursor_position=len(doc_text)), bypass_readonly=True)


def attribute_status(attribute):
    """Same shape as webui/app.py's attribute_status() -- a different front
    end on the same SteeringState, not a different implementation of it."""
    cfg = SESSION.state.attrs.get(attribute)
    info = SESSION.summary.get(attribute, {})
    default_from, default_to, best_layer = cs.default_window(attribute, SESSION.summary)
    if cfg is None:
        return dict(attribute=attribute, classes=cs.class_names(attribute), active=False, target_label=None,
                    from_idx=default_from, to_idx=default_to, n_scale=cs.DEFAULT_N_SCALE,
                    best_layer=info.get("best_layer", best_layer), best_acc=info.get("best_acc"))
    return dict(attribute=attribute, classes=cfg.labels, active=cfg.target_label is not None,
                target_label=cfg.target_label, from_idx=cfg.from_idx, to_idx=cfg.to_idx, n_scale=cfg.n_scale,
                best_layer=info.get("best_layer", cfg.best_layer), best_acc=info.get("best_acc"))


def get_sidebar_text():
    lines = []
    for attribute in cs.ATTRIBUTE_LABELS:
        a = attribute_status(attribute)
        marker = "*" if a["active"] else " "
        state_str = f"-> {a['target_label']}" if a["active"] else "off"
        lines.append(f"{marker} {attribute}  [{state_str}]")
        lines.append(f"   classes={a['classes']}")
        lines.append(f"   layers=[{a['from_idx']},{a['to_idx']})  scale={a['n_scale']}")
        if a["best_acc"] is not None:
            lines.append(f"   best layer {a['best_layer']} (acc {a['best_acc']:.3f})")
        lines.append("")
    lines.append(f"scores: {'on' if SESSION.show_scores else 'off'}")
    lines.append(f"messages: {len(SESSION.messages)}")
    return "\n".join(lines)


def get_scores_text():
    if SESSION.latest_scores is None:
        return "(nothing to read yet -- send a message first)"
    lines = []
    for attribute, r in SESSION.latest_scores.items():
        breakdown = "  ".join(f"{lbl}={p:.2f}" for lbl, p in zip(r["labels"], r["probs"]))
        lines.append(f"{attribute}: {r['predicted']}")
        lines.append(f"  {breakdown}")
    return "\n".join(lines)


def get_status_bar_text():
    if SESSION.busy:
        return " generating... "
    return " Enter: send/command    Tab: switch focus    Ctrl-C: quit "


def format_tag(active_config):
    if not active_config:
        return "off"
    return " | ".join(f"{a['attribute']}={a['target_label']} layers={a['from_idx']}-{a['to_idx']} scale={a['n_scale']}"
                       for a in active_config)


async def submit_message(text):
    SESSION.busy = True
    get_app().invalidate()
    SESSION.messages.append({"role": "user", "content": text})
    loop = asyncio.get_event_loop()
    try:
        reply = await loop.run_in_executor(
            None, cs.generate, SESSION.model, SESSION.tokenizer, SESSION.messages,
            SESSION.system_prompt, SESSION.device, SESSION.gen_args, SESSION.state)
    except Exception as e:  # keep the UI alive even if generation blows up
        SESSION.messages.pop()
        append_chat(f"[error] {e}")
        SESSION.busy = False
        get_app().invalidate()
        return

    SESSION.messages.append({"role": "assistant", "content": reply})
    active = SESSION.state.config_dict()
    SESSION.steering_log.append(active)
    append_chat(f"bot [{format_tag(active)}] >\n{reply}")

    if SESSION.show_scores and SESSION.reading_probes:
        SESSION.latest_scores = await loop.run_in_executor(
            None, cs.compute_user_scores, SESSION.model, SESSION.tokenizer, SESSION.messages,
            SESSION.reading_probes, SESSION.device)

    SESSION.busy = False
    get_app().invalidate()


def handle_command(line):
    parts = line.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit", "/q"):
        get_app().exit()

    elif cmd == "/help":
        append_chat(__doc__[__doc__.index("KEYS"):].strip())

    elif cmd == "/status":
        append_chat(get_sidebar_text())

    elif cmd == "/probes":
        for attr in cs.ATTRIBUTE_LABELS:
            info = SESSION.summary.get(attr)
            best = f"best layer {info['best_layer']} (acc {info['best_acc']:.3f})" if info else "no summary.json entry"
            append_chat(f"{attr:<18} classes={cs.class_names(attr)}  {best}")

    elif cmd == "/steer":
        bits = arg.split()
        if not bits:
            append_chat("usage: /steer <attribute> <label>   or   /steer <attribute> off   or   /steer off")
        elif bits == ["off"]:
            SESSION.state.clear_target()
            append_chat("steering disabled for all attributes")
        elif len(bits) != 2:
            append_chat("usage: /steer <attribute> <label>   or   /steer <attribute> off   or   /steer off")
        else:
            attribute, label = bits
            try:
                if label == "off":
                    SESSION.state.clear_target(attribute)
                    append_chat(f"steering disabled for '{attribute}'")
                else:
                    SESSION.state.set_target(attribute, label)
                    append_chat(f"steering '{attribute}' toward '{label}'")
            except (ValueError, FileNotFoundError) as e:
                append_chat(str(e))

    elif cmd == "/layers":
        bits = arg.split()
        try:
            if len(bits) == 2 and bits[1] == "best":
                SESSION.state.reset_layers_to_best(bits[0])
                cfg = SESSION.state.attrs[bits[0]]
                append_chat(f"'{bits[0]}' layers reset to best-default [{cfg.from_idx}, {cfg.to_idx})")
            elif len(bits) == 3 and all(b.lstrip("-").isdigit() for b in bits[1:]):
                attribute, from_idx, to_idx = bits[0], int(bits[1]), int(bits[2])
                if from_idx >= to_idx:
                    append_chat("<from> must be less than <to>")
                else:
                    SESSION.state.set_layers(attribute, from_idx, to_idx)
                    append_chat(f"'{attribute}' layers set to [{from_idx}, {to_idx})")
            else:
                append_chat("usage: /layers <attribute> <from> <to>   or   /layers <attribute> best")
        except (ValueError, FileNotFoundError) as e:
            append_chat(str(e))

    elif cmd == "/scale":
        bits = arg.split()
        if len(bits) != 2:
            append_chat("usage: /scale <attribute> <number>")
        else:
            try:
                SESSION.state.set_scale(bits[0], float(bits[1]))
                append_chat(f"'{bits[0]}' n_scale set to {SESSION.state.attrs[bits[0]].n_scale}")
            except (ValueError, FileNotFoundError) as e:
                append_chat(str(e))

    elif cmd == "/scores":
        if arg in ("on", "off"):
            SESSION.show_scores = arg == "on"
            append_chat(f"auto score readout {'enabled' if SESSION.show_scores else 'disabled'}")
        elif arg:
            append_chat("usage: /scores   or   /scores on|off")
        elif not SESSION.reading_probes:
            append_chat("no reading probes loaded -- see the [warn] lines at startup")
        else:
            SESSION.latest_scores = cs.compute_user_scores(
                SESSION.model, SESSION.tokenizer, SESSION.messages, SESSION.reading_probes, SESSION.device)
            append_chat(cs.format_scores(SESSION.latest_scores))

    elif cmd == "/reset":
        SESSION.messages = []
        SESSION.latest_scores = None
        append_chat("conversation cleared")

    elif cmd == "/system":
        if not arg:
            append_chat(f"current system prompt:\n{SESSION.system_prompt}")
        else:
            SESSION.system_prompt = arg
            SESSION.messages = []
            append_chat("system prompt updated, conversation cleared")

    elif cmd == "/save":
        if not arg:
            append_chat("usage: /save <name>")
        elif not SESSION.messages:
            append_chat("nothing to save yet")
        elif os.path.basename(arg) != arg:
            append_chat("filename must not contain path separators")
        else:
            cs.save_conversation(arg, SESSION.system_prompt, SESSION.messages, SESSION.steering_log)
            path = arg if os.path.splitext(arg)[1] else arg + ".json"
            append_chat(f"saved to data/manual.conversations/{path}")

    else:
        append_chat(f"unrecognized command: {cmd}  (/help for commands)")


def on_input_accept(buf):
    text = buf.text.strip()
    if text:
        if text.startswith("/"):
            handle_command(text)
        elif SESSION.busy:
            append_chat("[still generating, please wait]")
        else:
            append_chat(f"you > {text}")
            get_app().create_background_task(submit_message(text))
    return False  # always clear the input line


def build_app():
    input_buffer.accept_handler = on_input_accept

    chat_window = Window(
        content=BufferControl(buffer=chat_buffer, focusable=True),
        wrap_lines=True,
        right_margins=[ScrollbarMargin(display_arrows=True)],
    )
    input_window = Window(content=BufferControl(buffer=input_buffer, focusable=True), height=1)

    root = HSplit([
        Window(FormattedTextControl("Steered Chat -- TalkTuner control-probe steering"),
               height=1, style="reverse"),
        VSplit([
            Frame(title="Chat", body=chat_window),
            HSplit([
                Frame(title="Attributes", body=Window(FormattedTextControl(get_sidebar_text), wrap_lines=True)),
                Frame(title="Scores", body=Window(FormattedTextControl(get_scores_text), wrap_lines=True)),
            ], width=44),
        ]),
        Frame(title="you >", body=input_window),
        Window(FormattedTextControl(get_status_bar_text), height=1, style="reverse"),
    ])

    kb = KeyBindings()
    kb.add("c-c")(lambda event: event.app.exit())
    kb.add("tab")(focus_next)
    kb.add("s-tab")(focus_previous)

    return Application(
        layout=Layout(root, focused_element=input_window),
        key_bindings=kb,
        mouse_support=True,
        full_screen=True,
        refresh_interval=0.5,  # keeps the "generating..." status bar live
    )


def main():
    ap = argparse.ArgumentParser(description="Terminal UI for chatting with Llama-2-13b-chat, steered by control probes.")
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
    args = ap.parse_args()

    for spec in args.steer:
        if "=" not in spec:
            ap.error(f"--steer must be ATTRIBUTE=LABEL, got {spec!r}")

    global SESSION
    SESSION = Session(args)

    print(f"[..] loading {args.model} in fp16 (this takes a few minutes on first run)")
    try:
        SESSION.load_model()
    except ValueError as e:
        ap.error(str(e))
    print("[ok] loaded, launching UI")

    append_chat("[ok] loaded. /help for commands, Ctrl-C to quit")
    build_app().run()


if __name__ == "__main__":
    main()
