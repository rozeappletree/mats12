const messagesEl = document.getElementById("messages");
const attributesEl = document.getElementById("attributes");
const scoresEl = document.getElementById("scores");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const scoresToggle = document.getElementById("scores-toggle");

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function showBanner(message, isError = true) {
  let banner = document.querySelector(".error-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.className = "error-banner";
    document.getElementById("app").prepend(banner);
  }
  banner.classList.toggle("ok", !isError);
  banner.textContent = message;
  setTimeout(() => banner.remove(), 4000);
}

function showError(message) {
  showBanner(message, true);
}

function addMessage(role, content, tag) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (tag) {
    const tagEl = document.createElement("span");
    tagEl.className = "tag";
    tagEl.textContent = tag;
    div.appendChild(tagEl);
  }
  const textEl = document.createElement("span");
  textEl.textContent = content;
  div.appendChild(textEl);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function tagFromActive(active) {
  if (!active || active.length === 0) return "off";
  return active
    .map(a => `${a.attribute}=${a.target_label} layers=${a.from_idx}-${a.to_idx} scale=${a.n_scale}`)
    .join(" | ");
}

function renderAttributes(status) {
  attributesEl.innerHTML = "";
  if (!status.attributes.length) {
    attributesEl.innerHTML = '<p class="hint">no probes found in the loaded checkpoint directory</p>';
    return;
  }
  for (const attr of status.attributes) {
    const card = document.createElement("div");
    card.className = "attr-card" + (attr.active ? " active" : "");

    const head = document.createElement("div");
    head.className = "attr-head";
    const name = document.createElement("span");
    name.className = "attr-name";
    name.textContent = attr.attribute;
    head.appendChild(name);

    const bestCol = document.createElement("div");
    bestCol.className = "attr-best-col";
    if (attr.best_layer !== undefined && attr.best_layer !== null) {
      const best = document.createElement("span");
      best.className = "attr-best";
      best.textContent = attr.best_acc != null
        ? `best layer ${attr.best_layer} (acc ${attr.best_acc.toFixed(3)})`
        : `best layer ${attr.best_layer}`;
      bestCol.appendChild(best);
    }
    if (attr.top_layers && attr.top_layers.length) {
      const currentCenter = attr.from_idx + 6; // window is [layer-6, layer+7)
      const topLayers = document.createElement("div");
      topLayers.className = "top-layers";
      for (const { layer, acc } of attr.top_layers) {
        const chip = document.createElement("button");
        chip.className = "layer-chip" + (layer === currentCenter ? " selected" : "");
        chip.textContent = `L${layer}`;
        chip.title = `layer ${layer} (acc ${acc.toFixed(3)}) -- center steering window here`;
        chip.onclick = () => setLayerCenter(attr.attribute, layer);
        topLayers.appendChild(chip);
      }
      bestCol.appendChild(topLayers);
    }
    head.appendChild(bestCol);
    card.appendChild(head);

    const classes = document.createElement("div");
    classes.className = "attr-classes";
    for (const cls of attr.classes) {
      const btn = document.createElement("button");
      btn.className = "class-btn" + (attr.active && attr.target_label === cls ? " selected" : "");
      btn.textContent = cls;
      btn.onclick = () => setSteer(attr.attribute, cls);
      classes.appendChild(btn);
    }
    const offBtn = document.createElement("button");
    offBtn.className = "class-btn" + (!attr.active ? " selected" : "");
    offBtn.textContent = "off";
    offBtn.onclick = () => setSteer(attr.attribute, null);
    classes.appendChild(offBtn);
    card.appendChild(classes);

    const config = document.createElement("div");
    config.className = "attr-config";
    config.innerHTML = `
      layers <input type="number" class="from-input" value="${attr.from_idx}">
      - <input type="number" class="to-input" value="${attr.to_idx}">
      <button class="best-link">best</button>
      &middot; scale <input type="number" step="0.5" class="scale-input" value="${attr.n_scale}">
    `;
    const fromInput = config.querySelector(".from-input");
    const toInput = config.querySelector(".to-input");
    const scaleInput = config.querySelector(".scale-input");
    const bestLink = config.querySelector(".best-link");

    const applyLayers = () => setLayers(attr.attribute, parseInt(fromInput.value, 10), parseInt(toInput.value, 10));
    fromInput.onchange = applyLayers;
    toInput.onchange = applyLayers;
    bestLink.onclick = () => setLayersBest(attr.attribute);
    scaleInput.onchange = () => setScale(attr.attribute, parseFloat(scaleInput.value));

    card.appendChild(config);
    attributesEl.appendChild(card);
  }
}

function renderScores(scores) {
  if (!scores) {
    scoresEl.innerHTML = '<p class="hint">send a message to see scores</p>';
    return;
  }
  scoresEl.innerHTML = "";
  for (const [attribute, r] of Object.entries(scores)) {
    const row = document.createElement("div");
    row.className = "scores-row";
    const maxProb = Math.max(...r.probs);
    const barsHtml = r.labels
      .map((lbl, i) => `
        <div>${lbl} ${(r.probs[i] * 100).toFixed(0)}%
          <div class="scores-bar-bg"><div class="scores-bar-fill" style="width:${r.probs[i] * 100}%"></div></div>
        </div>`)
      .join("");
    row.innerHTML = `<div>${attribute}: <span class="predicted">${r.predicted}</span></div>${barsHtml}`;
    scoresEl.appendChild(row);
  }
}

let currentStatus = null;

async function refreshStatus() {
  currentStatus = await api("/api/status");
  renderAttributes(currentStatus);
  scoresToggle.checked = currentStatus.show_scores;
  return currentStatus;
}

async function setSteer(attribute, label) {
  try {
    await api("/api/steer", { method: "POST", body: JSON.stringify({ attribute, label }) });
    await refreshStatus();
  } catch (e) { showError(e.message); }
}

async function setLayers(attribute, from_idx, to_idx) {
  if (Number.isNaN(from_idx) || Number.isNaN(to_idx)) return;
  try {
    await api("/api/layers", { method: "POST", body: JSON.stringify({ attribute, from_idx, to_idx }) });
    await refreshStatus();
  } catch (e) { showError(e.message); }
}

async function setLayerCenter(attribute, layer) {
  try {
    await api("/api/layers", { method: "POST", body: JSON.stringify({ attribute, layer }) });
    await refreshStatus();
  } catch (e) { showError(e.message); }
}

async function setLayersBest(attribute) {
  try {
    await api("/api/layers", { method: "POST", body: JSON.stringify({ attribute, best: true }) });
    await refreshStatus();
  } catch (e) { showError(e.message); }
}

async function setScale(attribute, n_scale) {
  if (Number.isNaN(n_scale)) return;
  try {
    await api("/api/scale", { method: "POST", body: JSON.stringify({ attribute, n_scale }) });
    await refreshStatus();
  } catch (e) { showError(e.message); }
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  addMessage("user", text);
  chatInput.value = "";
  chatInput.disabled = true;
  sendBtn.disabled = true;
  sendBtn.textContent = "...";
  try {
    const data = await api("/api/chat", { method: "POST", body: JSON.stringify({ message: text }) });
    addMessage("bot", data.reply, tagFromActive(data.active));
    renderScores(data.scores);
  } catch (e) {
    showError(e.message);
  } finally {
    chatInput.disabled = false;
    sendBtn.disabled = false;
    sendBtn.textContent = "Send";
    chatInput.focus();
  }
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

scoresToggle.addEventListener("change", async () => {
  try {
    await api("/api/scores/toggle", { method: "POST", body: JSON.stringify({ on: scoresToggle.checked }) });
  } catch (e) { showError(e.message); }
});

document.getElementById("reset-btn").addEventListener("click", async () => {
  try {
    await api("/api/reset", { method: "POST", body: "{}" });
    messagesEl.innerHTML = "";
    renderScores(null);
  } catch (e) { showError(e.message); }
});

// -- system prompt modal --
const systemModal = document.getElementById("system-modal");
const systemInput = document.getElementById("system-input");
let defaultSystemPrompt = "";
document.getElementById("system-btn").addEventListener("click", async () => {
  const { system_prompt, default_system_prompt } = await api("/api/system");
  defaultSystemPrompt = default_system_prompt;
  systemInput.value = system_prompt;
  systemModal.classList.remove("hidden");
});
// Only refills the textarea -- "Save & clear history" is still what applies it.
document.getElementById("system-reset").addEventListener("click", () => {
  systemInput.value = defaultSystemPrompt;
  systemInput.focus();
});
document.getElementById("system-cancel").addEventListener("click", () => systemModal.classList.add("hidden"));
document.getElementById("system-save").addEventListener("click", async () => {
  try {
    await api("/api/system", { method: "POST", body: JSON.stringify({ prompt: systemInput.value }) });
    systemModal.classList.add("hidden");
    messagesEl.innerHTML = "";
    renderScores(null);
  } catch (e) { showError(e.message); }
});

// -- save modal --
const saveModal = document.getElementById("save-modal");
const saveInput = document.getElementById("save-input");
document.getElementById("save-btn").addEventListener("click", () => {
  saveInput.value = "";
  saveModal.classList.remove("hidden");
  saveInput.focus();
});
document.getElementById("save-cancel").addEventListener("click", () => saveModal.classList.add("hidden"));
document.getElementById("save-confirm").addEventListener("click", async () => {
  const name = saveInput.value.trim();
  if (!name) return;
  try {
    const data = await api("/api/save", { method: "POST", body: JSON.stringify({ name }) });
    saveModal.classList.add("hidden");
    showBanner(`saved to ${data.saved_to}`, false);
  } catch (e) { showError(e.message); }
});

async function loadHistory() {
  const { messages } = await api("/api/messages");
  messagesEl.innerHTML = "";
  for (const m of messages) addMessage(m.role === "user" ? "user" : "bot", m.content);
}

(async function init() {
  await refreshStatus();
  await loadHistory();
})();
