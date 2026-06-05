/**
 * Nanopika 51M — Chat Pipeline
 * Full browser-based LLM chat with streaming generation,
 * model.bin + vocab.json import, and adaptive logging.
 */

import { generate as generateLLM, kvCacheReset } from "./LLM/generate.js";
import { getOrderedWeights, modelReady, config } from "./LLM/model.js";
import { loadWeights } from "./LLM/serializer.js";
import {
  saveTokenizerVocab,
  loadTokenizerVocab,
  loadTokenizerVocabFromUrl,
} from "./LLM/tokenizer_serializer.js";
import { initAdaptiveSystem } from "./LLM/adaptive.js";

// ── DOM Refs ──────────────────────────────────────────────
const chatMessages  = document.getElementById("chat-messages");
const userInput     = document.getElementById("user-input");
const sendBtn       = document.getElementById("send-btn");
const clearBtn      = document.getElementById("clear-btn");
const loadingStatus = document.getElementById("loading-status");
const tokenCount    = document.getElementById("token-count");
const engineBadge   = document.getElementById("engine-badge");
const welcomeScreen = document.getElementById("welcome-screen");
const sidebarToggle = document.getElementById("sidebar-toggle");
const appShell      = document.querySelector(".app-shell");

// Model import DOM
const importModelBin  = document.getElementById("import-model-bin");
const importVocabJson = document.getElementById("import-vocab-json");
const loadModelBtn    = document.getElementById("load-model-btn");
const importHint      = document.getElementById("import-hint");
const modelBinLabel   = document.getElementById("model-bin-label");
const vocabJsonLabel  = document.getElementById("vocab-json-label");
const statusDot       = document.getElementById("status-dot");
const statusText      = document.getElementById("status-text");

// Generation controls
const temperatureInput = document.getElementById("temperature");
const topkInput        = document.getElementById("topk");
const maxTokensInput   = document.getElementById("max-tokens");

// Info panel
const infoLayers = document.getElementById("info-layers");
const infoHeads  = document.getElementById("info-heads");
const infoDim    = document.getElementById("info-dim");

// ── State ─────────────────────────────────────────────────
let modelFullyReady = false;
let isGenerating    = false;
let selectedModelBin  = null;
let selectedVocabJson = null;
let adaptive;
let weights;
let messageHistory = [];   // { role, text }[]

// ── Sidebar toggle ────────────────────────────────────────
sidebarToggle?.addEventListener("click", () => {
  appShell.classList.toggle("sidebar-collapsed");
});

// ── Auto-resize textarea ──────────────────────────────────
userInput?.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 180) + "px";

  const len = userInput.value.length;
  if (tokenCount) tokenCount.textContent = len > 0 ? `${len} chars` : "";
});

userInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) handleSend();
  }
});

sendBtn?.addEventListener("click", handleSend);
clearBtn?.addEventListener("click", clearChat);

function updateEngineBadge() {
  const gpuActive = window.gpuReady && window.ENGINE_MODE === "GPU";
  if (engineBadge) {
    engineBadge.textContent = gpuActive ? "GPU" : "CPU";
    engineBadge.className = "topbar-badge" + (gpuActive ? " gpu" : "");
  }
}

engineBadge?.addEventListener("click", () => {
  if (!window.gpuReady) {
    addSystemMessage("WebGPU is not supported/initialized; falling back to CPU.");
    return;
  }
  if (window.ENGINE_MODE === "GPU") {
    window.ENGINE_MODE = "CPU";
    addSystemMessage("Engine switched to CPU Mode (Pure Javascript)");
  } else {
    window.ENGINE_MODE = "GPU";
    addSystemMessage("Engine switched to GPU Mode (WebGPU Accelerated)");
  }
  updateEngineBadge();
});

// ── Quick prompt chips ────────────────────────────────────
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const prompt = chip.dataset.prompt;
    if (prompt && modelFullyReady && !isGenerating) {
      userInput.value = prompt;
      handleSend();
    }
  });
});

// ── Model file selection ──────────────────────────────────
importModelBin?.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  selectedModelBin = file;
  modelBinLabel.textContent = file.name;
  document.querySelector('label[for="import-model-bin"]').classList.add("selected");
  updateLoadButton();
  importHint.textContent = "Click 'Load Selected Files' to apply";
});

importVocabJson?.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  selectedVocabJson = file;
  vocabJsonLabel.textContent = file.name;
  document.querySelector('label[for="import-vocab-json"]').classList.add("selected");
  updateLoadButton();
  importHint.textContent = "Click 'Load Selected Files' to apply";
});

function updateLoadButton() {
  if (loadModelBtn) {
    loadModelBtn.disabled = !(selectedModelBin || selectedVocabJson);
  }
}

loadModelBtn?.addEventListener("click", async () => {
  if (!weights) return;

  loadModelBtn.disabled = true;
  setStatus("loading", "Loading files…");
  importHint.textContent = "Loading…";

  let loaded = 0;
  let errors = [];

  // Load model.bin
  if (selectedModelBin) {
    try {
      await loadWeights(selectedModelBin, weights);
      loaded++;
      console.log("[Chat] ✓ model.bin loaded from file");
    } catch (err) {
      console.error("[Chat] Failed to load model.bin:", err);
      errors.push("model.bin: " + err.message);
    }
  }

  // Load vocab.json
  if (selectedVocabJson) {
    try {
      const { importTokenizerFile } = await import("./LLM/tokenizer_serializer.js");
      const ok = await importTokenizerFile(selectedVocabJson);
      if (!ok) throw new Error("Invalid vocabulary format");
      loaded++;
      console.log("[Chat] ✓ vocab.json loaded from file");
    } catch (err) {
      console.error("[Chat] Failed to load vocab.json:", err);
      errors.push("vocab.json: " + err.message);
    }
  }

  if (errors.length === 0) {
    setStatus("ready", `Loaded ${loaded} file${loaded > 1 ? "s" : ""} ✓`);
    importHint.textContent = `✓ ${loaded} file${loaded > 1 ? "s" : ""} loaded successfully`;
    addSystemMessage(`✓ Model files loaded: ${[selectedModelBin?.name, selectedVocabJson?.name].filter(Boolean).join(", ")}`);
  } else {
    setStatus("error", "Load failed");
    importHint.textContent = "Error: " + errors.join("; ");
  }

  loadModelBtn.disabled = false;
  // Reset KV cache after weight change
  kvCacheReset();
});

// ── Status helper ─────────────────────────────────────────
function setStatus(state, text) {
  if (statusDot) {
    statusDot.className = "status-dot " + state;
  }
  if (statusText) statusText.textContent = text;
  if (loadingStatus) {
    loadingStatus.textContent = text;
    loadingStatus.className = state === "ready" ? "ready" : state === "generating" ? "generating" : state === "error" ? "error" : "";
  }
}

// ── Model initialisation pipeline ────────────────────────
(async () => {
  setStatus("loading", "Loading Nanopika 51M…");

  try {
    await modelReady;
    weights = getOrderedWeights();

    // Fill info panel
    if (config) {
      if (infoLayers) infoLayers.textContent = config.numLayers ?? 8;
      if (infoHeads)  infoHeads.textContent  = config.numHeads ?? 8;
      if (infoDim)    infoDim.textContent     = config.embeddingDim ?? 512;
    }

    // Try loading bundled model.bin
    let weightsLoaded = false;
    try {
      setStatus("loading", "Loading weights…");
      await loadWeights("./LLM/model.bin", weights);
      weightsLoaded = true;
      console.log("[Chat] ✓ Bundled model.bin loaded");
    } catch {
      console.info("[Chat] No bundled model.bin — using initialized weights");
    }

    // Try loading bundled vocab.json
    try {
      setStatus("loading", "Loading vocabulary…");
      const vocabLoaded = await loadTokenizerVocabFromUrl("./LLM/vocab.json");
      if (!vocabLoaded) await loadTokenizerVocab();
      console.log("[Chat] ✓ Tokenizer vocabulary loaded");
    } catch (err) {
      console.warn("[Chat] Tokenizer load warning:", err?.message);
    }

    // Engine badge
    updateEngineBadge();

    // Adaptive system
    adaptive = await initAdaptiveSystem(weights);

    // Enable UI
    modelFullyReady = true;
    userInput.disabled = false;
    sendBtn.disabled = false;
    document.querySelectorAll(".chip").forEach(chip => chip.disabled = false);
    userInput.focus();

    const statusMsg = weightsLoaded
      ? "Nanopika 51M ready (trained weights)"
      : "Nanopika 51M ready (random weights — train or import model.bin)";

    setStatus("ready", weightsLoaded ? "Ready · Trained weights" : "Ready · Random weights");
    console.log("[Chat] ✓", statusMsg);

  } catch (err) {
    setStatus("error", "Initialization failed");
    console.error("[Chat] Fatal init error:", err);
    addSystemMessage("⚠ Model initialization failed: " + err.message);
  }
})();

// ── Send message handler ──────────────────────────────────
async function handleSend() {
  const text = userInput.value.trim();
  if (!text || !modelFullyReady || isGenerating) return;

  // Hide welcome screen
  if (welcomeScreen) welcomeScreen.style.display = "none";

  // Clear input
  userInput.value = "";
  userInput.style.height = "auto";
  if (tokenCount) tokenCount.textContent = "";

  // Add user bubble
  addMessage("user", text);
  messageHistory.push({ role: "user", text });

  // Log to adaptive system
  if (adaptive?.logger) {
    adaptive.logger.logInteraction({ type: "user_message", userText: text });
    adaptive.onMessageSent?.();
  }

  // Lock UI
  setGenerating(true);

  // Show typing indicator
  const typingRow = showTypingIndicator();

  try {
    setStatus("generating", "Generating…");

    // Get generation params from UI sliders
    const temperature = parseFloat(temperatureInput?.value ?? 0.8);
    const topK        = parseInt(topkInput?.value ?? 40);
    const maxTokens   = parseInt(maxTokensInput?.value ?? 200);

    // Create bot bubble ready for streaming
    typingRow.remove();
    const { row, bubble } = createBotBubble();
    let generatedText = "";

    // Stream generation token by token
    await generateLLM(text, {
      temperature,
      topK,
      maxNewTokens: maxTokens,
      onToken: (token) => {
        generatedText += token;
        bubble.textContent = generatedText;
        const cursor = bubble.querySelector(".streaming-cursor");
        if (cursor) bubble.removeChild(cursor);
        const cur = document.createElement("span");
        cur.className = "streaming-cursor";
        bubble.appendChild(cur);
        scrollToBottom();
      }
    });

    // Remove cursor after done
    const cursor = bubble.querySelector(".streaming-cursor");
    if (cursor) cursor.remove();

    if (!generatedText.trim()) {
      bubble.textContent = "(No output generated — try loading a trained model.bin)";
      bubble.classList.add("error");
    }

    messageHistory.push({ role: "bot", text: generatedText });

    // Log response
    if (adaptive?.logger && generatedText) {
      adaptive.logger.logInteraction({
        type: "bot_response",
        userText: text,
        botText: generatedText,
      });
    }

  } catch (err) {
    if (typingRow.parentNode) typingRow.remove();
    console.error("[Chat] Generation error:", err);
    addMessage("bot", "⚠ Generation error: " + err.message, true);
  } finally {
    setStatus("ready", "Ready");
    setGenerating(false);
    userInput.focus();
  }
}

// ── Message rendering helpers ─────────────────────────────
function addMessage(role, text, isError = false) {
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  if (role === "user") {
    avatar.textContent = "U";
  } else {
    avatar.innerHTML = `<svg width="16" height="16" viewBox="0 0 28 28" fill="none">
      <circle cx="14" cy="14" r="13" stroke="url(#alag)" stroke-width="1.5"/>
      <path d="M8 14L14 8L20 14L14 20Z" fill="url(#albg)" opacity="0.8"/>
      <circle cx="14" cy="14" r="3" fill="#06b6d4"/>
      <defs>
        <linearGradient id="alag" x1="0" y1="0" x2="28" y2="28">
          <stop offset="0%" stop-color="#06b6d4"/><stop offset="100%" stop-color="#8b5cf6"/>
        </linearGradient>
        <linearGradient id="albg" x1="0" y1="0" x2="28" y2="28">
          <stop offset="0%" stop-color="#06b6d4" stop-opacity="0.5"/>
          <stop offset="100%" stop-color="#8b5cf6" stop-opacity="0.5"/>
        </linearGradient>
      </defs>
    </svg>`;
  }

  const content = document.createElement("div");
  content.className = "msg-content";

  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = role === "user" ? "You" : "Nanopika 51M";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble" + (isError ? " error" : "");
  bubble.textContent = text;

  content.appendChild(label);
  content.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(content);
  chatMessages.appendChild(row);
  scrollToBottom();
  return row;
}

function createBotBubble() {
  const row = document.createElement("div");
  row.className = "msg-row bot";

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.innerHTML = `<svg width="16" height="16" viewBox="0 0 28 28" fill="none">
    <circle cx="14" cy="14" r="13" stroke="url(#blag2)" stroke-width="1.5"/>
    <path d="M8 14L14 8L20 14L14 20Z" fill="url(#blbg2)" opacity="0.8"/>
    <circle cx="14" cy="14" r="3" fill="#06b6d4"/>
    <defs>
      <linearGradient id="blag2" x1="0" y1="0" x2="28" y2="28">
        <stop offset="0%" stop-color="#06b6d4"/><stop offset="100%" stop-color="#8b5cf6"/>
      </linearGradient>
      <linearGradient id="blbg2" x1="0" y1="0" x2="28" y2="28">
        <stop offset="0%" stop-color="#06b6d4" stop-opacity="0.5"/>
        <stop offset="100%" stop-color="#8b5cf6" stop-opacity="0.5"/>
      </linearGradient>
    </defs>
  </svg>`;

  const content = document.createElement("div");
  content.className = "msg-content";

  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = "Nanopika 51M";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.setAttribute("aria-live", "polite");

  content.appendChild(label);
  content.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(content);
  chatMessages.appendChild(row);
  scrollToBottom();
  return { row, bubble };
}

function addSystemMessage(text) {
  const div = document.createElement("div");
  div.style.cssText = "text-align:center;font-size:0.72rem;color:#475569;padding:4px 0;font-family:'Fira Code',monospace;";
  div.textContent = text;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function showTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg-row bot";
  row.innerHTML = `
    <div class="msg-avatar">
      <svg width="16" height="16" viewBox="0 0 28 28" fill="none">
        <circle cx="14" cy="14" r="13" stroke="url(#tlag)" stroke-width="1.5"/>
        <path d="M8 14L14 8L20 14L14 20Z" fill="url(#tlbg)" opacity="0.8"/>
        <circle cx="14" cy="14" r="3" fill="#06b6d4"/>
        <defs>
          <linearGradient id="tlag" x1="0" y1="0" x2="28" y2="28">
            <stop offset="0%" stop-color="#06b6d4"/><stop offset="100%" stop-color="#8b5cf6"/>
          </linearGradient>
          <linearGradient id="tlbg" x1="0" y1="0" x2="28" y2="28">
            <stop offset="0%" stop-color="#06b6d4" stop-opacity="0.5"/>
            <stop offset="100%" stop-color="#8b5cf6" stop-opacity="0.5"/>
          </linearGradient>
        </defs>
      </svg>
    </div>
    <div class="msg-content">
      <div class="msg-label">Nanopika 51M</div>
      <div class="typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    </div>`;
  chatMessages.appendChild(row);
  scrollToBottom();
  return row;
}

function clearChat() {
  // Remove all message rows and system messages, keep welcome screen hidden
  chatMessages.innerHTML = "";
  messageHistory = [];
  kvCacheReset();
  addSystemMessage("Chat cleared · KV cache reset");
}

function setGenerating(state) {
  isGenerating = state;
  if (sendBtn) sendBtn.disabled = state;
  if (userInput) userInput.disabled = state;
  document.querySelectorAll(".chip").forEach(chip => chip.disabled = state);
}

function scrollToBottom() {
  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
}

// ── Exports for compatibility ─────────────────────────────
export { addMessage as displayMessage, scrollToBottom };
