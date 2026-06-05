// ── MODULE: terminal_ui.js ─────────────────────────────────

let logEl, progressBar, progressPct, badge, executeBtn;

export function initTerminalUI() {
  // Use existing #terminal if it exists, otherwise fall back to container or body
  const container = document.getElementById("llm-terminal-container");
  const existingTerminal = document.getElementById("terminal");

  if (existingTerminal) {
    logEl = existingTerminal;
    // Apply some base styles to ensure it looks good as a log
    logEl.style.whiteSpace = "pre-wrap";
    logEl.style.wordBreak = "break-all";
  }

  // Still initialize components if they are in the dashboard but not the terminal
  progressBar = document.getElementById("train-progress");
  progressPct = document.getElementById("terminal-progress-pct"); // Might be null
  badge = document.getElementById("train-status");
  executeBtn = document.getElementById("train-btn");

  // Inject Dual-Engine Toggle ABOVE execute button
  const dashboard = document.querySelector(".dashboard");
  if (dashboard && !document.getElementById("engine-toggle-wrap")) {
    const toggleBlock = document.createElement("div");
    toggleBlock.id = "engine-toggle-wrap";
    toggleBlock.innerHTML = `
      <span class="engine-label" id="label-cpu">CPU</span>
      <label class="toggle-switch">
        <input type="checkbox" id="engine-toggle" checked>
        <span class="toggle-track">
          <span class="toggle-thumb"></span>
        </span>
      </label>
      <span class="engine-label" id="label-gpu">GPU</span>
      <span id="engine-badge">WebGPU ACTIVE</span>
    `;
    
    // Find a good place to insert it
    const controls = document.querySelector(".controls") || document.getElementById("train-status");
    if (controls) controls.parentNode.insertBefore(toggleBlock, controls);
    else dashboard.appendChild(toggleBlock);

    // Add Styles
    const style = document.createElement("style");
    style.textContent = `
      #engine-toggle-wrap {
        display: flex; align-items: center; gap: 12px;
        margin: 14px 0 20px;
        padding: 10px 16px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(120,80,255,0.2);
        border-radius: 10px;
        backdrop-filter: blur(5px);
      }
      .engine-label {
        font-size: 12px; font-weight: 700; letter-spacing: 1.5px;
        color: rgba(180,160,255,0.35);
        transition: color 0.3s;
        font-family: 'Orbitron', sans-serif;
      }
      .toggle-switch { position: relative; display: inline-block; width: 52px; height: 26px; cursor: pointer; }
      .toggle-switch input { display: none; }
      .toggle-track {
        display: block; width: 52px; height: 26px;
        background: linear-gradient(90deg, #7b5fff, #d44dff);
        border-radius: 13px;
        box-shadow: 0 0 12px rgba(120,80,255,0.4);
        transition: background 0.3s, box-shadow 0.3s;
        position: relative;
      }
      #engine-toggle:not(:checked) ~ .toggle-track { background: rgba(80,80,120,0.4); box-shadow: none; }
      .toggle-thumb {
        position: absolute; top: 3px; left: 3px;
        width: 20px; height: 20px; border-radius: 50%;
        background: #fff; box-shadow: 0 0 6px rgba(0,0,0,0.4);
        transition: transform 0.25s cubic-bezier(0.4,0,0.2,1);
      }
      #engine-toggle:checked ~ .toggle-track .toggle-thumb { transform: translateX(26px); }
      #engine-badge {
        margin-left: auto; font-size: 10px;
        padding: 3px 10px; border-radius: 20px;
        background: rgba(120,80,255,0.15);
        border: 1px solid rgba(120,80,255,0.4);
        color: #a78bff; transition: all 0.3s;
        font-family: 'Fira Code', monospace;
      }
      #engine-badge.cpu-mode {
        background: rgba(255,200,50,0.1);
        border-color: rgba(255,200,50,0.4);
        color: #ffd060; box-shadow: 0 0 8px rgba(255,200,50,0.2);
      }
      #engine-toggle-wrap.locked { opacity: 0.45; pointer-events: none; }
      
      /* Warn Log Styles */
      .log-row.log-warn { background: rgba(255,200,50,0.05); }
      .log-row.log-warn .log-label { color: #ffd060 !important; }
      .log-row.log-warn .log-data { color: #ffe599 !important; }
    `;
    document.head.appendChild(style);

    // Logic
    const toggleEl   = document.getElementById("engine-toggle");
    const badgeEl    = document.getElementById("engine-badge");
    const labelCPU   = document.getElementById("label-cpu");
    const labelGPU   = document.getElementById("label-gpu");
    const toggleWrap = document.getElementById("engine-toggle-wrap");

    function applyEngineMode(isGPU) {
      // Check for WebGPU availability if GPU mode is requested
      const actualGPU = isGPU && window.gpuReady !== false;
      
      window.ENGINE_MODE   = actualGPU ? "GPU" : "CPU";
      badgeEl.textContent  = actualGPU ? "WebGPU ACTIVE" : "CPU MODE";
      badgeEl.className    = actualGPU ? "" : "cpu-mode";
      labelGPU.style.color = actualGPU ? "#a78bff"  : "rgba(180,160,255,0.35)";
      labelCPU.style.color = actualGPU ? "rgba(180,160,255,0.35)" : "#ffd060";
      
      // Update toggle state if fallback happened
      if (isGPU && !actualGPU) {
        toggleEl.checked = false;
        console.warn("[Engine] WebGPU not ready or unavailable. Falling back to CPU.");
      }
      
      console.log(`[Engine] Mode set to ${window.ENGINE_MODE}`);
    }

    // Defer initialization slightly to allow WebGPU detection to run
    setTimeout(() => {
      // Access gpuReady from window if we don't want to deal with circular imports
      // webgpu_matmul.js sets gpuReady = true/false
      applyEngineMode(window.gpuReady === true);
    }, 100);

    toggleEl.addEventListener("change", () => applyEngineMode(toggleEl.checked));

    // Expose lock/unlock for train_loop.js
    window.lockEngineToggle   = () => toggleWrap.classList.add("locked");
    window.unlockEngineToggle = () => toggleWrap.classList.remove("locked");
  }

  console.log(
    "[TerminalUI] Initialized targeting:",
    logEl ? "Existing #terminal" : "None",
  );
}

export function logToTerminal(label, data, type = "info") {
  if (!logEl) {
    console.log(`[Terminal Fallback] ${label}:`, data);
    return;
  }

  const time = new Date().toLocaleTimeString("en-IN", { hour12: false });

  let dataStr = "";
  if (data) {
    dataStr = Object.entries(data)
      .map(([k, v]) => {
        const val = typeof v === "number" ? v.toFixed(6) : v;
        return `<span style="color: inherit; opacity: 0.7;">${k}:</span> <span style="font-weight: 600;">${val}</span>`;
      })
      .join("  ·  ");
  }

  const row = document.createElement("div");
  row.className = "log-row" + (type === "warn" ? " log-warn" : "") + (type === "error" ? " log-error-row" : "");
  row.style.padding = "6px 12px";
  row.style.borderBottom = "1px solid rgba(255,255,255,0.03)";
  row.style.display = "flex";
  row.style.gap = "15px";
  row.style.fontSize = "12px";

  row.innerHTML = `
    <span style="color: rgba(148, 163, 184, 0.5); min-width: 80px; font-family: monospace;">[${time}]</span>
    <span class="log-label" style="color: #8b5cf6; font-weight: 700; min-width: 130px; text-transform: uppercase;">${label}</span>
    <span class="log-data" style="flex: 1; color: #f1f5f9;">${dataStr}</span>
  `;

  logEl.appendChild(row);
  logEl.scrollTop = logEl.scrollHeight;
}

export function updateProgressBar(fraction, newTotal) {
  if (newTotal !== undefined && progressBar) {
    progressBar.dataset.total = newTotal; // Store for reference
  }
  
  if (progressBar) {
    progressBar.value = fraction * 100;
  }
  if (progressPct) {
    progressPct.textContent = Math.round(fraction * 100) + "%";
  }
}

export function setTrainingStatus(status) {
  if (!badge) return;
  switch (status) {
    case "idle":
      badge.textContent = "System Standby";
      if (executeBtn) executeBtn.disabled = false;
      break;
    case "running":
      badge.textContent = "Nanopika 51M Active";
      if (executeBtn) executeBtn.disabled = true;
      break;
    case "done":
      badge.textContent = "Processing Complete";
      if (executeBtn) executeBtn.disabled = false;
      break;
    case "error":
      badge.textContent = "Kernel Fault Detected";
      if (executeBtn) executeBtn.disabled = false;
      break;
  }
}
