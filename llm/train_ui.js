import { encode, tokenizer } from "./tokenizer.js";
import { runTrainingPipeline } from "./train_loop.js";
import { initTerminalUI, logToTerminal, setTrainingStatus } from "./terminal_ui.js";
import { getOrderedWeights, modelReady } from "./model.js";
import { exportWeights, loadWeights } from "./serializer.js";
import { exportTokenizerFile } from "./tokenizer_serializer.js";
import { getChunkedDataset } from "./data_loader.js";
import { ModelConfig } from "./model_config.js";


console.log("[TrainUI] Module loaded.");

function initTrainUI() {
  const trainBtn = document.getElementById("train-btn");
  const textarea = document.getElementById("train-textarea");
  
  if (!trainBtn) return;

  initTerminalUI();

  trainBtn.onclick = async () => {
    const rawText = textarea.value.trim();
    if (!rawText) return;

    const epochs = parseInt(document.getElementById("train-epochs").value) || 5;
    const peakLR = parseFloat(document.getElementById("peakLR").value) || 0.0005;
    const warmupSteps = parseInt(document.getElementById("warmupSteps").value) || 50;
    
    // Pass the tokenizer singleton so data_loader.js can call tokenizer.encode(rawText)
    const dataset = await getChunkedDataset(rawText, tokenizer, ModelConfig);


    const options = {
      epochs: epochs,
      peakLR: peakLR,
      warmupSteps: warmupSteps,
      microBatchSize: 1,
      fullBatchSize: 1
    };

    try {
      await runTrainingPipeline(dataset, options);
    } catch (err) {
      console.error(err);
      logToTerminal("FATAL", { msg: err.message });
      setTrainingStatus("idle");
    }
  };

  // Add functionality to Export/Import buttons
  const exportBtn = document.getElementById("export-btn");
  if (exportBtn) {
    exportBtn.onclick = async () => {
      logToTerminal("EXPORT", { msg: "Generating weight binary and tokenizer vocab..." });
      await modelReady;
      const weights = getOrderedWeights();
      exportWeights(weights, "model.bin");
      exportTokenizerFile("vocab.json");
    };
  }

  const importInput = document.getElementById("import-input");
  if (importInput) {
    importInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      logToTerminal("IMPORT", { msg: `Loading weights from ${file.name}...` });
      try {
        await modelReady;
        const weights = getOrderedWeights();
        await loadWeights(file, weights);
        logToTerminal("SUCCESS", { msg: "Weights loaded successfully." });
      } catch (err) {
        console.error(err);
        logToTerminal("ERROR", { msg: "Failed to load weights: " + err.message });
      }
    };
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initTrainUI);
} else {
  initTrainUI();
}
