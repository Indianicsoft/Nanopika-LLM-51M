// Engine mode: "GPU" (default) or "CPU"
// Set on window so all modules can read without import
window.ENGINE_MODE = "GPU";

const ENGINE_GPU = "GPU";
const ENGINE_CPU = "CPU";

import { config, arena, tensorMap, modelReady } from "./model.js";
import {
  accumulatedTrainStep,
  extendArenaWithAccumBuffer,
} from "./grad_accum.js";
import { getLr, optimizer } from "./backprop.js";
import { exportModel } from "./serializer.js";
import { saveTokenizerVocab } from "./tokenizer_serializer.js";
import {
  logToTerminal,
  updateProgressBar,
  setTrainingStatus,
} from "./terminal_ui.js";

// Dual-Engine Imports
import { executeGPUMatMul, gpuLayerNorm, gpuAttentionScores } from "../webgpu_matmul.js";
import { cpuMatMul, cpuLayerNorm, cpuAttentionScores } from "./cpu_engine.js";

/**
 * Recalculates total and warmup steps for the chunking architecture
 */
export function computeTrainingSteps(dataset, opts) {
  const batchesPerEpoch = dataset.totalBatches;
  const totalSteps      = opts.epochs * batchesPerEpoch;
  // Use UI-provided warmupSteps or default to 10%
  const warmupSteps     = opts.warmupSteps !== undefined ? opts.warmupSteps : Math.ceil(totalSteps * 0.10);

  console.log("[StepMath] Chunked step calculation:");
  console.log("  batchesPerEpoch :", batchesPerEpoch);
  console.log("  epochs          :", opts.epochs);
  console.log("  totalSteps      :", totalSteps);
  console.log("  warmupSteps     :", warmupSteps);

  return { totalSteps, warmupSteps, batchesPerEpoch };
}

/**
 * Global Router for Matrix Multiplication
 */
export async function engineMatMul(A, B, M, N, K) {
  if (window.ENGINE_MODE === ENGINE_GPU) {
    return await executeGPUMatMul(A, B, M, N, K);
  }
  return cpuMatMul(A, B, M, N, K); // Synchronous, no await needed
}

/**
 * Global Router for Layer Normalization
 */
export async function engineLayerNorm(input, gamma, beta, out, len) {
  if (window.ENGINE_MODE === ENGINE_GPU) {
    await gpuLayerNorm(input, gamma, beta, out, len);
  } else {
    cpuLayerNorm(input, gamma, beta, out, len);
  }
}

/**
 * Global Router for Attention Scores
 */
export async function engineAttentionScores(Q, K_mat, seqLen, dHead) {
  if (window.ENGINE_MODE === ENGINE_GPU) {
    return await gpuAttentionScores(Q, K_mat, seqLen, dHead);
  }
  return cpuAttentionScores(Q, K_mat, seqLen, dHead);
}

export function yieldFrame() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export async function runTrainingPipeline(dataset, opts = {}) {
  const defaults = {
    epochs: 3,
    stepsPerEpoch: 100,
    microBatchSize: 1,
    fullBatchSize: 1,
  };
  const settings = { ...defaults, ...opts };

  logToTerminal("INIT", { msg: "Waiting for Neural Arena..." });
  await modelReady;

  if (!arena) {
    logToTerminal("ERROR", { msg: "Model initialization failed." });
    return;
  }

  // Performance Warning
  if (navigator.gpu) {
    const adapter = await navigator.gpu.requestAdapter();
    if (
      adapter &&
      (adapter.info.architecture === "swiftshader" ||
        adapter.info.device.toLowerCase().includes("swiftshader"))
    ) {
      logToTerminal(
        "WARNING",
        {
          msg: "SwiftShader detected. Performance will be limited. Consider enabling Hardware Acceleration.",
        },
        "warn",
      );
    }
  }

  extendArenaWithAccumBuffer(arena);
  setTrainingStatus("running");

  // WebGPU Availability Check & Dynamic Fallback
  if (window.ENGINE_MODE === ENGINE_GPU && !window.gpuReady) {
    window.ENGINE_MODE = ENGINE_CPU;
    logToTerminal(
      "⚠ SYSTEM",
      {
        msg: "GPU mode selected but WebGPU is unavailable or failed to initialize. Falling back to CPU Engine.",
      },
      "warn",
    );
  }

  // Step Calculation for Chunking
  const stepMath = computeTrainingSteps(dataset, settings);
  const totalSteps = stepMath.totalSteps;
  const warmupSteps = stepMath.warmupSteps;

  // Pass schedule to optimizer
  optimizer.setWarmup(warmupSteps, totalSteps);

  // CPU Warning
  if (window.ENGINE_MODE === ENGINE_CPU) {
    logToTerminal(
      "⚠ SYSTEM",
      {
        msg: "EXECUTING PURE JS CPU ENGINE. Performance will be severely degraded compared to WebGPU hardware acceleration.",
      },
      "warn",
    );
  }

  // Lock engine toggle
  if (typeof window.lockEngineToggle === "function") {
    window.lockEngineToggle();
  }

  // ── Defensive abort: catch zero-batch dataset ─────────────
  if (!dataset || !dataset.batches || dataset.batches.length === 0) {
    logToTerminal("✖ ERROR", {
      msg: "TRAINING ABORTED — dataset.batches.length === 0. "
         + "Ensure input text has at least "
         + (config.maxSeqLen + 1) + " tokens after tokenization."
    }, "error");
    setTrainingStatus("idle");
    if (typeof window.unlockEngineToggle === "function") {
      window.unlockEngineToggle();
    }
    return; // hard abort — do not enter epoch loop
  }

  // ── Log actual dataset stats for transparency ──────────────
  logToTerminal("DATASET", {
    rawTokens    : dataset.totalTokens.toLocaleString(),
    paddedTokens : (dataset.paddedTokens || dataset.totalTokens).toLocaleString(),
    batches      : dataset.totalBatches.toLocaleString(),
    steps        : totalSteps.toLocaleString(),
    warmup       : warmupSteps.toLocaleString()
  });

  // ── Guard: warn if padding is heavy (>50% of dataset) ──────
  const padRatio = ((dataset.paddedTokens || dataset.totalTokens) - dataset.totalTokens)
                  / (dataset.paddedTokens || dataset.totalTokens || 1);
  if (padRatio > 0.5) {
    logToTerminal("⚠ SYSTEM", {
      msg: "Heavy padding detected ("
         + (padRatio * 100).toFixed(1)
         + "% of tokens are PAD). Add more training data for best results."
    }, "warn");
  }
  logToTerminal("LOADED", {
    params: arena.paramCount.toLocaleString(),
    layers: config.numLayers,
    heads: config.numHeads,
  });

  let globalStep = 0;
  updateProgressBar(0, totalSteps);

  // Initialize total steps in UI
  if (document.getElementById("totalStepsVal"))
    document.getElementById("totalStepsVal").textContent = totalSteps;

  // Sanity check — log mask stats before first step
  const firstBatch = dataset.batches[0];
  const totalMask = Array.from(firstBatch.mask).reduce((s, v) => s + v, 0);
  console.log("[MaskCheck] batch.length:", firstBatch.length,
    "| seqLen:", config.maxSeqLen,
    "| realTokens:", totalMask,
    "| padTokens:", firstBatch.mask.length - totalMask,
    "| padRatio:", ((firstBatch.mask.length - totalMask) / firstBatch.mask.length * 100).toFixed(1) + "%");

  try {
    for (let epoch = 1; epoch <= settings.epochs; epoch++) {
      logToTerminal("EPOCH_START", { epoch, total: settings.epochs });

      for (let batchIdx = 0; batchIdx < dataset.batches.length; batchIdx++) {
        const batch = dataset.batches[batchIdx];

        // Token-level progress callback
        const onTokenProgress = (t, total, currentLoss) => {
          if (t % 50 === 0 || t === total - 1) {
            const pct = Math.round((t / total) * 100);
            logToTerminal(`STEP_PROGRESS`, {
              batch: `${batchIdx + 1}/${dataset.batches.length}`,
              token: `${t}/${total} (${pct}%)`,
              loss: currentLoss.toFixed(4),
            });
          }
        };

        const { loss } = await accumulatedTrainStep(
          batch,
          settings.microBatchSize,
          globalStep,
          arena,
          onTokenProgress,
          settings.peakLR // Pass Peak LR from UI
        );

        const currentLR = getLr(globalStep, settings.peakLR);
        globalStep++;

        // UI Synchronization
        const uiMap = {
          lossVal: loss.toFixed(4),
          "train-loss": loss.toFixed(6),
          stepVal: globalStep,
          epochVal: epoch,
          lrVal: currentLR.toExponential(3),
          totalStepsVal: totalSteps,
        };

        for (const [id, val] of Object.entries(uiMap)) {
          const el = document.getElementById(id);
          if (el) el.textContent = val;
        }

        // Terminal Logging for every step completion
        logToTerminal(`STEP_SUCCESS`, {
          step: `${globalStep}/${totalSteps}`,
          loss: loss.toFixed(6),
          lr: currentLR.toExponential(3),
        });

        updateProgressBar(globalStep / totalSteps);
        await yieldFrame();
      }

      logToTerminal("EPOCH_DONE", {
        epoch,
        totalBatches: dataset.batches.length,
      });
      await yieldFrame();
    }

    logToTerminal("EXPORT", { msg: "Persisting weights to model_weights.bin" });
    exportModel(arena.weights, "model_weights.bin");

    // Persist tokenizer vocabulary alongside weights
    logToTerminal("EXPORT", { msg: "Saving tokenizer vocabulary to IndexedDB..." });
    try {
      await saveTokenizerVocab();
      logToTerminal("SUCCESS", { msg: "Tokenizer vocabulary saved ✓" });
    } catch (tokErr) {
      logToTerminal("WARNING", { msg: "Tokenizer save failed: " + tokErr.message });
    }

    logToTerminal("SUCCESS", { msg: "Training complete. Weights + tokenizer saved." });
  } catch (err) {
    logToTerminal("FATAL", { msg: err.message });
    console.error(err);
  } finally {
    setTrainingStatus("done");
    if (typeof window.unlockEngineToggle === "function") {
      window.unlockEngineToggle();
    }
  }
}
