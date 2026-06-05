import { gpuReady, executeGPUMatMul, cpuMatMul, gpuLayerNorm } from "../webgpu_matmul.js";
import { cpuLayerNorm, cpuSoftmax } from "./cpu_engine.js";

// Centralized tape for backprop
export const tape = [];

// ... WGSL strings ...

// ── MODULE 1: matmul(A, B, C, M, N, K, tapeEntry) ──────────
export async function matmul(A, B, C, M, N, K, tapeEntry = null) {
  let result;
  if (window.ENGINE_MODE === "GPU" && window.gpuReady) {
    result = await executeGPUMatMul(A, B, M, N, K);
  } else {
    result = cpuMatMul(A, B, M, N, K);
  }
  C.set(result);
  
  if (tapeEntry) {
    tape.push({
      op: "linear",
      M, N, K,
      X: new Float32Array(A), // Snapshot input
      W: B,
      dOut: tapeEntry.dOut,
      dW: tapeEntry.dW,
      dX: tapeEntry.dX
    });
  }
}

// ── MODULE 2: layerNorm(x, gamma, beta, out, eps, tapeEntry) 
export async function layerNorm(x, gamma, beta, out, eps = 1e-5, tapeEntry = null) {
  const n = x.length;

  // Always compute mean/variance for tape recording accuracy
  let mean = 0.0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0.0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - mean;
    variance += d * d;
  }
  variance /= n;

  if (window.ENGINE_MODE === "GPU" && window.gpuReady) {
    await gpuLayerNorm(x, gamma, beta, out, n, eps);
  } else {
    cpuLayerNorm(x, gamma, beta, out, n, eps);
  }

  if (tapeEntry) {
    tape.push({
      op: "layernorm",
      x: new Float32Array(x),
      gamma,
      beta,
      mean,
      variance,
      eps,
      dOut: tapeEntry.dOut,
      dX: tapeEntry.dX,
      dGamma: tapeEntry.dGamma,
      dBeta: tapeEntry.dBeta
    });
  }
}

export function softmax(x, out) {
  cpuSoftmax(x, out, x.length);
}

export const softmaxInPlace = softmax;

// ── MODULE 4: gelu(x, out, tapeEntry) ─────────────────────
export function gelu(x, out, tapeEntry = null) {
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    out[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
  }
  
  if (tapeEntry) {
    tape.push({
      op: "gelu",
      x: new Float32Array(x),
      dOut: tapeEntry.dOut,
      dIn: tapeEntry.dX
    });
  }
}
