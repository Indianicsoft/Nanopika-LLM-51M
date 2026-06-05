// ── LLM/cpu_engine.js ─────────────────────────────────────
// Zero external dependencies — pure math, all synchronous
// No new Array() — only new Float32Array() for allocations
// All functions mirror their WebGPU/WGSL counterparts exactly

/**
 * CPU Matrix Multiplication (M×K) * (K×N) = (M×N)
 * Cache-optimal (i, k, j) loop order mirroring WGSL pattern
 */
function cpuMatMul(A, B, M, N, K) {
  const C = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    const iOffset = i * N;
    const iKOffset = i * K;
    for (let k = 0; k < K; k++) {
      const a_ik = A[iKOffset + k];
      if (a_ik === 0) continue; 
      const kNOffset = k * N;
      for (let j = 0; j < N; j++) {
        C[iOffset + j] += a_ik * B[kNOffset + j];
      }
    }
  }
  return C;
}

/**
 * In-place version for pre-allocated output buffers
 */
function cpuMatMulInPlace(A, B, C, M, N, K) {
  C.fill(0);
  for (let i = 0; i < M; i++) {
    const iOffset = i * N;
    const iKOffset = i * K;
    for (let k = 0; k < K; k++) {
      const a_ik = A[iKOffset + k];
      if (a_ik === 0) continue;
      const kNOffset = k * N;
      for (let j = 0; j < N; j++) {
        C[iOffset + j] += a_ik * B[kNOffset + j];
      }
    }
  }
}

/**
 * Layer Normalization mirroring LAYERNORM_WGSL exactly
 */
function cpuLayerNorm(input, gamma, beta, out, len, eps = 1e-5) {
  // Pass 1 — mean:
  let mean = 0.0;
  for (let i = 0; i < len; i++) mean += input[i];
  mean /= len;

  // Pass 2 — variance:
  let variance = 0.0;
  for (let i = 0; i < len; i++) {
    const d = input[i] - mean;
    variance += d * d;
  }
  variance /= len;

  // Normalize: eps INSIDE sqrt
  const invStd = 1.0 / Math.sqrt(variance + eps);
  for (let i = 0; i < len; i++) {
    out[i] = (input[i] - mean) * invStd * gamma[i] + beta[i];
  }
}

/**
 * Softmax with max-subtraction trick mirroring SOFTMAX_WGSL
 */
function cpuSoftmax(logits, probs, size) {
  let maxVal = -Infinity;
  for (let i = 0; i < size; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }

  let sumExp = 0.0;
  for (let i = 0; i < size; i++) {
    probs[i] = Math.exp(logits[i] - maxVal);
    sumExp += probs[i];
  }

  const invSum = 1.0 / (sumExp + 1e-10); // epsilon guard on sum
  for (let i = 0; i < size; i++) {
    probs[i] *= invSum;
  }
}

/**
 * Attention Scores calculation [seqLen × seqLen]
 */
function cpuAttentionScores(Q, K_mat, seqLen, dHead) {
  // Transpose K: [seqLen×dHead] → [dHead×seqLen]
  const Kt = new Float32Array(dHead * seqLen);
  for (let i = 0; i < dHead; i++) {
    for (let j = 0; j < seqLen; j++) {
      Kt[i * seqLen + j] = K_mat[j * dHead + i];
    }
  }

  const scores = cpuMatMul(Q, Kt, seqLen, seqLen, dHead);
  const scale  = 1.0 / Math.sqrt(dHead);
  for (let i = 0; i < scores.length; i++) {
    scores[i] *= scale;
  }
  return scores;
}

/**
 * Cross-Entropy Loss mirroring backprop logic
 */
function cpuCrossEntropyLoss(probs, targetId) {
  const clamped = Math.max(1e-7, Math.min(0.9999999, probs[targetId]));
  return -Math.log(clamped);
}

/**
 * Xavier Initialization mirroring model initialization logic
 */
function cpuXavierFill(view, fanIn, fanOut, embeddingDim) {
  const limit = Math.sqrt(6.0 / (fanIn + fanOut)) * (1.0 / Math.sqrt(embeddingDim));
  for (let i = 0; i < view.length; i++) {
    view[i] = (Math.random() * 2.0 - 1.0) * limit;
  }
}

/**
 * Adam Optimizer Step mirroring backprop.js
 */
function cpuAdamStep(params, grads, m, v, t, lr, beta1, beta2, eps) {
  const bcM = 1.0 - Math.pow(beta1, t);
  const bcV = 1.0 - Math.pow(beta2, t);
  for (let k = 0; k < params.length; k++) {
    const g = grads[k];
    m[k]    = beta1 * m[k] + (1.0 - beta1) * g;
    v[k]    = beta2 * v[k] + (1.0 - beta2) * g * g;
    const mHat = m[k] / bcM;
    const vHat = v[k] / bcV;
    params[k] -= lr * mHat / (Math.sqrt(vHat) + eps); // epsilon guard
  }
}

export {
  cpuMatMul,
  cpuMatMulInPlace,
  cpuLayerNorm,
  cpuSoftmax,
  cpuAttentionScores,
  cpuCrossEntropyLoss,
  cpuXavierFill,
  cpuAdamStep
};
