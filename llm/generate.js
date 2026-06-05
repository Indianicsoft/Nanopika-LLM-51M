import {
  config,
  arena,
  tensorMap,
  modelReady,
  D_MODEL,
  SEQ_LEN,
  VOCAB_SIZE
} from "./model.js";
import { matmul, layerNorm, gelu, softmax } from "./engine.js";
import { encode, decode, EOS_TOKEN_ID } from "./tokenizer.js";
import { loadTokenizerVocab, loadTokenizerVocabFromUrl } from "./tokenizer_serializer.js";

// ── CONSTANTS ─────────────────────────────────────────────
const TEMPERATURE    = 0.8;
const TOP_K          = 40;
const TOP_P          = 0.9;
const MAX_NEW_TOKENS = 200;
// EOS_TOKEN_ID = 4 (<|EOS|>) — imported from tokenizer to always match the vocab.
const END_TOKEN_ID   = EOS_TOKEN_ID;


// ── KV CACHE ──────────────────────────────────────────────
// Per-layer cache: kvCacheK[layer][pos * embeddingDim .. (pos+1)*embeddingDim]
let kvCacheK = null;   // Float32Array[numLayers][maxSeqLen * embeddingDim]
let kvCacheV = null;
let currentPos = 0;
let cacheInitialized = false;

function initKVCache() {
  if (cacheInitialized) return;
  const layers = config.numLayers;
  const size   = config.maxSeqLen * config.embeddingDim;
  kvCacheK = new Array(layers);
  kvCacheV = new Array(layers);
  for (let l = 0; l < layers; l++) {
    kvCacheK[l] = new Float32Array(size);
    kvCacheV[l] = new Float32Array(size);
  }
  cacheInitialized = true;
}

function kvCacheReset() {
  currentPos = 0;
  if (!cacheInitialized) return;
  for (let l = 0; l < config.numLayers; l++) {
    kvCacheK[l].fill(0);
    kvCacheV[l].fill(0);
  }
}

/**
 * Slide the KV cache left by 1 position when we hit maxSeqLen.
 * This preserves recent context and drops the oldest token.
 */
function kvCacheSlide() {
  const dim = config.embeddingDim;
  for (let l = 0; l < config.numLayers; l++) {
    // Shift all entries left by one position (drop position 0)
    kvCacheK[l].copyWithin(0, dim);
    kvCacheV[l].copyWithin(0, dim);
  }
  currentPos = config.maxSeqLen - 1;
}

// ── LM HEAD: A x W^T ──────────────────────────────────────
// lmHead is stored [vocabSize × dim] (same layout as tokenEmbedding).
// We need output = lnFinal @ lmHead^T = [1×dim] × [dim×vocabSize] = [1×vocabSize].
// Rather than allocating a transposed copy, iterate directly:
function lmHeadMatmul(input, lmHead, out, dim, vocabSize) {
  for (let v = 0; v < vocabSize; v++) {
    let dot = 0.0;
    const rowBase = v * dim; // lmHead row v starts at rowBase
    for (let d = 0; d < dim; d++) {
      dot += input[d] * lmHead[rowBase + d];
    }
    out[v] = dot;
  }
}

// ── SINGLE-TOKEN FORWARD PASS ──────────────────────────────
// Processes one token at `position`, updates KV cache, returns logits.
async function forwardStep(tokenId, position) {
  const dim      = config.embeddingDim;
  const numHeads = config.numHeads;
  const headDim  = config.headDim;
  const ffDim    = config.ffDim;

  // 1. Embedding + Positional Encoding
  const x = new Float32Array(dim);
  const embOff = tokenId * dim;
  const posOff = (position % config.maxSeqLen) * dim;
  for (let d = 0; d < dim; d++) {
    x[d] = tensorMap.tokenEmbedding[embOff + d]
         + tensorMap.posEmbedding[posOff + d];
  }

  // 2. Transformer Layers
  for (let l = 0; l < config.numLayers; l++) {
    const layer = tensorMap.layers[l];

    // ── Pre-Attention LayerNorm ──
    const h = new Float32Array(dim);
    await layerNorm(x, layer.ln1_gamma, layer.ln1_beta, h, config.epsilon);

    // ── Multi-Head Self-Attention with KV Cache ──
    const qFull = new Float32Array(dim);
    const kFull = new Float32Array(dim);
    const vFull = new Float32Array(dim);

    await matmul(h, layer.attn_W_q, qFull, 1, dim, dim);
    await matmul(h, layer.attn_W_k, kFull, 1, dim, dim);
    await matmul(h, layer.attn_W_v, vFull, 1, dim, dim);

    // Store K, V into the cache at current position
    const cacheOff = position * dim;
    kvCacheK[l].set(kFull, cacheOff);
    kvCacheV[l].set(vFull, cacheOff);

    // Compute attention per head
    const attnOut = new Float32Array(dim);
    const numPositions = position + 1; // attend to all positions up to and including current

    for (let head = 0; head < numHeads; head++) {
      const hOff = head * headDim;

      // Extract Q for this head
      // Compute attention scores against all cached K positions
      const scores = new Float32Array(numPositions);
      const scale  = 1.0 / Math.sqrt(headDim);

      for (let t = 0; t < numPositions; t++) {
        let dot = 0;
        const kOff = t * dim + hOff;
        for (let d = 0; d < headDim; d++) {
          dot += qFull[hOff + d] * kvCacheK[l][kOff + d];
        }
        scores[t] = dot * scale;
      }

      // Softmax over scores
      let maxScore = -Infinity;
      for (let t = 0; t < numPositions; t++) {
        if (scores[t] > maxScore) maxScore = scores[t];
      }
      let sumExp = 0;
      for (let t = 0; t < numPositions; t++) {
        scores[t] = Math.exp(scores[t] - maxScore);
        sumExp += scores[t];
      }
      const invSum = 1.0 / (sumExp + 1e-10);
      for (let t = 0; t < numPositions; t++) {
        scores[t] *= invSum;
      }

      // Weighted sum of V
      for (let t = 0; t < numPositions; t++) {
        const w = scores[t];
        if (w < 1e-9) continue; // skip negligible weights
        const vOff = t * dim + hOff;
        for (let d = 0; d < headDim; d++) {
          attnOut[hOff + d] += w * kvCacheV[l][vOff + d];
        }
      }
    }

    // Project attention output
    const projected = new Float32Array(dim);
    await matmul(attnOut, layer.attn_W_out, projected, 1, dim, dim);

    // Residual connection
    for (let d = 0; d < dim; d++) x[d] += projected[d];

    // ── Pre-MLP LayerNorm ──
    const h2 = new Float32Array(dim);
    await layerNorm(x, layer.ln2_gamma, layer.ln2_beta, h2, config.epsilon);

    // ── Feed-Forward (MLP) ──
    const mid = new Float32Array(ffDim);
    await matmul(h2, layer.mlp_W_up, mid, 1, ffDim, dim);
    gelu(mid, mid);
    const ffOut = new Float32Array(dim);
    await matmul(mid, layer.mlp_W_down, ffOut, 1, dim, ffDim);

    // Residual connection
    for (let d = 0; d < dim; d++) x[d] += ffOut[d];
  }

  // 3. Final LayerNorm
  const lnFinal = new Float32Array(dim);
  await layerNorm(x, tensorMap.finalNorm_gamma, tensorMap.finalNorm_beta, lnFinal, config.epsilon);

  // 4. LM Head → Logits
  // lmHead is [vocabSize × dim] (weight-tied to tokenEmbedding).
  // Compute logits = lnFinal @ lmHead^T → [1×vocabSize]
  const logits = new Float32Array(config.vocabSize);
  lmHeadMatmul(lnFinal, tensorMap.lmHead, logits, dim, config.vocabSize);

  return logits;
}


// ── SAMPLING ──────────────────────────────────────────────

/**
 * Suppress non-printable control character tokens (byte IDs 0-31 except
 * tab (9), newline (10), and carriage return (13)). Also suppress ID 127 (DEL).
 * These produce invisible output that confuses users.
 */
function suppressNonPrintable(logits) {
  for (let i = 0; i < 32; i++) {
    if (i === 9 || i === 10 || i === 13) continue; // Allow tab, newline, CR
    logits[i] = -Infinity;
  }
  logits[127] = -Infinity; // DEL
}

/**
 * Apply repetition penalty: tokens that appear in recentIds get their
 * logits divided by the penalty factor (> 1.0 discourages repetition).
 */
function applyRepetitionPenalty(logits, recentIds, penalty) {
  if (penalty <= 1.0) return;
  const seen = new Set(recentIds);
  for (const id of seen) {
    if (id >= 0 && id < logits.length) {
      if (logits[id] > 0) {
        logits[id] /= penalty;
      } else {
        logits[id] *= penalty;
      }
    }
  }
}

function applyTemperature(logits, temp) {
  if (temp <= 0 || temp === 1.0) return;
  for (let i = 0; i < logits.length; i++) {
    logits[i] /= temp;
  }
}

function sampleTopKTopP(logits, k, p) {
  const probs = new Float32Array(logits.length);
  softmax(logits, probs);

  // Top-K: find the k-th largest probability threshold
  const indexed = [];
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > 0) { // Skip -Infinity/zero-prob entries
      indexed.push({ p: probs[i], i: i });
    }
  }
  indexed.sort((a, b) => b.p - a.p);
  const topK = indexed.slice(0, Math.min(k, indexed.length));

  if (topK.length === 0) {
    // Fallback: pick the argmax from raw logits
    let bestIdx = 32; // Start past control chars
    let bestVal = -Infinity;
    for (let i = 32; i < logits.length; i++) {
      if (logits[i] > bestVal) { bestVal = logits[i]; bestIdx = i; }
    }
    return bestIdx;
  }

  // Top-P: accumulate until we reach nucleus probability p
  let cumSum = 0;
  let cutoff = topK.length;
  for (let i = 0; i < topK.length; i++) {
    cumSum += topK[i].p;
    if (cumSum >= p) {
      cutoff = i + 1;
      break;
    }
  }
  const nucleus = topK.slice(0, cutoff);

  // Re-normalize
  let sum = 0;
  for (const x of nucleus) sum += x.p;
  const invSum = 1.0 / (sum + 1e-10);

  // Sample from distribution
  const r = Math.random();
  let cum = 0;
  for (const x of nucleus) {
    cum += x.p * invSum;
    if (cum >= r) return x.i;
  }
  return nucleus[nucleus.length - 1].i;
}

// ── TOKENIZER VOCABULARY LOADING ──────────────────────────
let tokenizerLoaded = false;

async function ensureTokenizer() {
  if (tokenizerLoaded) return;
  try {
    let loaded = await loadTokenizerVocabFromUrl("./LLM/vocab.json");
    if (!loaded) loaded = await loadTokenizerVocab();
    if (loaded) {
      console.log("[Generate] \u2713 Tokenizer vocabulary loaded");
    } else {
      console.warn("[Generate] No saved vocab \u2014 using byte-level fallback");
    }
  } catch (err) {
    console.warn("[Generate] Tokenizer load warning:", err?.message);
  }
  tokenizerLoaded = true;
}

// ── MAIN GENERATION FUNCTION ──────────────────────────────
/**
 * Generate a response to `promptText`.
 *
 * @param {string} promptText
 * @param {object|number} opts  Options object (or legacy maxTokens number):
 *   - temperature   {number}   default 0.8
 *   - topK          {number}   default 40
 *   - topP          {number}   default 0.9
 *   - maxNewTokens  {number}   default 200
 *   - onToken       {function} called with each decoded token string (streaming)
 * @returns {Promise<string>} Full generated text
 */
export async function generate(promptText, opts = {}) {
  const isLegacy     = typeof opts === "number";
  const temperature  = isLegacy ? TEMPERATURE      : (opts.temperature  ?? TEMPERATURE);
  const topK         = isLegacy ? TOP_K            : (opts.topK         ?? TOP_K);
  const topP         = isLegacy ? TOP_P            : (opts.topP         ?? TOP_P);
  const maxNewTokens = isLegacy ? opts             : (opts.maxNewTokens ?? MAX_NEW_TOKENS);
  const onToken      = typeof opts.onToken === "function" ? opts.onToken : null;

  await modelReady;
  await ensureTokenizer();

  // Integrity guard
  const embSample = tensorMap.tokenEmbedding[0];
  if (!isFinite(embSample)) {
    throw new Error("Weights contain NaN/Infinity \u2014 reload or import valid model.bin");
  }
  console.log(`[Generate] emb[0]=${embSample.toFixed(5)} | temp=${temperature} topK=${topK} maxNew=${maxNewTokens}`);

  initKVCache();
  kvCacheReset();

  // Tokenize with chat template
  const formattedPrompt = `user: ${promptText}\nassistant:`;
  const promptIds = encode(formattedPrompt);
  console.log(`[Generate] Prompt \u2192 ${promptIds.length} tokens`);
  if (promptIds.length === 0) {
    throw new Error("Prompt tokenized to 0 tokens. Import vocab.json or train first.");
  }

  // ── PHASE 1: Prefill ──────────────────────────────────────
  let logits;
  for (let t = 0; t < promptIds.length; t++) {
    if (currentPos >= config.maxSeqLen) kvCacheSlide();
    logits = await forwardStep(promptIds[t], currentPos);
    currentPos++;
    if (t % 32 === 0 && t > 0) await new Promise(r => setTimeout(r, 0));
  }
  console.log(`[Generate] Prefill done \u2014 ${currentPos} positions in KV cache`);

  // ── PHASE 2: Autoregressive Decoding ─────────────────────
  let fullText = "";
  let tokensGenerated = 0;
  const REPETITION_PENALTY = 1.3;
  const recentWindow = [];
  const RECENT_WINDOW_SIZE = 64;

  while (tokensGenerated < maxNewTokens) {
    suppressNonPrintable(logits);
    applyRepetitionPenalty(logits, recentWindow, REPETITION_PENALTY);
    applyTemperature(logits, temperature);
    const nextId = sampleTopKTopP(logits, topK, topP);

    // Debug first 5
    if (tokensGenerated < 5) {
      const dbg = decode(new Uint32Array([nextId]));
      console.log(`[Generate] tok[${tokensGenerated}] id=${nextId} \u2192 ${JSON.stringify(dbg)}`);
    }

    if (nextId === END_TOKEN_ID) {
      console.log(`[Generate] EOS at token ${tokensGenerated}`);
      break;
    }

    const tokenText = decode(new Uint32Array([nextId]));
    fullText += tokenText;

    // Stream callback
    if (onToken) onToken(tokenText);

    tokensGenerated++;
    recentWindow.push(nextId);
    if (recentWindow.length > RECENT_WINDOW_SIZE) recentWindow.shift();

    if (currentPos >= config.maxSeqLen) kvCacheSlide();
    logits = await forwardStep(nextId, currentPos);
    currentPos++;

    // Yield for streaming UI update
    await new Promise(r => setTimeout(r, 8));
  }

  if (tokensGenerated >= maxNewTokens) {
    console.log(`[Generate] Max tokens (${maxNewTokens}) reached`);
  }
  console.log(`[Generate] Done. ${tokensGenerated} tokens generated.`);
  return fullText;
}

export { kvCacheReset };
window.generateLLM = generate;

