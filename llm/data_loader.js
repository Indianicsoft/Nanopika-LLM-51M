// ── MODULE: data_loader.js ─────────────────────────────────
// Implements non-overlapping chunking with dynamic padding

const PAD_ID = 0;

function padTokenArray(tokens, targetLength) {
  if (tokens.length >= targetLength) return tokens;

  const padded = new Uint32Array(targetLength); // fills with 0 = PAD_ID
  padded.set(tokens);
  // Remaining positions already 0 from Uint32Array zero-init
  return padded;
}

function buildChunkedBatches(tokens, config) {
  const contextWindow = config.maxSeqLen;
  const batchSize     = config.batchSize || 1;
  const tokensNeeded    = tokens.length + 1; // +1 for shifted target tail

  // Each chunk requires contextWindow tokens.
  // Total chunks = ceil(tokensNeeded / contextWindow)
  // Total token slots = totalChunks × contextWindow + 1
  const totalChunks     = Math.ceil(tokensNeeded / contextWindow);
  const requiredTokens  = totalChunks * contextWindow + 1;

  // Pad token array to requiredTokens length with PAD_ID (0)
  const paddedTokens    = padTokenArray(tokens, requiredTokens);

  console.log("[DataLoader] Padding: "
    + tokens.length + " → " + paddedTokens.length + " tokens ("
    + (paddedTokens.length - tokens.length) + " pad tokens appended)");

  // PADDING INVARIANTS:
  // 1. inputs and targets are ALWAYS [actualSize × contextWindow] — no jagged rows
  // 2. mask[i] === 0 for every position where targets[i] === PAD_ID
  // 3. At least one batch will always exist after padding (totalChunks >= 1)
  // 4. Downstream loss must multiply by mask

  const totalBatches = Math.ceil(totalChunks / batchSize);
  const batches      = [];

  for (let b = 0; b < totalBatches; b++) {
    const bStart     = b * batchSize;
    const bEnd       = Math.min(bStart + batchSize, totalChunks);
    const actualSize = bEnd - bStart;

    const inputs  = new Uint32Array(actualSize * contextWindow);
    const targets = new Uint32Array(actualSize * contextWindow);
    const mask    = new Uint8Array (actualSize * contextWindow);

    for (let s = 0; s < actualSize; s++) {
      const tokenStart = (bStart + s) * contextWindow;

      for (let t = 0; t < contextWindow; t++) {
        const inputTok  = paddedTokens[tokenStart + t];
        const targetTok = paddedTokens[tokenStart + t + 1];

        inputs [s * contextWindow + t] = inputTok;
        targets[s * contextWindow + t] = targetTok;

        // mask = 1 if target is a real token, 0 if pad
        mask[s * contextWindow + t] =
          (targetTok !== PAD_ID && (tokenStart + t) < tokens.length) ? 1 : 0;
      }
    }

    batches.push({ inputs, targets, mask, length: actualSize });
  }

  return { batches, paddedTokens, totalChunks };
}

export async function getChunkedDataset(rawText, tokenizer, config) {
  const tokens = tokenizer.encode(rawText);
  const { batches, paddedTokens, totalChunks } = buildChunkedBatches(tokens, config);

  console.log("[DataLoader] Chunking stats:");
  console.log("  Raw tokens    :", tokens.length);
  console.log("  Padded tokens :", paddedTokens?.length ?? "n/a");
  console.log("  Context window:", config.maxSeqLen);
  console.log("  Total chunks  :", totalChunks);
  console.log("  Batch size    :", config.batchSize);
  console.log("  Total batches :", batches.length);

  return {
    batches,
    totalTokens  : tokens.length,
    paddedTokens : paddedTokens.length,
    totalChunks,
    totalBatches : batches.length
  };
}

export { PAD_ID, padTokenArray, buildChunkedBatches };
