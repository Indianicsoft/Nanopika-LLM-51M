// ── IMPORTS ───────────────────────────────────────────────
// Use the MAIN model arena (from model.js), not the standalone memory.js arena.
// memory.js arena is a separate allocation pool for a different subsystem.
import { arena, config, D_MODEL, D_FF, SEQ_LEN,
         activations, dActivations, VOCAB_SIZE }        from "./model.js"

import { tape, matmul }                                  from "./engine.js"
import { encode, decode, vocab, EOS_TOKEN_ID }           from "./tokenizer.js"
import { forwardPass, softmaxInPlace,
         logits, probs, lossAcc, lossOut }               from "./train.js"
import { backward, zeroGrads, optimizer, getLr }         from "./backprop.js"
import { kvCacheReset }                                  from "./generate.js"
import { DataLogger }                                    from "./adaptive.js"
export { DataLogger };

// Centralized Model State (for freezing)
import {
  dW_q, dW_k, dW_v, dW_ff1, dW_ff2,
  dW_out, dEmbedding, embeddingTable, frozenGrads
} from "./model.js"


// ── SPECIAL TOKEN REGISTRY ───────────────────────────────────
// These IDs match the Tokenizer singleton constructor order.
const TOKEN_USER = 2;  // <|user|>  (index 2 in specialTokens)
const TOKEN_ASST = 3;  // <|bot|>   (index 3 in specialTokens)
const TOKEN_END  = 4;  // <|EOS|>   (index 4 in specialTokens = EOS_TOKEN_ID)

// ── MODULE 1: formatConversationData() ───────────────────
export function formatConversationData(chatLogs) {
    // 1. First pass: compute total length
    let totalLen = 0;
    const msgTokens = chatLogs.map(msg => {
        const tokens = encode(msg.content);
        totalLen += 1 + tokens.length + 1; // [ROLE, ...TOKENS, END]
        return tokens;
    });

    // 2. Allocate buffers
    const inputIds = new Uint32Array(totalLen);
    const lossMask = new Uint8Array(totalLen);

    // 3. Second pass: fill
    let offset = 0;
    for (let i = 0; i < chatLogs.length; i++) {
        const msg = chatLogs[i];
        const tokens = msgTokens[i];
        const isAssistant = msg.role === "assistant";

        // Write boundary token
        inputIds[offset] = isAssistant ? TOKEN_ASST : TOKEN_USER;
        lossMask[offset] = 0; 
        offset++;

        // Write encoded tokens
        for (let j = 0; j < tokens.length; j++) {
            inputIds[offset] = tokens[j];
            lossMask[offset] = isAssistant ? 1 : 0;
            offset++;
        }

        // Write TOKEN_END
        inputIds[offset] = TOKEN_END;
        lossMask[offset] = isAssistant ? 1 : 0;
        offset++;
    }

    // 4. Build targetIds (shifted left by 1)
    const targetIds = new Uint32Array(totalLen);
    for (let i = 0; i < totalLen - 1; i++) {
        targetIds[i] = inputIds[i + 1];
    }
    targetIds[totalLen - 1] = TOKEN_END;

    return { inputIds, targetIds, lossMask };
}

// ── MODULE 2A: frozenGrads + freezeBaseModel() ────────────

export function freezeBaseModel() {
    frozenGrads.add(dW_q);
    frozenGrads.add(dW_k);
    frozenGrads.add(dW_v);
    frozenGrads.add(dW_ff1);
    frozenGrads.add(dW_ff2);
    frozenGrads.add(dEmbedding);
    // dW_out NOT frozen
}

// ── MODULE 2B: unfreezeAll() ─────────────────────────────
export function unfreezeAll() {
    frozenGrads.clear();
}

// ── MODULE 2C: attachClassificationHead() ─────────────────
export function attachClassificationHead(numClasses) {
    const dModel = config ? config.embeddingDim : 512;

    // 1. Allocate head buffers (plain Float32Arrays — the model arena is not
    //    a MemoryArena class and does not have an .allocate() method)
    const W_head  = new Float32Array(dModel * numClasses);
    const b_head  = new Float32Array(numClasses);
    const dW_head = new Float32Array(dModel * numClasses);
    const db_head = new Float32Array(numClasses);

    // 2. Xavier init
    const scale = Math.sqrt(2.0 / (dModel + numClasses));
    for (let i = 0; i < W_head.length; i++) {
        W_head[i] = (Math.random() * 2 - 1) * scale;
    }
    b_head.fill(0);

    // 3. Freeze base model weights
    freezeBaseModel();

    // 4. Register with optimizer (optimizer stores flat arrays internally)
    // NOTE: The AdamOptimizer in backprop.js uses arena-level flat buffers for
    // its main params/grads, so custom heads are tracked separately here.
    // A proper integration would extend the arena; for now we store them.
    return { W_head, b_head, dW_head, db_head };
}


// ── MODULE 2D: classificationForward() ───────────────────
export function classificationForward(contextVec, W_head, b_head, numClasses, outLogits) {
    matmul(contextVec, W_head, outLogits, 1, numClasses, D_MODEL);
    for (let i = 0; i < numClasses; i++) {
        outLogits[i] += b_head[i];
    }
}

// Pre-allocated dummy gradient buffer used as placeholder
const dummyGrads = new Float32Array(VOCAB_SIZE);


// ── MODULE 3: rlTrainingStep() ────────────────────────────
export function rlTrainingStep(inputSequence, generatedSequence, rewardScore, globalStep) {
    // STEP 1 — Concatenate prompt + response
    const fullLen = inputSequence.length + generatedSequence.length;
    const fullSeq = new Uint32Array(fullLen);
    fullSeq.set(inputSequence, 0);
    fullSeq.set(generatedSequence, inputSequence.length);

    // STEP 2 — Build RL loss mask (only generated tokens)
    const rlMask = new Uint8Array(fullLen);
    for (let i = inputSequence.length; i < fullLen; i++) {
        rlMask[i] = 1;
    }

    // STEP 4 — Scaled forward pass + optimize
    zeroGrads();
    kvCacheReset();
    lossAcc[0] = 0;
    let totalTokens = 0;

    for (let t = 0; t < fullLen - 1; t++) {
        forwardPass(fullSeq, t);
        softmaxInPlace(logits, probs, VOCAB_SIZE);
        const targetId = fullSeq[t + 1];

        if (rlMask[t] === 1) {
            const loss = -Math.log(Math.max(1e-7, Math.min(0.9999999, probs[targetId])));
            lossAcc[0] += loss;
            totalTokens++;

            // Tape entry must match backwardCrossEntropy's expected signature:
            // { op, probs, targetId, mask, realTokens, dLogits }
            tape.push({
                op:         "crossentropy",
                probs:      new Float32Array(probs),
                targetId,
                mask:       1,
                realTokens: totalTokens,  // updated progressively (safe approximation)
                dLogits:    dActivations.logits
            });
        }

    }

    const avgLoss = lossAcc[0] / Math.max(totalTokens, 1);
    const scaledLoss = avgLoss * (-rewardScore);
    lossOut[0] = scaledLoss;

    backward();
    const lr = getLr(globalStep);
    optimizer.step(lr);

    return scaledLoss;
}

// ── EXPORTS ───────────────────────────────────────────────
export {
  TOKEN_USER, TOKEN_ASST, TOKEN_END
}
