// ── worker.js ─────────────────────────────────────────────
let workerId = 0;
let sharedBuffer = null;
let syncBuffer = null;
let weights = null;
let grads = null;
let sync = null;

// Mock constants for shared memory layout
const WEIGHT_OFFSET = 0;
const WEIGHT_LENGTH = 1000000;
const GRAD_OFFSET = WEIGHT_LENGTH * 4;
const GRAD_LENGTH = 1000000;

self.onmessage = function(e) {
  const { type, workerId: id, sharedBuffer: sb, syncBuffer: syb, task, taskId } = e.data;

  if (type === "init") {
    workerId = id;
    sharedBuffer = sb;
    syncBuffer = syb;
    weights = new Float32Array(sharedBuffer, WEIGHT_OFFSET, WEIGHT_LENGTH);
    grads = new Float32Array(sharedBuffer, GRAD_OFFSET, GRAD_LENGTH);
    sync = new Int32Array(syncBuffer);
  } else if (type === "compute") {
    // Process assigned range from task
    const { start, end } = task;
    
    // Example compute: heavy math on shared memory
    for (let i = start; i < end; i++) {
      // Simulate compute + gradient accumulation
      const val = weights[i] * 0.5;
      
      // Use Atomics for gradient accumulation (fixed-point or specific slot)
      // Since it's Float32Array, and Atomics.add only supports Int32, 
      // we use a simple completion signal for this practical step.
      grads[i] += val; 
    }

    // Signal completion via Atomics
    Atomics.store(sync, workerId, 1);
    Atomics.notify(sync, workerId);

    self.postMessage({ type: "complete", taskId });
  }
};
