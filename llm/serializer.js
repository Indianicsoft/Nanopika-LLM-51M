export function exportWeights(weightViews, filename = "model.bin") {
  let totalFloats = 0;
  for (const view of weightViews) {
    totalFloats += view.length;
  }

  const outBuffer = new ArrayBuffer(totalFloats * 4);
  const outF32 = new Float32Array(outBuffer);
  
  let offset = 0;
  for (const view of weightViews) {
    // Use Float32Array.set() which works regardless of whether
    // the source buffer is a regular or SharedArrayBuffer
    outF32.set(view, offset);
    offset += view.length;
  }

  const blob = new Blob([outBuffer], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
  console.log("[exportWeights]", filename, "(" + (outBuffer.byteLength / (1024*1024)).toFixed(1) + " MB)");
}

export function exportModel(weightsView, filename = "model_weights.bin") {
  // Create a new regular ArrayBuffer and copy from the view.
  // weightsView.buffer may be a SharedArrayBuffer, which cannot be sliced
  // or passed to Blob directly. This copy ensures compatibility.
  const byteLen = weightsView.byteLength;
  const buffer = new ArrayBuffer(byteLen);
  const dst = new Uint8Array(buffer);
  const src = new Uint8Array(weightsView.buffer, weightsView.byteOffset, byteLen);
  dst.set(src);

  const blob   = new Blob([buffer], { type: "application/octet-stream" });
  const url    = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href          = url;
  anchor.download      = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);

  console.log("[exportModel]", filename,
    "(" + (byteLen / (1024*1024)).toFixed(1) + " MB)");
}

// ── MODULE 2: loadWeights(source, weightViews) ─────────────
export async function loadWeights(source, weightViews) {
  let arrayBuf;
  if (source instanceof Blob || source instanceof File) {
    arrayBuf = await source.arrayBuffer();
  } else {
    const response = await fetch(source);
    if (!response.ok) {
      throw new Error(`Failed to fetch weights from ${source}: ${response.status} ${response.statusText}`);
    }
    arrayBuf = await response.arrayBuffer();
  }

  // Defensive Check: Total expected size vs Actual size
  let expectedTotalBytes = 0;
  for (const view of weightViews) {
    expectedTotalBytes += view.byteLength;
  }

  if (arrayBuf.byteLength < expectedTotalBytes) {
    const expectedMB = (expectedTotalBytes / (1024 * 1024)).toFixed(2);
    const actualMB = (arrayBuf.byteLength / (1024 * 1024)).toFixed(2);
    console.warn(`[Serializer] Buffer underflow. Expected ${expectedMB} MB, but file is only ${actualMB} MB. Stopping restoration.`);
    return false;
  }
  
  let offset = 0;
  for (let i = 0; i < weightViews.length; i++) {
    const view = weightViews[i];
    if (offset + view.byteLength > arrayBuf.byteLength) {
      console.warn(`[Serializer] Unexpected end of buffer at weight_${i}. Stopping restoration.`);
      break;
    }
    const chunk = new Uint8Array(arrayBuf, offset, view.byteLength);
    const dest = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
    dest.set(chunk);
    offset += view.byteLength;
  }

  console.log(`[Serializer] Successfully loaded ${offset} bytes into ${weightViews.length} views.`);
  return true;
}

// ── MODULE 3A: quantizeToF16(f32View) ─────────────────────
// Float16Array requires Chrome 120+, Firefox 129+, Safari 18.2+
export function quantizeToF16(f32View) {
  if (typeof Float16Array === "undefined") {
    console.warn("Float16Array is not supported in this environment. Skipping quantization.");
    return f32View;
  }
  const f16 = new Float16Array(f32View.length);
  for (let i = 0; i < f32View.length; i++) {
    f16[i] = f32View[i];
  }
  return f16;
}

// ── MODULE 3B: dequantizeToF32(f16View, outF32View) ───────
export function dequantizeToF32(f16View, outF32View) {
  // call dequantizeToF32(f16Weights, scratch0) before matmul
  // scratch0 then holds F32-upcast weights for the math engine
  for (let i = 0; i < f16View.length; i++) {
    outF32View[i] = f16View[i];
  }
}

// ── MODULE 3C: exportF16Weights(weightViews, filename) ────
export function exportF16Weights(weightViews, filename = "model_f16.bin") {
  if (typeof Float16Array === "undefined") {
    return exportWeights(weightViews, filename);
  }
  const quantizedViews = weightViews.map(view => quantizeToF16(view));
  exportWeights(quantizedViews, filename);
}
