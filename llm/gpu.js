// ── gpu.js ────────────────────────────────────────────────
async function gpuMatMul(A, B, M, N, K) {
  if (!navigator.gpu) return cpuFallback(A, B, M, N, K);

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return cpuFallback(A, B, M, N, K);
  const device = await adapter.requestDevice();

  const shaderCode = `
    struct Dims {
      m: u32,
      n: u32,
      k: u32,
    };

    @group(0) @binding(0) var<storage, read> A: array<f32>;
    @group(0) @binding(1) var<storage, read> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;
    @group(0) @binding(3) var<uniform> dims: Dims;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row = global_id.y;
      let col = global_id.x;

      if (row >= dims.m || col >= dims.n) {
        return;
      }

      var sum = 0.0;
      for (var k = 0u; k < dims.k; k = k + 1u) {
        sum = sum + A[row * dims.k + k] * B[k * dims.n + col];
      }
      C[row * dims.n + col] = sum;
    }
  `;

  const shaderModule = device.createShaderModule({ code: shaderCode });

  const bufA = createGPUBuffer(device, A, GPUBufferUsage.STORAGE);
  const bufB = createGPUBuffer(device, B, GPUBufferUsage.STORAGE);
  const bufC = device.createBuffer({
    size: M * N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bufDims = device.createBuffer({
    size: 12,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(bufDims, 0, new Uint32Array([M, N, K]));

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufC } },
      { binding: 3, resource: { buffer: bufDims } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "main" },
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
  passEncoder.end();

  const gpuReadBuffer = device.createBuffer({
    size: M * N * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  commandEncoder.copyBufferToBuffer(bufC, 0, gpuReadBuffer, 0, M * N * 4);

  device.queue.submit([commandEncoder.finish()]);

  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(gpuReadBuffer.getMappedRange().slice());
  gpuReadBuffer.unmap();

  return result;
}

function createGPUBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function cpuFallback(A, B, M, N, K) {
  const C = new Float32Array(M * N);
  if (typeof matmul === 'function') {
    matmul(A, B, C, M, N, K);
  } else {
    // Minimal fallback logic if matmul isn't in scope
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        for (let j = 0; j < N; j++) {
          C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
      }
    }
  }
  return C;
}
