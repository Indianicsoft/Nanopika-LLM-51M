import { gpuDevice, executeGPUMatMul } from "../webgpu_matmul.js";

// ── MODULE 1: F16 Support Check ─────────────────────────────
export async function checkF16Support() {
  if (!navigator.gpu) return false;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return false;
  return adapter.features.has("shader-f16");
}

// ── MODULE 2: F16 GEMM Shader ──────────────────────────────
export const GEMM_F16_WGSL = /* wgsl */ `
  enable f16;

  struct Dims { M: u32, N: u32, K: u32 }
  @group(0) @binding(0) var<uniform>             dims : Dims;
  @group(0) @binding(1) var<storage, read>       matA : array<f16>;
  @group(0) @binding(2) var<storage, read>       matB : array<f16>;
  @group(0) @binding(3) var<storage, read_write> matC : array<f32>;

  const TILE_SIZE = 16u;
  var<workgroup> tileA : array<f16, 256>;
  var<workgroup> tileB : array<f16, 256>;

  @compute @workgroup_size(16, 16)
  fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id) wid : vec3<u32>
  ) {
    let row = gid.x;
    let col = gid.y;
    let lx = lid.x;
    let ly = lid.y;

    var acc : f32 = 0.0;
    let numTiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < numTiles; t++) {
      let kA = t * TILE_SIZE + ly;
      let kB = t * TILE_SIZE + lx;

      if (row < dims.M && kA < dims.K) {
        tileA[lx * TILE_SIZE + ly] = matA[row * dims.K + kA];
      } else {
        tileA[lx * TILE_SIZE + ly] = 0.0h;
      }

      if (kB < dims.K && col < dims.N) {
        tileB[lx * TILE_SIZE + ly] = matB[kB * dims.N + col];
      } else {
        tileB[lx * TILE_SIZE + ly] = 0.0h;
      }

      workgroupBarrier();

      for (var k = 0u; k < TILE_SIZE; k++) {
        acc += f32(tileA[lx * TILE_SIZE + k]) * f32(tileB[k * TILE_SIZE + ly]);
      }

      workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
      matC[row * dims.N + col] = acc;
    }
  }
`;

// ── MODULE 3: F16 MatMul Execution ─────────────────────────
export async function executeGPUMatMulF16(A, B, M, N, K) {
  const gpuF16Compute = await checkF16Support();
  
  if (!gpuF16Compute) {
    console.warn("F16 not supported, falling back to F32 computation.");
    return executeGPUMatMul(A, B, M, N, K);
  }

  if (!executeGPUMatMulF16._pipeline) {
    const shaderModule = gpuDevice.createShaderModule({ code: GEMM_F16_WGSL });
    executeGPUMatMulF16._pipeline = await gpuDevice.createComputePipelineAsync({
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "main" },
    });
  }
  const pipeline = executeGPUMatMulF16._pipeline;

  const dimsBuffer = gpuDevice.createBuffer({ size: 12, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  gpuDevice.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([M, N, K]));

  const bufA = gpuDevice.createBuffer({ size: A.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuDevice.queue.writeBuffer(bufA, 0, A);

  const bufB = gpuDevice.createBuffer({ size: B.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuDevice.queue.writeBuffer(bufB, 0, B);

  const outBytes = M * N * 4;
  const bufC = gpuDevice.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const stagingBuf = gpuDevice.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const bindGroup = gpuDevice.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: dimsBuffer } },
      { binding: 1, resource: { buffer: bufA } },
      { binding: 2, resource: { buffer: bufB } },
      { binding: 3, resource: { buffer: bufC } },
    ],
  });

  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(M / 16), Math.ceil(N / 16), 1);
  pass.end();
  
  encoder.copyBufferToBuffer(bufC, 0, stagingBuf, 0, outBytes);
  gpuDevice.queue.submit([encoder.finish()]);
  await gpuDevice.queue.onSubmittedWorkDone();

  await stagingBuf.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();

  dimsBuffer.destroy();
  bufA.destroy();
  bufB.destroy();
  bufC.destroy();
  stagingBuf.destroy();

  return result;
}
