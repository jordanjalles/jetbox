const canvas = document.getElementById('canvas');
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const context = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
canvas.width = canvas.clientWidth * window.devicePixelRatio;
canvas.height = canvas.clientHeight * window.devicePixelRatio;
context.configure({ device, format, alphaMode: 'premultiplied' });

const shaderCode = `
@group(0) @binding(0) var<storage, read_write> pixels : array<u32>;
@group(0) @binding(1) var<uniform> params : struct {
  width : u32;
  height : u32;
  zoom : f32;
  offsetX : f32;
  offsetY : f32;
  maxIter : u32;
};

fn mandelbrot(cRe : f32, cIm : f32) -> u32 {
  var zRe = 0.0;
  var zIm = 0.0;
  var i : u32 = 0u;
  while (i < params.maxIter && (zRe*zRe + zIm*zIm) <= 4.0) {
    let tmp = zRe*zRe - zIm*zIm + cRe;
    zIm = 2.0*zRe*zIm + cIm;
    zRe = tmp;
    i = i + 1u;
  }
  return i;
}

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let x = f32(gid.x);
  let y = f32(gid.y);
  let re = (x / f32(params.width) - 0.5) * params.zoom + params.offsetX;
  let im = (y / f32(params.height) - 0.5) * params.zoom + params.offsetY;
  let iter = mandelbrot(re, im);
  let color = if (iter == params.maxIter) {
    0x000000ffu
  } else {
    let t = f32(iter) / f32(params.maxIter);
    let r = u32(9.0 * (1.0 - t) * t * t * t * 255.0);
    let g = u32(15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0);
    let b = u32(8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0);
    (b << 24u) | (g << 16u) | (r << 8u) | 0xffu
  };
  let idx = gid.y * params.width + gid.x;
  pixels[idx] = color;
}
`;

const shaderModule = device.createShaderModule({ code: shaderCode });
const pipeline = device.createComputePipeline({ compute: { module: shaderModule, entryPoint: 'main' } });

const bufferSize = canvas.width * canvas.height * 4;
const pixelBuffer = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const uniformBuffer = device.createBuffer({ size: 4*6, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
  { binding: 0, resource: { buffer: pixelBuffer } },
  { binding: 1, resource: { buffer: uniformBuffer } }
] });

function render() {
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(canvas.width/16), Math.ceil(canvas.height/16));
  passEncoder.end();
  commandEncoder.copyBufferToTexture({ buffer: pixelBuffer, bytesPerRow: canvas.width*4, rowsPerImage: canvas.height }, { texture: context.getCurrentTexture() }, [canvas.width, canvas.height]);
  device.queue.submit([commandEncoder.finish()]);
}

const params = {
  width: canvas.width,
  height: canvas.height,
  zoom: 3.0,
  offsetX: -0.5,
  offsetY: 0.0,
  maxIter: 200u
};
const uniformData = new Uint32Array([params.width, params.height, Math.float32Array.from([params.zoom])[0], Math.float32Array.from([params.offsetX])[0], Math.float32Array.from([params.offsetY])[0], params.maxIter]);
device.queue.writeBuffer(uniformBuffer, 0, uniformData.buffer);

render();
