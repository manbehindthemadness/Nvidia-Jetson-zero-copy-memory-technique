# Implementing Jetson memory handoffs

## Start with the boundary

For each frame, write down the producer, consumer, allocator, buffer shape and format, and the operation that signals completion. A zero-copy handoff lets the consumer access the producer's existing allocation or view. A kernel can still write a new output when it changes the layout or computes a result. Measure the full pipeline to see whether skipping a copy, CPU operation, or synchronization point helps your workload.

Jetson's shared physical DRAM, CUDA unified virtual addressing, and CUDA Unified Memory answer different questions. Jetson describes where the memory lives. Unified virtual addressing describes how CUDA identifies addresses. `cudaMallocManaged` creates an allocation with CUDA's managed-access and coherency rules. The performance of device, pinned host, registered host, and managed allocations depends on the Jetson generation, cache behavior, and access pattern. Query the device and benchmark the allocation types with your producer and consumer. [NVIDIA CUDA for Tegra memory management](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-for-tegra-appnote/index.html#memory-management) · [CUDA Runtime memory API](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/group__CUDART__MEMORY.html)

## CPU and CUDA use one allocation

1. Query `cudaDevAttrCanMapHostMemory` and allocate with `cudaHostAlloc(..., cudaHostAllocMapped)`. If an existing host allocation is suitable for pinning, check `cudaHostRegister` support and use its registration and unregistration API. Ordinary pinned staging with `cudaMemcpyAsync` is another option for bulk traffic.
2. Obtain the CUDA alias with `cudaHostGetDevicePointer`. Keep the host pointer for CPU access and the alias for the kernel. This works even when the numerical addresses differ.
3. Finish CPU writes before CUDA uses the buffer. Synchronize the CUDA stream, or wait for a completion event, before CPU reads GPU results or the producer reuses the buffer. Keep the allocation alive through the final use. Use `cudaFreeHost` for CUDA-allocated pinned memory and `cudaHostUnregister` for registered host memory.

Mapped memory is a good candidate when the CPU produces a small payload or CUDA reads it only once. Repeated GPU reads can favor a copy into cacheable device memory. Test both paths and account for the producer's ability to write into mapped storage in the first place. The runnable sample verifies CPU-visible results from a CUDA kernel using one allocation. [Mapped host example](../examples/mapped_host.cu) · [NVIDIA Tegra allocation guidance](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-for-tegra-appnote/index.html#memory-management) · [CUDA memory API](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/group__CUDART__MEMORY.html)

## TensorRT writes to a buffer the next CUDA stage can consume

Allocate the input and output on the GPU, or reuse buffers owned by a framework such as PyTorch or CuPy. Check each TensorRT binding's memory location, current shape, type, and format. For a fixed, linear tensor, pass the device pointer to `set_tensor_address` and enqueue with `execute_async_v3`. Dynamic shapes and packed layouts need their actual strides, capacity, and alignment. Data-dependent output sizes can require an output allocator. [TensorRT Python inference](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html) · [TensorRT execution context API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/_static/python-api/infer/Core/ExecutionContext.html)

Use buffers, the execution context, and streams that belong to the same CUDA device and a compatible CUDA context. Use one stream for GPU preprocessing, inference, and postprocessing, or record an event on the producer stream and wait on the consumer stream. Keep the owning tensor, context, and stream alive through the last consumer. Give each in-flight frame an independent buffer slot, or wait for completion before recycling a slot. The example runs a small TensorRT network and lets PyTorch consume the exact output allocation. [TensorRT device I/O example](../examples/trt_device_io.py)

### Add OpenCV CUDA processing

With a CUDA-enabled OpenCV build that exposes the external-memory APIs, create a `GpuMat` view over a compatible image allocation. For example, the following uses an H×W×3 interleaved CuPy `uint8` device array named `bgr` and its producing CuPy CUDA stream named `producer_stream`.

```python
height, width, channels = bgr.shape
assert channels == 3 and bgr.dtype == cupy.uint8
assert bgr.strides[1:] == (3, 1)
assert bgr.strides[0] >= width * 3
image = cv2.cuda.createGpuMatFromCudaMemory(
    height, width, cv2.CV_8UC3, bgr.data.ptr, bgr.strides[0]
)
stream = cv2.cuda.wrapStream(producer_stream.ptr)
gray = cv2.cuda.cvtColor(image, cv2.COLOR_BGR2GRAY, stream=stream)
```

`step` is the row pitch in bytes. Keep `bgr` and `producer_stream` alive until OpenCV is done. The grayscale operation produces a new GPU allocation for the transformed image. The input `GpuMat` shares the original allocation. Planar NCHW tensors require a layout conversion before this interleaved three-channel view can describe them. `GpuMat` support in `imshow` depends on the HighGUI backend and build. [OpenCV CUDA memory and stream APIs](https://docs.opencv.org/4.13.0/d9/d41/group__cudacore__struct.html) · [HighGUI CUDA display issue](https://github.com/opencv/opencv/issues/22328)

## CUDA writes into an OpenGL texture

1. Create a graphics context on the CUDA-capable GPU, make it current, and allocate a supported texture format such as `GL_RGBA8`. Register the texture once with `cudaGraphicsGLRegisterImage` and `cudaGraphicsRegisterFlagsSurfaceLoadStore`.
2. Per frame, call `cudaGraphicsMapResources` on the stream that will write the image. Get the current CUDA array with `cudaGraphicsSubResourceGetMappedArray`. A registered texture is exposed as a CUDA array. A registered GL buffer can expose a linear device pointer.
3. Build a CUDA surface object for the mapped array. Have the kernel read the linear input tensors and use `surf2Dwrite` to write the final pixel values. The X coordinate to `surf2Dwrite` is a byte offset. Keep the surface alive until the CUDA work that uses it completes.
4. Destroy the surface after completion and unmap the graphics resource before OpenGL samples the texture. The CUDA interop map and unmap APIs order access between OpenGL and the CUDA stream. Use resources only during their valid ownership period. Unregister before deleting the texture.

The examples use an explicit stream wait before surface destruction. A renderer can use multiple texture slots and fences to overlap frames once it has verified correct ownership and lifetimes. A simpler GPU-resident alternative is to colorize into a linear CUDA buffer and copy it into the mapped array with `cudaMemcpy2DToArrayAsync`. That retains a GPU-to-GPU transfer while avoiding host readback. This alternative performed similarly to direct writes in the [Depth Pro 2 case study](../benchmarks/depth-pro-2/README.md). [CUDA–OpenGL API](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/group__CUDART__OPENGL.html) · [CUDA graphics resource synchronization](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/group__CUDART__INTEROP.html) · [Model-free texture example](../examples/gl_texture.cu)

## Import a camera or decoder surface when available

When a Jetson capture or decoder stack exports a supported DMA-BUF or `NvBufSurface`, retain the buffer and import it using the APIs for that stack. For a Jetson `NVBUF_MEM_SURFACE_ARRAY`, `NvBufSurfaceMapEglImage` can expose an EGLImage. Register the EGLImage with CUDA and retrieve its `cudaEglFrame`. Use the returned plane layout and pitch or array handles. Apply the producer's wait and cache synchronization rules before CUDA reads the surface and defer requeue or destruction until CUDA completes its use. An NV12 surface may still need a color and layout conversion before a planar RGB TensorRT input can use it. [Jetson r39.2 `NvBufSurface` reference](https://docs.nvidia.com/jetson/archives/r39.2/ApiReference/group__ds__aaa.html) · [NVIDIA video decode and CUDA example](https://docs.nvidia.com/jetson/archives/r39.2/ApiReference/l4t_mm_02_video_dec_cuda.html) · [Tegra EGL interoperability](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-for-tegra-appnote/index.html#egl-interoperability)

The USB OAK used in the benchmark supplies host frames through DepthAI. Its transport and host-to-CUDA input upload remain in the measured pipeline. The GPU-to-display handoff applies after the frame is on the Jetson GPU. [Benchmark input path](../benchmarks/depth-pro-2/README.md)

## Verify and measure the change

- Compare the baseline and shared-buffer paths on the same input. Verify the values or pixels after the handoff. Keep validation readbacks outside the timed loop.
- Use CUDA events for GPU execution. Measure host wall time for preprocessing, copies, map and unmap, and presentation submission. For live workloads, also record frame age, sequence gaps, throughput, p50 and p95, power mode, and software versions.
- Profile a short run with `nsys profile --trace=cuda,nvtx --output=handoff <your-command>`. Inspect H2D, D2H, and D2D operations and GPU idle periods. A copy-free boundary can still use memory bandwidth or wait for another producer. [Nsight Systems CUDA tracing](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace)
