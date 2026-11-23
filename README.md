# Zero-copy memory handoffs on NVIDIA Jetson

Keep a frame in the same allocation when the next stage can consume it directly. On Jetson, the CPU and integrated GPU share physical DRAM. CUDA, TensorRT, graphics, and media APIs still have their own allocation, layout, and synchronization rules. Pick the handoff supported by the producer and consumer, then measure what it saves on the target device. [NVIDIA CUDA for Tegra](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-for-tegra-appnote/index.html)

| Handoff | Technique |
|---|---|
| CPU-produced data → CUDA | Map pinned host memory and give CUDA its device alias |
| CUDA → TensorRT → CUDA | Bind compatible device pointers to the engine and order work on a CUDA stream |
| CUDA → OpenCV CUDA | Wrap an existing device pointer and stream with `GpuMat` and `Stream` |
| CUDA → OpenGL | Register a texture and write to its mapped CUDA surface |
| Supported camera or decoder → CUDA | Import the platform surface through DMA-BUF and EGLImage APIs |

## Start here

- [Implementation guide](docs/jetson-memory-handoffs.md) covers allocation, layout, ownership, synchronization, and profiling.
- [Runnable examples](examples/README.md) verify mapped CPU/CUDA memory, TensorRT-to-PyTorch device buffers, and CUDA writes into an OpenGL texture without a model.
- [Depth Pro 2-S case study](benchmarks/depth-pro-2/README.md) includes benchmark code, reproduction commands, and raw results for four resolutions.
- [Research history](docs/research-history.md) traces the public OpenCV and NVIDIA reports, the OAK Myriad X port, and the Jetson TensorRT builds.

## What changed in a live OAK viewer

The original display path read TensorRT inverse depth back to the CPU, colorized it, and uploaded the display image to OpenGL. The GPU paths reuse the device-resident output. One colorizes into a temporary CUDA buffer and copies it to the texture. The other colorizes directly into a mapped texture. All three use the same TensorRT engine and OpenGL renderer.

At **864×544** on Jetson AGX Orin, direct texture writes raised live throughput from **22.27 to 30.04 FPS** and cut median camera-exposure-to-swap time from **108.49 to 70.92 ms**. The CPU path recorded **122** camera sequence gaps during the two measured blocks. Both GPU paths recorded **zero**. The OAK preview was configured for 30 FPS. [Live measurements](benchmarks/depth-pro-2/results/oak.json)

The staged GPU path also reached **30.04 FPS**. In this workload, keeping colorization and display on the GPU delivered most of the improvement. All paths still upload the USB camera input from host memory. [Benchmark method and limitations](benchmarks/depth-pro-2/README.md)
