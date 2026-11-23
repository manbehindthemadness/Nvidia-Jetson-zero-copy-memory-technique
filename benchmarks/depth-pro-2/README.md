# Depth Pro 2-S on Jetson AGX Orin

This study measures the handoff from TensorRT inference to a live RGB and inverse-depth display. It holds the TensorRT engine, OAK input, depth scaling, colormap, and OpenGL renderer fixed. Each mode produces the same side-by-side RGBA image. The CPU mode uses NumPy and OpenCV for colorization. Both GPU modes run the same CUDA colorization, including percentile-based scaling and temporal smoothing.

| Mode | Inference-to-display handoff |
|---|---|
| `cpu` | Copy inverse depth to the CPU, colorize and compose in host memory, upload the RGBA texture |
| `gpu-copy` | Colorize and compose into a linear CUDA allocation, then copy it to the mapped OpenGL texture |
| `gpu-surface` | Colorize and compose directly into the mapped texture via CUDA surface writes |

The input is a USB OAK-D Pro W preview. DepthAI delivers planar RGB bytes to host memory. All paths normalize those bytes to FP32 on the CPU and upload the TensorRT input. The GPU paths reuse the input tensor for the RGB display panel and consume the `inverse_depth` output on the inference stream. [TensorRT input contract](https://github.com/manbehindthemadness/Depth-Pro-2-S-on-Nvidia-AGX-Orin-TensorRT-JetPack-7.2.1/tree/v1.0.1#prebuilt-engines) · [CUDA–OpenGL implementation](interop.cu) · [Benchmark driver](benchmark.py)

## Live results

Jetson AGX Orin ran in MAXN with JetPack 7.2.1-b49, Jetson Linux 39.2.1, CUDA 13.2, TensorRT 10.16.2.10, NVIDIA OpenGL 595.78, Python 3.12, OpenCV 4.8.0, and DepthAI 2.31.1.0. The OAK negotiated USB SuperSpeed. The RGB preview was set to 30 FPS with a one-frame host queue.

| Resolution | CPU FPS / latency | GPU copy FPS / latency | GPU surface FPS / latency |
|---|---:|---:|---:|
| 320×200 | 30.06 / 60.86 ms | 30.08 / 53.56 ms | 30.09 / 54.15 ms |
| 640×400 | 30.01 / 74.58 ms | 30.08 / 57.87 ms | 30.07 / 57.74 ms |
| 864×544 | 22.27 / 108.49 ms | 30.04 / 70.93 ms | 30.04 / 70.92 ms |
| 1280×800 | 10.56 / 166.44 ms | 16.94 / 131.82 ms | 17.00 / 131.03 ms |

Latency is the median time from camera exposure midpoint to OpenGL swap submission after the draw completed. It ends before desktop compositor presentation and panel scanout. Each mode ran two eight-second measurement blocks, with 12 warmup frames per block and reversed mode order for the second block. Vblank synchronization was requested off. The benchmark omits text overlays and counts processed frames separately from camera sequence gaps. [Live measurements, including p95 and frame counts](results/oak.json)

At **864×544**, direct surface writes increased throughput **35%** and reduced median latency **35%**. The CPU path recorded **122** sequence gaps. Both GPU paths recorded **zero**. At 320×200 and 640×400, the GPU paths reduced latency while all three modes reached camera cadence. At 1280×800 the surface path processed about 17 FPS, with about 50 ms of TensorRT GPU work per frame.

## Unpaced display workload

The same fixed RGB inputs were cycled without camera pacing. Each mode timed 120 frames per resolution after warmup. Frame work includes input preparation, inference, compositing, drawing, and swap submission.

| Resolution | CPU p50 / p95 | GPU copy p50 / p95 | GPU surface p50 / p95 |
|---|---:|---:|---:|
| 320×200 | 9.12 / 9.48 ms | 5.53 / 5.89 ms | 5.53 / 5.60 ms |
| 640×400 | 27.46 / 27.87 ms | 13.16 / 13.61 ms | 13.23 / 13.51 ms |
| 864×544 | 43.64 / 45.93 ms | 22.23 / 22.74 ms | 22.20 / 22.71 ms |
| 1280×800 | 92.85 / 95.69 ms | 55.24 / 55.65 ms | 55.36 / 55.70 ms |

All three modes produced byte-identical OpenGL textures for the two validation inputs at each resolution. Verification read back the texture before timing and checked the engine's inverse depth against the saved Apple reference. The 864×544 workload fell **49%** with direct surface writes. The two GPU paths performed similarly, which puts most of the measured gain at the CPU readback, colorization, and upload boundary. [Unpaced measurements and numerical checks](results/fixture.json)

## Inference-only baseline

The independent batch-one comparison timed warmed, GPU-resident inference with CUDA events. Each median used 72 samples. The PyTorch baseline ran eager `2.12.1+cu132` with cuDNN 9.20 and original Apple weights. Internal TensorRT tactics use mixed precision. Both frameworks accepted the same RGB inputs, and all outputs used for timing were finite. The PyTorch wheel emitted a support warning for Orin SM 8.7.

| Resolution | TensorRT engine | PyTorch CUDA BF16 | PyTorch CUDA FP16 |
|---|---:|---:|---:|
| 320×200 | 4.02 ms | 33.21 ms | 33.15 ms |
| 640×400 | 10.54 ms | 36.81 ms | 37.18 ms |
| 864×544 | 18.62 ms | 41.43 ms | 41.61 ms |
| 1280×800 | 48.91 ms | 76.24 ms | 74.80 ms |

These timings compare inference implementations. The live display modes above all use the same TensorRT engine, so their differences measure changes in the display path. [Inference rerun, with pinned-transfer results and numerical checks](results/trt-vs-pytorch.json) · [Original published TensorRT benchmark](https://github.com/manbehindthemadness/Depth-Pro-2-S-on-Nvidia-AGX-Orin-TensorRT-JetPack-7.2.1/tree/v1.0.1#tensorrt-versus-pytorch-cuda)

## Reproduce the display benchmark

Run from this repository's root on an Orin with compatible JetPack TensorRT bindings, CUDA Toolkit, X11 and OpenGL development headers, and Python NumPy and OpenCV. The renderer uses NVIDIA GLX and needs access to an active X11 graphical session. The OAK path also requires DepthAI 2.31.1.0 and an attached OAK RGB camera. OpenCV's CPU APIs are sufficient to reproduce these modes.

Select the engine for your resolution from the [published engine repository](https://github.com/manbehindthemadness/Depth-Pro-2-S-on-Nvidia-AGX-Orin-TensorRT-JetPack-7.2.1/tree/v1.0.1#prebuilt-engines). The following assumes the 864×544 archive was extracted to `work/model`.

```sh
nvcc -std=c++17 -arch=sm_87 -O2 -Xcompiler=-fPIC -shared \
  benchmarks/depth-pro-2/interop.cu -lGL -lX11 -o benchmarks/depth-pro-2/interop.so

python3 benchmarks/depth-pro-2/benchmark.py --source oak --size 864 544 \
  --engine work/model/depthpro2-s-864x544-fp16.engine --seconds 8 \
  --report work/live-864x544.json
```

For unpaced inputs, generate a `.npz` with the same width and height using the [published export procedure](https://github.com/manbehindthemadness/Depth-Pro-2-S-on-Nvidia-AGX-Orin-TensorRT-JetPack-7.2.1/tree/v1.0.1#build-on-jetson). The fixture is written beside the ONNX export. The next command assumes you placed that fixture in `work/model`.

```sh
python3 benchmarks/depth-pro-2/benchmark.py --source fixture --size 864 544 \
  --engine work/model/depthpro2-s-864x544-fp16.engine \
  --fixture work/model/depthpro2-s-864x544.npz --samples 60 \
  --report work/fixture-864x544.json
```

Pass `--fixture` to the OAK command to run the same validation before live timing. Use `--source verify` with a fixture for validation alone. Select a mode with `--modes cpu`, `--modes gpu-copy`, or `--modes gpu-surface`. The commands preserve each resolution's measurements in a separate JSON file.

## Model attribution

Depth Pro 2 is by Peiyun Hu, Marko Mihajlovic, Lyndon Duong, Lahav Lipson, Wei Dong, Amaël Delaunoy, Stephan R. Richter, and Vladlen Koltun. This repository contains benchmark source and numeric measurements. Obtain model weights and TensorRT derivatives from their published projects and follow their terms. [Apple Depth Pro 2 source](https://github.com/apple-aiml-research/ml-depth-pro-2) · [Model card](https://huggingface.co/apple/DepthPro2) · [Engine attribution and license files](https://github.com/manbehindthemadness/Depth-Pro-2-S-on-Nvidia-AGX-Orin-TensorRT-JetPack-7.2.1/tree/v1.0.1#attribution-and-licenses)

Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement. [Apple model agreement](https://github.com/manbehindthemadness/Depth-Pro-2-S-on-Nvidia-AGX-Orin-TensorRT-JetPack-7.2.1/blob/v1.0.1/licenses/APPLE-MODEL.txt)
