# Runnable examples on Jetson

These examples generate their own test inputs. Use the JetPack CUDA toolkit and select the compute capability for your device. The commands below use `sm_87` for Orin.

## Mapped CPU and CUDA memory

The CPU initializes a mapped pinned buffer, CUDA increments it through its device alias, and the CPU checks the same allocation after the stream completes.

```sh
mkdir -p build
nvcc -std=c++17 -arch=sm_87 -O2 examples/mapped_host.cu -o build/mapped-host
./build/mapped-host
```

## CUDA writes directly into an OpenGL texture

Use a Jetson X11 graphical session with NVIDIA GLX and the system OpenGL and X11 development headers. From SSH, set `DISPLAY` and `XAUTHORITY` for the active session. This sample uses an offscreen GLX buffer. It validates all pixels with an OpenGL readback after the CUDA handoff.

```sh
mkdir -p build
nvcc -std=c++17 -arch=sm_87 -O2 examples/gl_texture.cu -lGL -lX11 -o build/gl-texture
./build/gl-texture
```

See the [Depth Pro 2 display experiment](../benchmarks/depth-pro-2/README.md) for a complete desktop preview with CUDA and OpenGL.

## TensorRT and PyTorch share device buffers

Use Python with JetPack's TensorRT 10 bindings and CUDA-enabled PyTorch. The script builds a small fixed-shape engine, binds its input and output to PyTorch CUDA tensors, then runs a PyTorch operation on the output using the same stream.

```sh
python3 examples/trt_device_io.py
```

See the [implementation guide](../docs/jetson-memory-handoffs.md) for equivalent OpenCV CUDA views, stream handoffs, and media surfaces.
