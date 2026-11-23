#!/usr/bin/env python3
"""Compare CPU readback, CUDA texture copy, and direct CUDA surface rendering."""

import argparse
import ctypes as c
from ctypes.util import find_library
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import sys
from time import monotonic, sleep

HERE = Path(__file__).resolve().parent

import cv2
import numpy as np
import tensorrt as trt

try:
    import depthai as dai
except ImportError:
    dai = None

MODES = ("cpu", "gpu-copy", "gpu-surface")
HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2


class TimedEngine:
    def __init__(self, path, width, height):
        self.cuda = c.CDLL(find_library("cudart") or "libcudart.so")
        pointer = c.c_void_p
        signatures = {
            "GetErrorString": (c.c_char_p, c.c_int),
            "Malloc": (c.c_int, c.POINTER(pointer), c.c_size_t),
            "Free": (c.c_int, pointer),
            "StreamCreate": (c.c_int, c.POINTER(pointer)),
            "StreamSynchronize": (c.c_int, pointer),
            "StreamDestroy": (c.c_int, pointer),
            "MemcpyAsync": (c.c_int, pointer, pointer, c.c_size_t, c.c_int, pointer),
            "EventCreate": (c.c_int, c.POINTER(pointer)),
            "EventRecord": (c.c_int, pointer, pointer),
            "EventElapsedTime": (c.c_int, c.POINTER(c.c_float), pointer, pointer),
            "EventDestroy": (c.c_int, pointer),
        }
        for name, (result, *args) in signatures.items():
            api = getattr(self.cuda, f"cuda{name}")
            api.restype = result
            api.argtypes = args

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not load {path}")
        expected = {"pixel_values": ((1, 3, height, width), trt.TensorIOMode.INPUT),
                    "inverse_depth": ((1, height, width), trt.TensorIOMode.OUTPUT)}
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if names != expected.keys():
            raise ValueError(f"Unexpected engine I/O {sorted(names)}")
        for name, (shape, mode) in expected.items():
            if (tuple(self.engine.get_tensor_shape(name)) != shape or
                    self.engine.get_tensor_dtype(name) != trt.float32 or
                    self.engine.get_tensor_mode(name) != mode or
                    self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE or
                    self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise ValueError(f"{name} must be a device-resident linear FP32 tensor of shape {shape}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create a TensorRT execution context")
        self.stream = c.c_void_p()
        self.begin = c.c_void_p()
        self.end = c.c_void_p()
        self.host = {}
        self.device = {}
        try:
            self.check(self.cuda.cudaStreamCreate(c.byref(self.stream)))
            self.check(self.cuda.cudaEventCreate(c.byref(self.begin)))
            self.check(self.cuda.cudaEventCreate(c.byref(self.end)))
            for name, (shape, _) in expected.items():
                host = np.empty(shape, np.float32)
                device = c.c_void_p()
                self.check(self.cuda.cudaMalloc(c.byref(device), host.nbytes))
                self.host[name] = host
                self.device[name] = device
                if not self.context.set_tensor_address(name, device.value):
                    raise RuntimeError(f"Could not bind {name}")
        except (RuntimeError, MemoryError):
            self.close()
            raise

    def check(self, status):
        if status:
            raise RuntimeError(self.cuda.cudaGetErrorString(status).decode())

    def infer(self, readback):
        image = self.host["pixel_values"]
        self.check(self.cuda.cudaMemcpyAsync(self.device["pixel_values"], image.ctypes.data,
                                             image.nbytes, HOST_TO_DEVICE, self.stream))
        self.check(self.cuda.cudaEventRecord(self.begin, self.stream))
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError("TensorRT dispatch failed")
        self.check(self.cuda.cudaEventRecord(self.end, self.stream))
        if readback:
            result = self.host["inverse_depth"]
            self.check(self.cuda.cudaMemcpyAsync(result.ctypes.data, self.device["inverse_depth"],
                                                 result.nbytes, DEVICE_TO_HOST, self.stream))
            self.check(self.cuda.cudaStreamSynchronize(self.stream))
            return result[0]
        return None

    def compute_ms(self):
        elapsed = c.c_float()
        self.check(self.cuda.cudaEventElapsedTime(c.byref(elapsed), self.begin, self.end))
        return elapsed.value

    def close(self):
        if self.stream:
            self.cuda.cudaStreamSynchronize(self.stream)
        self.context = None
        for event in (self.begin, self.end):
            if event:
                self.cuda.cudaEventDestroy(event)
        for address in self.device.values():
            self.cuda.cudaFree(address)
        if self.stream:
            self.cuda.cudaStreamDestroy(self.stream)


class Viewer:
    def __init__(self, width, height):
        self.handle = c.c_void_p()
        self.lib = c.CDLL(str(HERE / "interop.so"))
        signatures = {
            "error": (c.c_char_p,),
            "create": (c.c_int, c.c_int, c.c_int, c.c_void_p, c.POINTER(c.c_void_p)),
            "renderer": (c.c_char_p, c.c_void_p),
            "reset": (None, c.c_void_p),
            "events": (c.c_int, c.c_void_p),
            "upload": (None, c.c_void_p, c.c_void_p),
            "gpu": (c.c_int, c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_int),
            "draw": (None, c.c_void_p),
            "read": (c.c_int, c.c_void_p, c.c_void_p),
            "destroy": (None, c.c_void_p),
        }
        for name, (result, *args) in signatures.items():
            function = getattr(self.lib, f"viewer_{name}")
            function.restype = result
            function.argtypes = args
        palette = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_MAGMA)
        palette = cv2.cvtColor(palette, cv2.COLOR_BGR2RGBA)
        self.check(self.lib.viewer_create(width, height, palette.ctypes.data, c.byref(self.handle)))
        self.renderer = self.lib.viewer_renderer(self.handle).decode()
        self.rgba = np.full((height, 2 * width, 4), 255, np.uint8)
        self.width = width
        self.limit = None

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.viewer_error().decode())

    def poll(self):
        if not self.lib.viewer_events(self.handle):
            raise InterruptedError("Experiment window closed")

    def reset(self):
        self.limit = None
        self.lib.viewer_reset(self.handle)

    def from_cpu(self, depth, raw_rgb):
        np.nan_to_num(depth, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        limit = max(float(np.percentile(depth[::8, ::8], 98)), .01)
        self.limit = limit if self.limit is None else .92 * self.limit + .08 * limit
        gray = np.clip(depth * (255 / self.limit), 0, 255).astype(np.uint8)
        self.rgba[:, :self.width, :3] = raw_rgb.transpose(1, 2, 0)
        self.rgba[:, self.width:, :3] = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)[:, :, ::-1]
        self.lib.viewer_upload(self.handle, self.rgba.ctypes.data)

    def from_gpu(self, engine, direct):
        self.check(self.lib.viewer_gpu(self.handle, engine.device["pixel_values"],
                                       engine.device["inverse_depth"], engine.stream, direct))

    def draw(self):
        self.lib.viewer_draw(self.handle)

    def read(self):
        pixels = np.empty_like(self.rgba)
        self.check(self.lib.viewer_read(self.handle, pixels.ctypes.data))
        return pixels

    def close(self):
        self.lib.viewer_destroy(self.handle)


@contextmanager
def experiment(path, width, height):
    # The OpenGL context must be current for CUDA texture registration.
    viewer = Viewer(width, height)
    try:
        engine = TimedEngine(path, width, height)
        try:
            yield engine, viewer
        finally:
            engine.close()
    finally:
        viewer.close()


def render(engine, viewer, mode, raw_rgb):
    depth = engine.infer(readback=mode == "cpu")
    if mode == "cpu":
        viewer.from_cpu(depth, raw_rgb)
    else:
        viewer.from_gpu(engine, direct=mode == "gpu-surface")
    ready = monotonic()
    viewer.draw()
    return ready


def verify(engine, viewer, fixture):
    verification = defaultdict(list)
    textures = defaultdict(list)
    input_buffer = engine.host["pixel_values"][0]
    for image, reference in zip(fixture["pixels"], fixture["inverse_depth"], strict=True):
        np.copyto(input_buffer, image)
        depth = engine.infer(readback=True)
        mae = float(np.mean(np.abs(depth - reference)))
        if not np.isfinite(depth).all() or mae >= .02:
            raise RuntimeError(f"Unexpected TensorRT fixture output, MAE {mae}")
        verification["depth_mae"].append(mae)
    for mode in MODES:
        viewer.reset()
        for image in fixture["pixels"]:
            np.copyto(input_buffer, image)
            raw_rgb = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
            render(engine, viewer, mode, raw_rgb)
            textures[mode].append(viewer.read())
    for cpu, copy, surface in zip(*(textures[mode] for mode in MODES), strict=True):
        if not np.array_equal(copy, surface):
            raise RuntimeError("CUDA copy and CUDA surface produced different textures")
        if not np.array_equal(cpu[:, :viewer.width], surface[:, :viewer.width]):
            raise RuntimeError("CPU and GPU RGB panels differ")
        delta = np.abs(cpu.astype(np.int16) - surface)
        max_delta = int(delta.max())
        match = float(np.all(delta == 0, axis=2).mean())
        if max_delta > 4 or match < .999:
            raise RuntimeError(f"CPU and GPU colormaps differ, max error {max_delta}, match {match:.5f}")
        verification["texture"].append({"exact_pixel_fraction": match, "max_channel_delta": max_delta})
    return dict(verification)


def next_frame(queue, viewer):
    deadline = monotonic() + 8
    while monotonic() < deadline:
        viewer.poll()
        frame = queue.tryGet()
        if frame is not None:
            return frame
        sleep(.0005)
    raise TimeoutError("No OAK RGB frame in eight seconds")


def oak_pipeline(width, height):
    pipeline = dai.Pipeline()
    camera = pipeline.create(dai.node.ColorCamera)
    camera.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camera.setPreviewKeepAspectRatio(True)
    camera.setPreviewSize(width, height)
    camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camera.setInterleaved(False)
    camera.setFps(30)
    output = pipeline.create(dai.node.XLinkOut)
    output.setStreamName("rgb")
    output.input.setBlocking(False)
    output.input.setQueueSize(4)
    camera.preview.link(output.input)
    return pipeline


def measure(args, engine, viewer, mode, frames, queue, width, height):
    viewer.reset()
    metrics = defaultdict(list)
    sequences = []
    warmup = args.warmup
    index = 0
    input_buffer = engine.host["pixel_values"][0]
    started = None
    while started is None or (monotonic() - started < args.seconds if queue else index < warmup + args.samples):
        viewer.poll()
        frame = next_frame(queue, viewer) if queue else None
        if index == warmup:
            started = monotonic()
        begin = monotonic()
        if frame is not None:
            if (frame.getWidth(), frame.getHeight(), frame.getType()) != (width, height, dai.ImgFrame.Type.RGB888p):
                raise RuntimeError("The OAK RGB preview differs from the engine input format")
            rgb = frame.getData().reshape(3, height, width)
            np.multiply(rgb, np.float32(1 / 255), out=input_buffer, dtype=np.float32)
        else:
            image, rgb = frames[index % len(frames)]
            np.copyto(input_buffer, image)
        prepared = monotonic()
        texture_ready = render(engine, viewer, mode, rgb)
        done = monotonic()
        if index >= warmup:
            metrics["frame_work_ms"].append(1000 * (done - begin))
            metrics["upload_infer_and_compose_ms"].append(1000 * (texture_ready - prepared))
            metrics["draw_and_swap_ms"].append(1000 * (done - texture_ready))
            metrics["tensorrt_gpu_ms"].append(engine.compute_ms())
            if frame is not None:
                sequences.append(frame.getSequenceNum())
                exposure = frame.getTimestamp(dai.CameraExposureOffset.MIDDLE)
                metrics["exposure_to_swap_ms"].append((dai.Clock.now() - exposure).total_seconds() * 1000)
        index += 1
    elapsed = monotonic() - started
    count = len(metrics["frame_work_ms"])
    if not count:
        raise RuntimeError("No measured frames")
    report = {"frames": count, "seconds": elapsed, "frames_per_second": count / elapsed, "samples_ms": dict(metrics)}
    if sequences:
        report["sequence_gaps"] = sequences[-1] - sequences[0] + 1 - count
    return report


def summarize(values):
    mean = float(np.mean(values))
    p50, p95 = map(float, np.quantile(values, [.5, .95]))
    return {"mean": mean, "p50": p50, "p95": p95, "max": max(values)}


def combine(blocks):
    count = sum(block["frames"] for block in blocks)
    elapsed = sum(block["seconds"] for block in blocks)
    report = {"frames": count, "seconds": elapsed, "frames_per_second": count / elapsed,
              "block_fps": [block["frames_per_second"] for block in blocks]}
    for key in blocks[0]["samples_ms"]:
        report[key] = summarize([value for block in blocks for value in block["samples_ms"][key]])
    if "sequence_gaps" in blocks[0]:
        report["sequence_gaps"] = sum(block["sequence_gaps"] for block in blocks)
    return report


def load_fixture(path, width, height):
    with np.load(path) as archive:
        images = archive["pixels"]
        reference = archive["inverse_depth"]
    if (images.ndim != 4 or images.shape[1:] != (3, height, width) or not len(images) or
            reference.shape != (len(images), height, width) or
            images.dtype != np.float32 or reference.dtype != np.float32):
        raise ValueError(f"Expected FP32 NCHW pixels and NHW inverse_depth at {width}x{height} in {path}")
    return {"pixels": images, "inverse_depth": reference}


def run(args):
    width, height = args.size
    size = f"{width}x{height}"
    fixture = load_fixture(args.fixture, width, height) if args.fixture else None
    with experiment(args.engine, width, height) as (engine, viewer):
        result = {"engine": args.engine.name, "renderer": viewer.renderer}
        if fixture is not None:
            result["verification"] = verify(engine, viewer, fixture)
        if args.source == "verify":
            return result

        live = args.source == "oak"
        frames = None if live else [(image, np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))
                                    for image in fixture["pixels"]]
        capture = dai.Device(oak_pipeline(width, height)) if live else nullcontext()
        with capture as oak:
            queue = oak.getOutputQueue("rgb", maxSize=1, blocking=False) if live else None
            if live:
                result["usb_speed"] = oak.getUsbSpeed().name
            blocks = defaultdict(list)
            for round_number in range(args.rounds):
                order = args.modes if round_number % 2 == 0 else args.modes[::-1]
                for mode in order:
                    block = measure(args, engine, viewer, mode, frames, queue, width, height)
                    blocks[mode].append(block)
                    print(f"{size} {mode} round {round_number + 1}: {block['frames_per_second']:.2f} fps, "
                          f"{np.median(block['samples_ms']['frame_work_ms']):.2f} ms frame work", flush=True)
            result["modes"] = {mode: combine(runs) for mode, runs in blocks.items()}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("verify", "fixture", "oak"), required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), required=True)
    parser.add_argument("--fixture", type=Path, help="Reference .npz produced by export_depth.py")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=36, help="Measured frames per fixture block")
    parser.add_argument("--seconds", type=float, default=8, help="Measured seconds per OAK block")
    parser.add_argument("--warmup", type=int, default=12, help="Warmup frames per block")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rounds, args.samples, args.seconds) <= 0 or args.warmup < 0:
        parser.error("Rounds, samples and seconds must be positive, and warmup cannot be negative")
    if min(args.size) <= 0 or not args.engine.is_file():
        parser.error("Specify a positive width and height and a readable engine file")
    if args.source in ("fixture", "verify") and args.fixture is None:
        parser.error("--fixture is required for the fixture and verify sources")
    if args.source == "oak" and dai is None:
        parser.error("Install DepthAI 2.31.1.0 for the OAK source")
    if not os.environ.get("DISPLAY"):
        parser.error("Run with access to the Jetson's graphical X11 session")
    # NVIDIA's GLX driver reads this when the library creates its first context.
    os.environ.setdefault("__GL_SYNC_TO_VBLANK", "0")
    cv2.setNumThreads(2)
    size = "x".join(map(str, args.size))
    report = {"source": args.source, "tensorrt": trt.__version__, "opencv": cv2.__version__,
              "depthai": dai.__version__ if dai else None,
              "requested_vblank_sync": os.environ["__GL_SYNC_TO_VBLANK"],
              "rounds": args.rounds, "warmup_frames": args.warmup, "runs": {size: run(args)}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.report}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (InterruptedError, KeyboardInterrupt):
        sys.exit("Experiment stopped")
