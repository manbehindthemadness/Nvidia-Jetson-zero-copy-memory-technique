"""Bind TensorRT to PyTorch CUDA allocations and consume its output on the same stream."""

import tensorrt as trt
import torch


def main():
    shape = (1, 1, 8, 8)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    tensor = network.add_input("input", trt.float32, shape)
    layer = network.add_elementwise(tensor, tensor, trt.ElementWiseOperation.SUM)
    result = layer.get_output(0)
    result.name = "output"
    network.mark_output(result)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT could not build the sample network")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("TensorRT could not load the sample engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT could not create the sample execution context")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        pixels = torch.arange(64, device="cuda", dtype=torch.float32).reshape(shape)
        output = torch.empty_like(pixels)
        for name, buffer in (("input", pixels), ("output", output)):
            if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise RuntimeError(f"Unexpected packed TensorRT layout for {name}")
            if not context.set_tensor_address(name, buffer.data_ptr()):
                raise RuntimeError(f"TensorRT could not bind {name}")
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT dispatch failed")
        squared = output.square()
        # Verification returns a scalar to the CPU after the GPU consumer has run.
        if not torch.equal(squared, (2 * pixels).square()):
            raise RuntimeError("PyTorch did not observe the expected TensorRT output")
    print("TensorRT and PyTorch verified the shared CUDA output")


if __name__ == "__main__":
    main()
