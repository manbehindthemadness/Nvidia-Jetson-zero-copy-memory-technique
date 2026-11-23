#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

__global__ void add_one(float *samples, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        samples[i] += 1;
}

int main() try {
    int device, can_map;
    check(cudaGetDevice(&device), "Read CUDA device");
    check(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, device), "Query host mapping");
    if (!can_map)
        throw std::runtime_error("The selected device cannot map pinned host memory");

    constexpr int count = 4096;
    constexpr int threads = 128;
    float *host = nullptr;
    check(cudaHostAlloc(reinterpret_cast<void **>(&host), count * sizeof(float), cudaHostAllocMapped),
          "Allocate mapped host memory");
    std::unique_ptr<float, decltype(&cudaFreeHost)> allocation(host, cudaFreeHost);
    float *mapped = nullptr;
    check(cudaHostGetDevicePointer(reinterpret_cast<void **>(&mapped), host, 0), "Get CUDA alias");

    for (int i = 0; i < count; ++i)
        host[i] = static_cast<float>(i);
    add_one<<<(count + threads - 1) / threads, threads>>>(mapped, count);
    check(cudaGetLastError(), "Launch add_one");
    check(cudaStreamSynchronize(nullptr), "Wait before CPU reads the allocation");
    for (int i = 0; i < count; ++i)
        if (host[i] != static_cast<float>(i + 1))
            throw std::runtime_error("CPU did not observe the expected CUDA result");

    std::cout << "CPU and CUDA verified " << count << " values in one allocation\n";
    return EXIT_SUCCESS;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
}
