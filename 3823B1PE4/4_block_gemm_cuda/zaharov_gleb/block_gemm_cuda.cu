#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

struct DeviceDeleter {
    void operator()(float* pointer) const noexcept {
        cudaFree(pointer);
    }
};

using DeviceBuffer = std::unique_ptr<float, DeviceDeleter>;

DeviceBuffer AllocateDevice(std::size_t bytes) {
    float* pointer = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes), "cudaMalloc");
    return DeviceBuffer(pointer);
}

constexpr unsigned int kTileSize = 32;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                std::size_t n) {
    __shared__ float tile_a[kTileSize][kTileSize];
    __shared__ float tile_b[kTileSize][kTileSize];
    __shared__ float tile_c[kTileSize][kTileSize];

    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * kTileSize + y;
    const std::size_t column = static_cast<std::size_t>(blockIdx.x) * kTileSize + x;
    tile_c[y][x] = 0.0f;

    for (std::size_t offset = 0; offset < n; offset += kTileSize) {
        // Adjacent threads load adjacent elements of each row.
        tile_a[y][x] = row < n && offset + x < n ? a[row * n + offset + x] : 0.0f;
        tile_b[y][x] = offset + y < n && column < n ? b[(offset + y) * n + column] : 0.0f;
        __syncthreads();

        float sum = tile_c[y][x];
        #pragma unroll
        for (unsigned int k = 0; k < kTileSize; ++k) {
            sum = fmaf(tile_a[y][k], tile_b[k][x], sum);
        }
        tile_c[y][x] = sum;
        // All threads must finish reading before the next tile is loaded.
        __syncthreads();
    }

    if (row < n && column < n) {
        c[row * n + column] = tile_c[y][x];
    }
}

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                               const std::vector<float>& b,
                               int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix dimension must be non-negative");
    }
    const std::size_t dimension = static_cast<std::size_t>(n);
    if (dimension != 0 &&
        dimension > std::numeric_limits<std::size_t>::max() / sizeof(float) / dimension) {
        throw std::length_error("Matrix size is too large");
    }
    const std::size_t size = dimension * dimension;
    if (a.size() != size || b.size() != size) {
        throw std::invalid_argument("Input matrices must contain n * n elements");
    }
    if (size == 0) {
        return {};
    }

    const std::size_t bytes = size * sizeof(float);
    const auto device_a = AllocateDevice(bytes);
    const auto device_b = AllocateDevice(bytes);
    const auto device_c = AllocateDevice(bytes);
    CheckCuda(cudaMemcpy(device_a.get(), a.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy A host to device");
    CheckCuda(cudaMemcpy(device_b.get(), b.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy B host to device");

    const dim3 threads(kTileSize, kTileSize);
    const unsigned int tiles = static_cast<unsigned int>((dimension + kTileSize - 1) / kTileSize);
    const dim3 blocks(tiles, tiles);
    BlockGemmKernel<<<blocks, threads>>>(device_a.get(), device_b.get(), device_c.get(), dimension);
    CheckCuda(cudaGetLastError(), "BlockGemmKernel launch");

    std::vector<float> result(size);
    // This blocking copy also waits for kernel execution to complete.
    CheckCuda(cudaMemcpy(result.data(), device_c.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy C device to host");
    return result;
}
