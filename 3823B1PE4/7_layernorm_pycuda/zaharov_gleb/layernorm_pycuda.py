import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize


_SOURCE = r"""
#include <math.h>

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float partial[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    if (lane == 0) partial[warp] = value;
    __syncthreads();

    if (warp == 0) {
        value = lane < 8 ? partial[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0) partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}

extern "C" __global__ void layernorm(float *data, const float *gamma,
                                      const float *beta, int width, float eps) {
    const int tid = threadIdx.x;
    float *row = data + (size_t)blockIdx.x * width;

    // Subtract an anchor before summing to retain precision for nearly
    // constant rows with a large common offset.
    const float anchor = row[0];
    float sum = 0.0f;
    for (int col = tid; col < width; col += blockDim.x)
        sum += row[col] - anchor;
    const float centered_mean = block_sum(sum) / width;

    float squared_sum = 0.0f;
    for (int col = tid; col < width; col += blockDim.x) {
        const float delta = (row[col] - anchor) - centered_mean;
        squared_sum += delta * delta;
    }
    const float inv_std = rsqrtf(block_sum(squared_sum) / width + eps);

    for (int col = tid; col < width; col += blockDim.x) {
        const float delta = (row[col] - anchor) - centered_mean;
        row[col] = delta * inv_std * gamma[col] + beta[col];
    }
}
"""


@context_dependent_memoize
def _kernel():
    return SourceModule(_SOURCE).get_function("layernorm")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    if not isinstance(row_size, (int, np.integer)) or row_size <= 0:
        raise ValueError("row_size must be a positive integer")

    row_size = int(row_size)
    if row_size > np.iinfo(np.int32).max:
        raise ValueError("row_size is too large")

    eps = np.float32(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    x = np.ascontiguousarray(np.asarray(input, dtype=np.float32).reshape(-1))
    g = np.ascontiguousarray(np.asarray(gamma, dtype=np.float32).reshape(-1))
    b = np.ascontiguousarray(np.asarray(beta, dtype=np.float32).reshape(-1))
    if x.size % row_size or g.size != row_size or b.size != row_size:
        raise ValueError("input length must be divisible by row_size; gamma and beta must match it")

    result = np.empty_like(x)
    if x.size == 0:
        return result

    d_x = cuda.mem_alloc(x.nbytes)
    d_g = cuda.mem_alloc(g.nbytes)
    d_b = cuda.mem_alloc(b.nbytes)

    try:
        cuda.memcpy_htod(d_x, x)
        cuda.memcpy_htod(d_g, g)
        cuda.memcpy_htod(d_b, b)
        _kernel()(d_x, d_g, d_b, np.int32(row_size), eps,
                  block=(256, 1, 1), grid=(x.size // row_size, 1, 1))
        cuda.memcpy_dtoh(result, d_x)
    finally:
        d_b.free()
        d_g.free()
        d_x.free()
    return result
