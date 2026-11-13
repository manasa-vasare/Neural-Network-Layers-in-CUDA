// Phase 2 of 8: The three GPU kernels
// These are the functions that run on the GPU: matrix multiplication with bias, ReLU, and sigmoid.

#include <cuda_runtime.h>

// Each CUDA thread calculates one output matrix element and adds its bias.
__global__ void matrixMulBiasKernel(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum + bias[col];
    }
}

// Each thread applies ReLU to one element.
__global__ void reluKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] = fmaxf(0.0f, values[idx]);
    }
}

// Each thread applies sigmoid to one element.
__global__ void sigmoidKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] = 1.0f / (1.0f + expf(-values[idx]));
    }
}
