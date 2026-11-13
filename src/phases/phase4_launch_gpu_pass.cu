// Phase 4 of 8: Launching the GPU forward pass
// This function launches the four GPU kernels in order (matrix multiply, ReLU, matrix multiply, sigmoid).
// nearlyEqual is a helper function used by unit tests to compare floating-point values.

#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void matrixMulBiasKernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K);
__global__ void reluKernel(float* values, int count);
__global__ void sigmoidKernel(float* values, int count);

void launchGpuForwardPass(const float* dX, const float* dW1, const float* db1,
                          float* dZ1, const float* dW2, const float* db2,
                          float* dZ2)
{
    constexpr int BATCH       = 256;
    constexpr int INPUT_SIZE  = 128;
    constexpr int HIDDEN_SIZE = 64;
    constexpr int OUTPUT_SIZE = 10;

    dim3 block(16, 16);
    dim3 grid1((HIDDEN_SIZE + block.x - 1) / block.x,
               (BATCH + block.y - 1) / block.y);
    matrixMulBiasKernel<<<grid1, block>>>(
        dX, dW1, db1, dZ1, BATCH, HIDDEN_SIZE, INPUT_SIZE);

    int hiddenCount = BATCH * HIDDEN_SIZE;
    int threads = 256;
    int blocks = (hiddenCount + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(dZ1, hiddenCount);

    dim3 grid2((OUTPUT_SIZE + block.x - 1) / block.x,
               (BATCH + block.y - 1) / block.y);
    matrixMulBiasKernel<<<grid2, block>>>(
        dZ1, dW2, db2, dZ2, BATCH, OUTPUT_SIZE, HIDDEN_SIZE);

    int outputCount = BATCH * OUTPUT_SIZE;
    int outputBlocks = (outputCount + threads - 1) / threads;
    sigmoidKernel<<<outputBlocks, threads>>>(dZ2, outputCount);

    CUDA_CHECK(cudaGetLastError());
}

bool nearlyEqual(float a, float b, float tolerance = 1e-5f)
{
    return std::fabs(a - b) <= tolerance;
}
