// Phase 5 of 8: The three kernel tests
// These functions test each kernel individually on tiny inputs whose mathematical answers
// can be verified by hand before full network execution.

#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

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
bool nearlyEqual(float a, float b, float tolerance);

bool testMatrixKernel()
{
    // A is 2x3, B is 3x2. Expected A*B = [22,28; 49,64].
    // Add bias [1,-1] => [23,27; 50,63].
    std::vector<float> hA = {1, 2, 3, 4, 5, 6};
    std::vector<float> hB = {1, 2, 3, 4, 5, 6};
    std::vector<float> hBias = {1, -1};
    std::vector<float> hC(4, 0.0f);

    float *dA = nullptr, *dB = nullptr, *dBias = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBias, hBias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), hBias.size() * sizeof(float), cudaMemcpyHostToDevice));

    matrixMulBiasKernel<<<dim3(1, 1), dim3(16, 16)>>>(dA, dB, dBias, dC, 2, 2, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = nearlyEqual(hC[0], 23.0f, 1e-5f) && nearlyEqual(hC[1], 27.0f, 1e-5f) &&
              nearlyEqual(hC[2], 50.0f, 1e-5f) && nearlyEqual(hC[3], 63.0f, 1e-5f);

    cudaFree(dA); cudaFree(dB); cudaFree(dBias); cudaFree(dC);
    return ok;
}

bool testReLUKernel()
{
    std::vector<float> h = {-2.0f, 0.0f, 3.0f, -4.0f, 5.0f};
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (static_cast<int>(h.size()) + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(d, static_cast<int>(h.size()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = nearlyEqual(h[0], 0.0f, 1e-5f) && nearlyEqual(h[1], 0.0f, 1e-5f) &&
              nearlyEqual(h[2], 3.0f, 1e-5f) && nearlyEqual(h[3], 0.0f, 1e-5f) &&
              nearlyEqual(h[4], 5.0f, 1e-5f);

    cudaFree(d);
    return ok;
}

bool testSigmoidKernel()
{
    std::vector<float> h = {-2.0f, 0.0f, 2.0f};
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (static_cast<int>(h.size()) + threads - 1) / threads;
    sigmoidKernel<<<blocks, threads>>>(d, static_cast<int>(h.size()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = nearlyEqual(h[0], 1.0f / (1.0f + std::exp(2.0f)), 1e-5f) &&
              nearlyEqual(h[1], 0.5f, 1e-5f) &&
              nearlyEqual(h[2], 1.0f / (1.0f + std::exp(-2.0f)), 1e-5f);

    cudaFree(d);
    return ok;
}
