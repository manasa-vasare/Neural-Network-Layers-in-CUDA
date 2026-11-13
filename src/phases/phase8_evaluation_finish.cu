// Phase 8 of 8: Compare, print results and finish
// Computes maximum absolute difference between CPU reference and GPU output,
// prints performance metrics, displays output values, releases allocated VRAM and events.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

int executePhase8(int BATCH, int INPUT_SIZE, int HIDDEN_SIZE, int OUTPUT_SIZE,
                  double cpuMs, float gpuKernelMs, double gpuEndToEndMs,
                  const std::vector<float>& cpuOutput, const std::vector<float>& gpuOutput,
                  cudaEvent_t eventStart, cudaEvent_t eventStop,
                  float* dX, float* dW1, float* db1, float* dZ1, float* dW2, float* db2, float* dZ2)
{
    float maxDifference = 0.0f;
    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        maxDifference = std::max(maxDifference, std::fabs(cpuOutput[i] - gpuOutput[i]));
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nForward-pass dimensions:\n";
    std::cout << "Input: " << BATCH << " x " << INPUT_SIZE << "\n";
    std::cout << "Hidden layer: " << BATCH << " x " << HIDDEN_SIZE << "\n";
    std::cout << "Output: " << BATCH << " x " << OUTPUT_SIZE << "\n";

    std::cout << "\nAverage CPU compute time: " << cpuMs << " ms\n";
    std::cout << "Average GPU kernel-only time: " << gpuKernelMs << " ms\n";
    std::cout << "Average GPU end-to-end time: " << gpuEndToEndMs << " ms\n";

    if (gpuKernelMs > 0.0f)
        std::cout << "Kernel-only speedup: " << cpuMs / gpuKernelMs << "x\n";
    if (gpuEndToEndMs > 0.0)
        std::cout << "End-to-end speedup: " << cpuMs / gpuEndToEndMs << "x\n";

    std::cout << "\nMaximum absolute output difference: " << maxDifference << "\n";
    std::cout << "CPU and GPU outputs: "
              << (maxDifference < 1e-4f ? "MATCH (within tolerance)" : "DO NOT MATCH")
              << "\n";

    std::cout << "\nFirst 10 output values (GPU):\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << gpuOutput[i] << (i == 9 ? '\n' : ' ');
    }

    CUDA_CHECK(cudaEventDestroy(eventStart));
    CUDA_CHECK(cudaEventDestroy(eventStop));
    cudaFree(dX); cudaFree(dW1); cudaFree(db1); cudaFree(dZ1);
    cudaFree(dW2); cudaFree(db2); cudaFree(dZ2);

    return (maxDifference < 1e-4f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
