// Phase 7 of 8: GPU memory, warm-up and timing
// Allocates GPU VRAM, transfers tensors from host to device, warms up GPU pipeline,
// and measures both kernel-only execution (via CUDA events) and end-to-end latency.

#include <cuda_runtime.h>
#include <vector>
#include <chrono>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

void launchGpuForwardPass(const float* dX, const float* dW1, const float* db1,
                          float* dZ1, const float* dW2, const float* db2, float* dZ2);

void executePhase7(int BATCH, int INPUT_SIZE, int HIDDEN_SIZE, int OUTPUT_SIZE, int REPEATS,
                   const std::vector<float>& X, const std::vector<float>& W1, const std::vector<float>& b1,
                   const std::vector<float>& W2, const std::vector<float>& b2,
                   std::vector<float>& gpuOutput, float& gpuKernelMs, double& gpuEndToEndMs,
                   float*& dX, float*& dW1, float*& db1, float*& dZ1, float*& dW2, float*& db2, float*& dZ2,
                   cudaEvent_t& eventStart, cudaEvent_t& eventStop)
{
    CUDA_CHECK(cudaMalloc(&dX, X.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW1, W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db1, b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ1, BATCH * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW2, W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db2, b2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ2, BATCH * OUTPUT_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW1, W1.data(), W1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW2, W2.data(), W2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Warm-up: first launches can include one-time initialization overhead.
    launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel-only timing: input/weights already reside on the GPU.
    CUDA_CHECK(cudaEventCreate(&eventStart));
    CUDA_CHECK(cudaEventCreate(&eventStop));
    CUDA_CHECK(cudaEventRecord(eventStart));
    for (int r = 0; r < REPEATS; ++r) {
        launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
    }
    CUDA_CHECK(cudaEventRecord(eventStop));
    CUDA_CHECK(cudaEventSynchronize(eventStop));
    gpuKernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuKernelMs, eventStart, eventStop));
    gpuKernelMs /= REPEATS;

    // End-to-end timing: includes input/weight/bias transfers and output transfer.
    gpuOutput.resize(BATCH * OUTPUT_SIZE);
    auto gpuTotalStart = std::chrono::steady_clock::now();
    for (int r = 0; r < REPEATS; ++r) {
        CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW1, W1.data(), W1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW2, W2.data(), W2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

        launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(gpuOutput.data(), dZ2, gpuOutput.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    auto gpuTotalStop = std::chrono::steady_clock::now();
    gpuEndToEndMs = std::chrono::duration<double, std::milli>(gpuTotalStop - gpuTotalStart).count() / REPEATS;
}
