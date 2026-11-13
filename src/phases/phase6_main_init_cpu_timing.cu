// Phase 6 of 8: Start of main (GPU check, tests, input data, CPU timing)
// Detects the GPU hardware, runs kernel unit tests, generates deterministic input data,
// and measures execution time for CPU forward pass.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

bool testMatrixKernel();
bool testReLUKernel();
bool testSigmoidKernel();
void cpuForwardPass(const std::vector<float>& X, const std::vector<float>& W1, const std::vector<float>& b1,
                    const std::vector<float>& W2, const std::vector<float>& b2, std::vector<float>& output);

// Example snippet showing Phase 6 execution flow
void executePhase6(int BATCH, int INPUT_SIZE, int HIDDEN_SIZE, int OUTPUT_SIZE, int REPEATS,
                   std::vector<float>& X, std::vector<float>& W1, std::vector<float>& b1,
                   std::vector<float>& W2, std::vector<float>& b2,
                   std::vector<float>& cpuOutput, double& cpuMs)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable NVIDIA GPU was detected." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA\n";
    std::cout << "GPU detected: " << prop.name << "\n\n";

    std::cout << "Testing individual CUDA kernels...\n";
    bool matrixOK  = testMatrixKernel();
    bool reluOK    = testReLUKernel();
    bool sigmoidOK = testSigmoidKernel();

    std::cout << "Matrix multiplication + bias: " << (matrixOK ? "PASSED" : "FAILED") << "\n";
    std::cout << "ReLU: " << (reluOK ? "PASSED" : "FAILED") << "\n";
    std::cout << "Sigmoid: " << (sigmoidOK ? "PASSED" : "FAILED") << "\n";

    if (!(matrixOK && reluOK && sigmoidOK)) {
        std::cerr << "A kernel test failed. Stop and debug before continuing.\n";
        std::exit(EXIT_FAILURE);
    }

    // Deterministic input and weights: both CPU and GPU receive identical values.
    X.resize(BATCH * INPUT_SIZE);
    W1.resize(INPUT_SIZE * HIDDEN_SIZE);
    b1.resize(HIDDEN_SIZE);
    W2.resize(HIDDEN_SIZE * OUTPUT_SIZE);
    b2.resize(OUTPUT_SIZE);

    for (size_t i = 0; i < X.size(); ++i)
        X[i] = 0.5f * std::sin(static_cast<float>(i) * 0.013f);

    for (size_t i = 0; i < W1.size(); ++i)
        W1[i] = 0.08f * std::cos(static_cast<float>(i) * 0.017f);

    for (size_t i = 0; i < b1.size(); ++i)
        b1[i] = 0.01f * static_cast<float>(i % 7);

    for (size_t i = 0; i < W2.size(); ++i)
        W2[i] = 0.10f * std::sin(static_cast<float>(i) * 0.019f);

    for (size_t i = 0; i < b2.size(); ++i)
        b2[i] = 0.02f * static_cast<float>(i % 5);

    cpuOutput.resize(BATCH * OUTPUT_SIZE);
    auto cpuStart = std::chrono::steady_clock::now();
    for (int r = 0; r < REPEATS; ++r) {
        cpuForwardPass(X, W1, b1, W2, b2, cpuOutput);
    }
    auto cpuStop = std::chrono::steady_clock::now();
    cpuMs = std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count() / REPEATS;
}
