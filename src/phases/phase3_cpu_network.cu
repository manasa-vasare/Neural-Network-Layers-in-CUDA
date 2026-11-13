// Phase 3 of 8: The CPU version of the network
// These ordinary C++ functions compute the same forward pass on the CPU,
// serving as the ground truth reference to validate GPU correctness.

#include <vector>
#include <cmath>
#include <algorithm>

extern const int BATCH;
extern const int INPUT_SIZE;
extern const int HIDDEN_SIZE;
extern const int OUTPUT_SIZE;

void cpuMatMulBias(const std::vector<float>& A,
                   const std::vector<float>& B,
                   const std::vector<float>& bias,
                   std::vector<float>& C,
                   int M, int N, int K)
{
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum + bias[col];
        }
    }
}

void cpuReLU(std::vector<float>& values)
{
    for (float& x : values) x = std::max(0.0f, x);
}

void cpuSigmoid(std::vector<float>& values)
{
    for (float& x : values) x = 1.0f / (1.0f + std::exp(-x));
}

void cpuForwardPass(const std::vector<float>& X,
                    const std::vector<float>& W1,
                    const std::vector<float>& b1,
                    const std::vector<float>& W2,
                    const std::vector<float>& b2,
                    std::vector<float>& output)
{
    std::vector<float> z1(256 * 64); // BATCH * HIDDEN_SIZE
    cpuMatMulBias(X, W1, b1, z1, 256, 64, 128);
    cpuReLU(z1);

    std::vector<float> z2(256 * 10); // BATCH * OUTPUT_SIZE
    cpuMatMulBias(z1, W2, b2, z2, 256, 10, 64);
    cpuSigmoid(z2);

    output = z2;
}
