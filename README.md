# Neural Network Layers in CUDA

[![CUDA](https://img.shields.io/badge/CUDA-13.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20RTX%205060%20Ti-76B900?logo=nvidia)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An optimized, from-scratch CUDA implementation of core neural network inference layers (matrix multiplication with bias, ReLU activation, and Sigmoid activation). This project implements and evaluates a two-layer feedforward neural network, validating numerical correctness between single-threaded CPU execution and massively parallel GPU execution while profiling acceleration metrics and PCIe transfer overhead.

---

## 📌 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Network Dimensions](#-network-dimensions)
- [Project Structure & Phases](#-project-structure--phases)
- [Experimental Setup](#-experimental-setup)
- [Benchmark Results & Performance](#-benchmark-results--performance)
  - [Multi-Run Stability Analysis](#multi-run-stability-analysis)
  - [Comparison with Reference Workstation](#comparison-with-reference-workstation)
  - [Numerical Output Verification](#numerical-output-verification)
- [Performance Insights & Analysis](#-performance-insights--analysis)
- [Build & Run Instructions](#-build--run-instructions)
- [Troubleshooting & Technical Insights](#-troubleshooting--technical-insights)
- [Viva-Voce / Technical Q&A](#-viva-voce--technical-qa)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🧠 Overview & Architecture

Modern Deep Learning inference relies heavily on GPU acceleration to parallelize matrix operations. This experiment models a complete two-layer multi-layer perceptron (MLP) forward pass:

$$\begin{aligned}
\mathbf{Layer\ 1:}\quad &\mathbf{Z}_1 = \mathbf{X} \mathbf{W}_1 + \mathbf{b}_1, \quad &&\mathbf{A}_1 = \text{ReLU}(\mathbf{Z}_1) \\
\mathbf{Layer\ 2:}\quad &\mathbf{Z}_2 = \mathbf{A}_1 \mathbf{W}_2 + \mathbf{b}_2, \quad &&\mathbf{Y} = \sigma(\mathbf{Z}_2) = \frac{1}{1 + e^{-\mathbf{Z}_2}}
\end{aligned}$$

### Custom CUDA Kernels
1. **`matrixMulBiasKernel`**: 2D grid/block layout where each CUDA thread computes one output scalar using fused inner products and adds the corresponding row/column bias.
2. **`reluKernel`**: 1D grid layout applying element-wise thresholding: $f(x) = \max(0, x)$.
3. **`sigmoidKernel`**: 1D grid layout computing non-linear logistic activation: $\sigma(x) = \frac{1}{1 + e^{-x}}$.

---

## 📐 Network Dimensions

| Tensor | Mathematical Shape | Dimensions in Code | Description |
| :--- | :--- | :--- | :--- |
| $\mathbf{X}$ | $\text{BATCH} \times \text{INPUT\_SIZE}$ | $256 \times 128$ | Input mini-batch |
| $\mathbf{W}_1$ | $\text{INPUT\_SIZE} \times \text{HIDDEN\_SIZE}$ | $128 \times 64$ | First layer weight matrix |
| $\mathbf{b}_1$ | $1 \times \text{HIDDEN\_SIZE}$ | $64$ elements | First layer bias vector |
| $\mathbf{A}_1$ | $\text{BATCH} \times \text{HIDDEN\_SIZE}$ | $256 \times 64$ | Hidden layer activations |
| $\mathbf{W}_2$ | $\text{HIDDEN\_SIZE} \times \text{OUTPUT\_SIZE}$ | $64 \times 10$ | Second layer weight matrix |
| $\mathbf{b}_2$ | $1 \times \text{OUTPUT\_SIZE}$ | $10$ elements | Second layer bias vector |
| $\mathbf{Y}$ | $\text{BATCH} \times \text{OUTPUT\_SIZE}$ | $256 \times 10$ | Final inference predictions |

---

## 📂 Project Structure & Phases

The code is organized both as a modular step-by-step breakdown (reflecting the 8 lab phases) and a unified production implementation:

```text
Neural-Network-Layers-in-CUDA/
├── README.md               # Comprehensive documentation and benchmark report
├── build.bat               # Windows compilation script
├── .gitignore              # Ignores build artifacts and binaries
└── src/
    ├── neural_network.cu   # Complete unified end-to-end program (Phases 1-8)
    └── phases/             # Step-by-step implementation phases
        ├── phase1_constants.cu            # Phase 1: Headers, macros, constants
        ├── phase2_gpu_kernels.cu          # Phase 2: MatMul+bias, ReLU, Sigmoid kernels
        ├── phase3_cpu_network.cu          # Phase 3: CPU reference implementation
        ├── phase4_launch_gpu_pass.cu      # Phase 4: Forward pass orchestrator
        ├── phase5_kernel_tests.cu         # Phase 5: Individual kernel unit tests
        ├── phase6_main_init_cpu_timing.cu # Phase 6: Device detection & CPU profiling
        ├── phase7_gpu_memory_timing.cu    # Phase 7: GPU memory, warm-up & timing
        └── phase8_evaluation_finish.cu    # Phase 8: Verification & output metrics
```

---

## 🖥 Experimental Setup

* **GPU:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM / 16,311 MiB)
* **GPU Driver:** 610.88 (CUDA UMD Version: 13.3, WDDM mode)
* **CUDA Toolkit:** v13.4 (nvcc release 13.4, V13.4.92)
* **Host Compiler:** Microsoft Visual C++ MSVC v143 (`cl.exe` v14.52.37025) via VS Developer Command Prompt
* **OS:** Microsoft Windows 11 (64-bit)

---

## 📊 Benchmark Results & Performance

### Multi-Run Stability Analysis
As specified by stability testing protocols, the network was evaluated across three distinct consecutive executions (averaged over 20 iterations each):

| Metric | Run 1 | Run 2 | Run 3 | Three-Run Average |
| :--- | :---: | :---: | :---: | :---: |
| **CPU Average Compute Time** | 0.924320 ms | 0.862445 ms | 1.025935 ms | **0.937567 ms** |
| **GPU Kernel-Only Time** | 0.022136 ms | 0.027530 ms | 0.029248 ms | **0.026305 ms** |
| **GPU End-to-End Time** | 0.118755 ms | 0.129475 ms | 0.146890 ms | **0.131707 ms** |
| **Kernel-Only Speedup** | **41.756417×** | **31.327915×** | **35.077100×** | **36.053811×** |
| **End-to-End Speedup** | **7.783420×** | **6.661093×** | **6.984376×** | **7.142963×** |
| **Max Absolute Difference** | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| **Output Match Status** | **MATCH** | **MATCH** | **MATCH** | **MATCH** |

---

### Comparison with Reference Workstation

Below is the observation table comparing our observed measurements on the **RTX 5060 Ti** against the baseline workstation (**RTX 4500 Ada Generation**):

| Measurement Parameter | Observed Value (Run 1) | Observed Value (Mean) | Reference (RTX 4500 Ada) | Unit |
| :--- | :---: | :---: | :---: | :---: |
| **CPU Average Compute Time** | `0.924320` | `0.937567` | `0.526860` | ms |
| **GPU Kernel-Only Time** | `0.022136` | `0.026305` | `0.026517` | ms |
| **GPU End-to-End Time** | `0.118755` | `0.131707` | `0.066010` | ms |
| **Kernel-Only Speedup** | **41.756417** | **36.053811** | `19.868913` | × |
| **End-to-End Speedup** | **7.783420** | **7.142963** | `7.981518` | × |
| **Maximum Absolute Difference** | `0.000000` | `0.000000` | `0.000000` | — |
| **Kernel Unit Tests** | **All PASSED** | **All PASSED** | All PASSED | — |
| **CPU / GPU Output Match** | **MATCH** | **MATCH** | MATCH | — |
| **GPU Architecture Model** | **RTX 5060 Ti** | **RTX 5060 Ti** | RTX 4500 Ada Gen | — |

---

### Numerical Output Verification

Both the CPU and GPU implementations were fed identical deterministic inputs generated by trigonometric progressions:

$$X[i] = 0.5 \cdot \sin(0.013 \cdot i), \quad W_1[i] = 0.08 \cdot \cos(0.017 \cdot i)$$

**First 10 GPU Output Values (Predictions):**
```text
0.498196 0.503101 0.508006 0.512911 0.517814 0.497729 0.502638 0.507548 0.512457 0.517364
```
Maximum difference across all 2,560 output elements was **0.000000**, well beneath the verification tolerance ($1 \times 10^{-4}$).

---

## 🔍 Performance Insights & Analysis

1. **Massive Compute Acceleration (up to 41.76×):**  
   The compute execution time on the GPU was as low as **0.022 ms**. Massive SIMD parallelism on CUDA cores drastically outperforms single-threaded host CPU nested loops for matrix operations.
2. **Impact of PCIe Overhead (Amdahl's Law):**  
   While raw kernel execution is accelerated by over **41×**, the end-to-end speedup drops to **~7.8×**. The PCIe bus overhead (`cudaMemcpy` of inputs, weights, and results) takes $\approx 0.096\text{ ms}$, representing **81.4% of total elapsed GPU pipeline time**. This underscores that for shallow networks and small batches, memory transfer over the PCIe bus is the primary bottleneck.

---

## 🛠 Build & Run Instructions

### Prerequisites
* Windows 10/11 (x64)
* Visual Studio 2022 or Visual Studio Build Tools with **Desktop development with C++** (`cl.exe`)
* NVIDIA CUDA Toolkit 12.x or 13.x with `nvcc` in PATH

### Compilation

Open the **Developer Command Prompt for VS** (or **x64 Native Tools Command Prompt**):

```cmd
# Clone repository
git clone https://github.com/manasa-vasare/Neural-Network-Layers-in-CUDA.git
cd Neural-Network-Layers-in-CUDA

# Compile with optimization flag
nvcc -O2 src/neural_network.cu -o neural_network.exe

# Execute
neural_network.exe
```

Alternatively, run the automated batch script:
```cmd
build.bat
```

---

## ⚙ Troubleshooting & Technical Insights

### Case Study: Resolving PTX Toolchain Version Mismatch
During testing, compiling with default settings triggered the following runtime exception:
```text
CUDA error at neural_network.cu:197 - the provided PTX was compiled with an unsupported toolchain.
```

* **Root Cause:** The installed CUDA compiler was **CUDA 13.4**, whereas the system display driver supported **CUDA 13.3**. When `nvcc` compiles without explicit target flags, it produces intermediate virtual PTX assembly. When the program runs, the driver's JIT compiler was unable to parse the newer PTX ISA specification.
* **Solution:** Explicitly target your GPU's native machine code (SASS) during compilation so the runtime does not require JIT PTX translation:
  ```cmd
  nvcc -O2 -arch=sm_89 src/neural_network.cu -o neural_network.exe
  ```
  *(Or update the NVIDIA display driver to match or exceed the CUDA toolkit version).*

---

## 💡 Viva-Voce / Technical Q&A

<details>
<summary><b>1. What is a CUDA kernel?</b></summary>
A function declared with the <code>__global__</code> execution specifier that runs in parallel on the GPU device across many threads, invoked via execution configuration syntax <code><<<grid, block>>></code>.
</details>

<details>
<summary><b>2. What is the difference between host and device memory?</b></summary>
Host memory is the system RAM accessible directly by the CPU. Device memory is high-bandwidth GDDR/VRAM located directly on the graphics card accessible by GPU streaming multiprocessors (SMs).
</details>

<details>
<summary><b>3. What do blockIdx, blockDim, and threadIdx represent?</b></summary>
Built-in CUDA variables: <code>threadIdx</code> gives the thread position inside a block, <code>blockDim</code> provides the number of threads per block dimension, and <code>blockIdx</code> indicates the block coordinate inside the grid.
</details>

<details>
<summary><b>4. Why is a boundary check required in a CUDA kernel?</b></summary>
Thread grids are launched in block multiples (typically multiples of 32 threads/warps). If matrix dimensions are not exact multiples of block size, boundary checks (e.g., <code>if (row < M && col < N)</code>) prevent out-of-bounds memory access.
</details>

<details>
<summary><b>5. Why are CUDA events used for kernel timing rather than host clocks?</b></summary>
CUDA kernel launches are asynchronous with respect to host CPU code. CPU timers only measure how fast calls are enqueued unless explicitly synchronized. CUDA events (<code>cudaEventRecord</code>) insert timestamps directly into the GPU command stream, enabling microsecond-precision hardware timing.
</details>

<details>
<summary><b>6. What is the difference between kernel-only and end-to-end timing?</b></summary>
Kernel-only timing measures purely execution of computation on the GPU cores when memory already resides in VRAM. End-to-end timing includes host-to-device memory transfers, kernel execution, synchronization, and device-to-host transfer.
</details>

<details>
<summary><b>7. Why does GPU data transfer overhead reduce speedup?</b></summary>
Host-to-device transfers are constrained by PCIe bus bandwidth, which is much slower than GPU VRAM memory bandwidth. For small models, data movement overhead exceeds total compute time.
</details>

<details>
<summary><b>8. Why is the GPU warmed up before timing?</b></summary>
Initial kernel launches include one-time overheads such as CUDA runtime context initialization, driver allocation, and instruction caching. Warm-up runs ensure measurements isolate steady-state execution.
</details>

---

## 📄 License
This repository is licensed under the [MIT License](LICENSE).
