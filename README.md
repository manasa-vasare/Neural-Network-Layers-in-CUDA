# Experiment 5: Neural Network Layers in CUDA

[![CUDA](https://img.shields.io/badge/CUDA-13.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20RTX%205060%20Ti-76B900?logo=nvidia)](https://www.nvidia.com/)
[![Course](https://img.shields.io/badge/Lab-Cloud%20%2F%20GPU%20Computing-blue)](#)

> **Course / Subject:** Cloud / GPU Computing Laboratory  
> **Experiment No.:** 5  
> **Experiment Title:** Neural Network Layers in CUDA  
> **Platform:** Windows 11 (64-bit), CUDA Toolkit 13.4, Visual Studio Build Tools (MSVC v143)

---

## 🎯 Aim & Learning Outcomes

### Aim
To implement and evaluate basic neural-network layers using custom CUDA kernels:
1. Matrix multiplication with bias addition
2. Rectified Linear Unit (ReLU) activation
3. Sigmoid activation

Execute the identical forward pass on both the CPU and GPU, validate numerical correctness within tolerance, measure execution time, and evaluate GPU acceleration and speedup.

### Learning Outcomes
* Explain CUDA kernels, thread blocks, grid configurations, and memory models.
* Implement custom 2D grid matrix multiplication with bias in CUDA C++.
* Implement element-wise non-linear activation kernels (ReLU, Sigmoid).
* Allocate, initialize, and synchronize host-to-device and device-to-host data transfers.
* Compare single-threaded CPU reference results against parallel GPU executions.
* Profile kernel execution using CUDA events and evaluate PCIe transfer overhead.

---

## 📌 Table of Contents

- [Aim & Learning Outcomes](#-aim--learning-outcomes)
- [Mathematical Formulation & Theory](#-mathematical-formulation--theory)
- [Network Configuration & Dimensions](#-network-configuration--dimensions)
- [Project Structure & Implementation Phases](#-project-structure--implementation-phases)
- [Hardware & Software Specifications](#-hardware--software-specifications)
- [Laboratory Benchmark Observations](#-laboratory-benchmark-observations)
  - [Observation Table (Section 13)](#observation-table-section-13)
  - [Multi-Run Stability Analysis](#multi-run-stability-analysis)
  - [Numerical Output Verification](#numerical-output-verification)
- [Performance Insights & Discussion](#-performance-insights--discussion)
- [Compilation & Execution Guide](#-compilation--execution-guide)
- [Troubleshooting & Bug Fixes](#-troubleshooting--bug-fixes)
- [Viva-Voce Questions & Answers](#-viva-voce-questions--answers)
- [Conclusion](#-conclusion)

---

## 🧠 Mathematical Formulation & Theory

This laboratory experiment implements inference (forward propagation only) for a two-layer feedforward neural network:

$$\begin{aligned}
\mathbf{Layer\ 1:}\quad &\mathbf{Z}_1 = \mathbf{X} \mathbf{W}_1 + \mathbf{b}_1, \quad &&\mathbf{A}_1 = \text{ReLU}(\mathbf{Z}_1) \\
\mathbf{Layer\ 2:}\quad &\mathbf{Z}_2 = \mathbf{A}_1 \mathbf{W}_2 + \mathbf{b}_2, \quad &&\mathbf{Y} = \sigma(\mathbf{Z}_2) = \frac{1}{1 + e^{-\mathbf{Z}_2}}
\end{aligned}$$

### Custom CUDA Kernels
1. **Matrix Multiplication with Bias (`matrixMulBiasKernel`):**  
   For matrix $A$ ($M \times K$) and $B$ ($K \times N$), output $C$ has dimensions $M \times N$:
   $$C[\text{row}, \text{col}] = \sum_{k=0}^{K-1} A[\text{row}, k] \times B[k, \text{col}] + \text{bias}[\text{col}]$$
   A 2D block grid maps row and column indices. Boundary condition `if (row < M && col < N)` prevents out-of-bounds memory corruption.
2. **ReLU Activation (`reluKernel`):**  
   $$f(x) = \max(0, x)$$
3. **Sigmoid Activation (`sigmoidKernel`):**  
   $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

---

## 📐 Network Configuration & Dimensions

| Tensor | Mathematical Shape | Dimensions | Description |
| :--- | :--- | :--- | :--- |
| $\mathbf{X}$ | $\text{BATCH} \times \text{INPUT\_SIZE}$ | $256 \times 128$ | Input mini-batch |
| $\mathbf{W}_1$ | $\text{INPUT\_SIZE} \times \text{HIDDEN\_SIZE}$ | $128 \times 64$ | Hidden layer weights |
| $\mathbf{b}_1$ | $1 \times \text{HIDDEN\_SIZE}$ | $64$ elements | Hidden layer biases |
| $\mathbf{A}_1$ | $\text{BATCH} \times \text{HIDDEN\_SIZE}$ | $256 \times 64$ | Hidden layer activations |
| $\mathbf{W}_2$ | $\text{HIDDEN\_SIZE} \times \text{OUTPUT\_SIZE}$ | $64 \times 10$ | Output layer weights |
| $\mathbf{b}_2$ | $1 \times \text{OUTPUT\_SIZE}$ | $10$ elements | Output layer biases |
| $\mathbf{Y}$ | $\text{BATCH} \times \text{OUTPUT\_SIZE}$ | $256 \times 10$ | Final inference predictions |

Deterministic input values are generated using trigonometric series ($X[i] = 0.5 \cdot \sin(0.013i)$, $W_1[i] = 0.08 \cdot \cos(0.017i)$) to guarantee exact numerical reproducibility across CPU and GPU.

---

## 📂 Project Structure & Implementation Phases

The repository contains both the full executable source and the modular breakdown of the **8 lab phases** as instructed by the laboratory manual:

```text
Neural-Network-Layers-in-CUDA/
├── README.md               # Laboratory documentation & results analysis
├── build.bat               # Build script for Developer Command Prompt
├── .gitignore              # Ignores build artifacts (.exe, .obj)
└── src/
    ├── neural_network.cu   # Complete unified program (Phases 1-8)
    └── phases/             # Step-by-step modular code files
        ├── phase1_constants.cu            # Phase 1: Headers, macros, constants
        ├── phase2_gpu_kernels.cu          # Phase 2: MatMul+bias, ReLU, Sigmoid kernels
        ├── phase3_cpu_network.cu          # Phase 3: CPU reference implementation
        ├── phase4_launch_gpu_pass.cu      # Phase 4: Forward pass kernel launcher
        ├── phase5_kernel_tests.cu         # Phase 5: Individual kernel unit tests
        ├── phase6_main_init_cpu_timing.cu # Phase 6: Device detection & CPU profiling
        ├── phase7_gpu_memory_timing.cu    # Phase 7: GPU memory, warm-up & timing
        └── phase8_evaluation_finish.cu    # Phase 8: Verification & output metrics
```

---

## 🖥 Hardware & Software Specifications

* **GPU:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM / 16,311 MiB)
* **GPU Driver:** 610.88 (CUDA UMD Version: 13.3, WDDM mode)
* **CUDA Toolkit:** v13.4 (nvcc release 13.4, V13.4.92)
* **Host Compiler:** Microsoft Visual C++ MSVC v143 (`cl.exe` v14.52.37025) via VS Developer Command Prompt
* **Host Operating System:** Microsoft Windows 11 (64-bit)

---

## 📊 Laboratory Benchmark Observations

### Observation Table (Section 13)

| Measurement Parameter | Observed Value (Run 1) | Observed Value (Mean of 3 Runs) | Reference (RTX 4500 Ada) | Unit |
| :--- | :---: | :---: | :---: | :---: |
| **CPU average compute time** | `0.924320` | `0.937567` | `0.526860` | ms |
| **GPU kernel-only time** | `0.022136` | `0.026305` | `0.026517` | ms |
| **GPU end-to-end time** | `0.118755` | `0.131707` | `0.066010` | ms |
| **Kernel-only speedup** | **41.756417** | **36.053811** | `19.868913` | × |
| **End-to-end speedup** | **7.783420** | **7.142963** | `7.981518` | × |
| **Maximum absolute difference** | `0.000000` | `0.000000` | `0.000000` | — |
| **Kernel validation** | **All PASSED** | **All PASSED** | All PASSED | — |
| **CPU/GPU output validation** | **MATCH** | **MATCH** | MATCH | — |
| **GPU model** | **RTX 5060 Ti** | **RTX 5060 Ti** | RTX 4500 Ada Gen | — |

---

### Multi-Run Stability Analysis
As directed by **Section 8, Step 6** of the manual, the program was executed three consecutive times to record timing stability:

| Metric | Run 1 | Run 2 | Run 3 | Average |
| :--- | :---: | :---: | :---: | :---: |
| **CPU Average Compute Time** | 0.924320 ms | 0.862445 ms | 1.025935 ms | **0.937567 ms** |
| **GPU Kernel-Only Time** | 0.022136 ms | 0.027530 ms | 0.029248 ms | **0.026305 ms** |
| **GPU End-to-End Time** | 0.118755 ms | 0.129475 ms | 0.146890 ms | **0.131707 ms** |
| **Kernel-Only Speedup** | **41.756417×** | **31.327915×** | **35.077100×** | **36.053811×** |
| **End-to-End Speedup** | **7.783420×** | **6.661093×** | **6.984376×** | **7.142963×** |
| **Max Absolute Difference** | `0.000000` | `0.000000` | `0.000000` | **0.000000** |
| **Status** | **MATCH** | **MATCH** | **MATCH** | **MATCH** |

---

### Numerical Output Verification

Deterministic forward pass predictions matched exactly between CPU and GPU:
```text
First 10 output values (GPU):
0.498196 0.503101 0.508006 0.512911 0.517814 0.497729 0.502638 0.507548 0.512457 0.517364
```
Maximum absolute difference across all output elements: **`0.000000`** (within $1 \times 10^{-4}$ tolerance).

---

## 🔍 Performance Insights & Discussion

1. **Kernel Acceleration:**  
   The GPU kernel execution completed in **~0.022 ms – 0.029 ms**, demonstrating an acceleration factor of **31.3× to 41.8×** over single-threaded host CPU execution.
2. **PCIe Memory Transfer Overhead:**  
   End-to-end execution took **~0.119 ms – 0.147 ms**, reducing the speedup to **~6.7× – 7.8×**. Host-to-device and device-to-host PCIe data movement accounts for **~81.4% of total time**. This illustrates **Amdahl's Law**: for small networks and inference batch sizes, PCIe bus data transfer overhead is the primary scaling bottleneck.

---

## 🛠 Compilation & Execution Guide

### Requirements
* Visual Studio 2022/2026 Developer Command Prompt (providing `cl.exe`)
* NVIDIA CUDA Toolkit 12.x / 13.x (providing `nvcc.exe`)

### Commands
In the **Developer Command Prompt for VS**:
```cmd
# Compile with optimization level 2
nvcc -O2 src/neural_network.cu -o neural_network.exe

# Run the executable
neural_network.exe
```

Or execute the provided batch script:
```cmd
build.bat
```

---

## ⚙ Troubleshooting & Bug Fixes

### Error: PTX Unsupported Toolchain
```text
CUDA error at neural_network.cu:197 - the provided PTX was compiled with an unsupported toolchain.
```
* **Cause:** CUDA Toolkit (13.4) version was newer than the installed display driver's CUDA UMD runtime (13.3). nvcc compiled default virtual PTX that the driver's JIT compiler could not recognize.
* **Fix:** Compile targeting your GPU's exact native architecture (SASS) directly, bypassing PTX JIT compilation:
  ```cmd
  nvcc -O2 -arch=sm_89 src/neural_network.cu -o neural_network.exe
  ```

---

## 💡 Viva-Voce Questions & Answers

<details>
<summary><b>1. What is a CUDA kernel?</b></summary>
A function declared with the <code>__global__</code> qualifier that executes in parallel on the GPU device across many concurrent threads, invoked using execution syntax <code><<<grid, block>>></code>.
</details>

<details>
<summary><b>2. What is the difference between host and device memory?</b></summary>
Host memory is system RAM managed and accessible by the CPU. Device memory is high-bandwidth VRAM on the graphics card accessible directly by GPU cores.
</details>

<details>
<summary><b>3. What do blockIdx, blockDim, and threadIdx represent?</b></summary>
Built-in CUDA coordinate variables: <code>threadIdx</code> identifies the thread within a block, <code>blockDim</code> gives thread count per block dimension, and <code>blockIdx</code> identifies the block within the grid.
</details>

<details>
<summary><b>4. Why is a boundary check required in a CUDA kernel?</b></summary>
Grids are allocated in block-size multiples (e.g. 16×16 or 256 threads). When matrix dimensions do not evenly divide the block size, boundary conditions like <code>if (row < M && col < N)</code> prevent invalid out-of-bounds memory writes.
</details>

<details>
<summary><b>5. Why are CUDA events used for kernel timing?</b></summary>
CUDA kernel launches are asynchronous on the host CPU. CPU clocks only measure queueing latency. <code>cudaEventRecord</code> inserts markers into the GPU command stream for true microsecond-level hardware execution timing.
</details>

<details>
<summary><b>6. What is the difference between kernel-only and end-to-end timing?</b></summary>
Kernel-only timing measures pure compute time when data is already on the GPU. End-to-end timing includes memory allocation, host-to-device transfers, kernel execution, and device-to-host result transfer.
</details>

<details>
<summary><b>7. Why does GPU data transfer overhead reduce speedup?</b></summary>
PCIe bus bandwidth is much slower than GPU VRAM memory bandwidth. When computational workload is relatively small, PCIe transfer time exceeds the GPU calculation time.
</details>

<details>
<summary><b>8. Why is the GPU warmed up before timing?</b></summary>
The first CUDA call incurs one-time overheads: driver initialization, context creation, and instruction caching. Warm-up calls ensure subsequent benchmark measurements capture steady-state performance.
</details>

---

## 📝 Conclusion

Custom CUDA kernels for matrix multiplication with bias, ReLU, and sigmoid activations were implemented and validated against a CPU reference model. Numerical equivalence was verified with a maximum absolute difference of `0.000000`. Timing analysis on the **NVIDIA GeForce RTX 5060 Ti** demonstrated a **41.76× kernel-only speedup** and a **7.78× end-to-end speedup**, confirming effective GPU acceleration while illustrating the real-world impact of PCIe transfer latency.
