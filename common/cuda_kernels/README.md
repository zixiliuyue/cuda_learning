# 高频基础 Kernel 库（common/cuda_kernels）

本目录收录 LLM / 深度学习中最常用的几个 CUDA 基础算子，供教学与快速复用。
所有 Kernel 均为 **header-only**（`.cuh`），任意 `.cu` 文件 `#include` 后即可直接调用，
无需单独编译。每个头文件都提供一个 `launch_xxx(...)` 主机端封装，一行即可启动。

## 目录结构

| 文件 | 说明 |
|------|------|
| `softmax_kernel.cuh`     | Softmax 前向（warp 归约 + 数值稳定） |
| `layer_norm_kernel.cuh`  | Layer Normalization 前向（含 gamma/beta） |
| `gelu_kernel.cuh`        | GELU 激活（tanh 精确版 + sigmoid 近似版） |
| `dropout_kernel.cuh`     | Dropout（cuRAND 随机掩码，训练/推理双模式） |
| `demo.cu`                | 独立演示程序，验证所有 Kernel 并输出性能表 |
| `CMakeLists.txt`         | 编译 demo 的 CMake 配置 |
| `README.md`              | 本说明文件 |

---

## 1. Softmax

### 数学公式

对每一行向量 `x`（长度 `cols`），第 `i` 个输出：

```
m            = max_j(x_j)                         # 数值稳定：先取最大值
softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m)    # 减 max 后再指数
```

减去每行最大值 `m` 后，指数参数恒 `<= 0`，避免 `exp` 上溢；由于分子分母同乘
`exp(-m)`，结果与原始定义完全等价。

### 并行策略
- 每一行由**一个 Block** 处理；
- Block 内先用 `__shfl_down_sync` 做 warp 内归约，再用共享内存跨 warp 归约，
  两趟归约分别求 **max** 和 **Σexp**；
- 支持一维（`rows=1`）与二维（`[rows, cols]`，逐行 softmax，常见于 attention 分数）。

### 调用方式

```cpp
#include "softmax_kernel.cuh"

// d_in / d_out 形状均为 [rows, cols]（行优先）
launch_softmax(d_in, d_out, rows, cols);          // block_size 默认 256
launch_softmax(d_in, d_out, rows, cols, 128, stream);
```

---

## 2. Layer Normalization

### 数学公式

对每一行向量 `x`（长度 `D = cols`）：

```
mu      = (1/D) Σ_i x_i
var     = (1/D) Σ_i (x_i - mu)^2 = E[x^2] - E[x]^2
x_hat_i = (x_i - mu) / sqrt(var + eps)
y_i     = gamma_i * x_hat_i + beta_i
```

- `eps`：防止方差为 0 时除零（典型 `1e-5`）；
- `gamma` / `beta`：可学习的缩放与偏置参数（长度 `D`），传 `nullptr` 时分别退化为 1 与 0。

### 并行策略
- 每行一个 Block；
- **一次遍历同时累加 `Σx` 与 `Σx²`**，再由 `mu = Σx/D`、`var = Σx²/D - mu²` 反推，
  避免两遍读取数据；
- warp 归约 + 共享内存归约（共享内存需 `2 * nWarp` 个 float）。

### 调用方式

```cpp
#include "layer_norm_kernel.cuh"

launch_layer_norm(d_in, d_out, d_gamma, d_beta, rows, cols, /*eps=*/1e-5f);
// gamma/beta 可传 nullptr 表示不缩放/不偏置
launch_layer_norm(d_in, d_out, nullptr, nullptr, rows, cols);
```

> LayerNorm 在特征维度归一化，与 batch 大小无关，比 BatchNorm 更适合变长序列的 Transformer。

---

## 3. GELU 激活

### 两种实现对比

| 版本 | 公式 | 特点 |
|------|------|------|
| tanh 精确版 | `0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))` | 与真实 erf 版几乎无差，GPT/BERT 使用 |
| sigmoid 近似版 | `x * sigmoid(1.702 * x)` | 计算最快，尾部略有偏差 |

两者是**不同的近似**，输出并不完全相同（demo 中会打印两者最大差异）。
精度敏感场景用 tanh 版；追求推理速度且精度容忍度高时可用 sigmoid 版。

### 并行策略
逐元素（element-wise）算子，采用 grid-stride 循环，兼容任意规模。

### 调用方式

```cpp
#include "gelu_kernel.cuh"

launch_gelu(d_in, d_out, n, /*use_tanh=*/true);   // 精确 tanh 版
launch_gelu(d_in, d_out, n, /*use_tanh=*/false);  // 近似 sigmoid 版
```

---

## 4. Dropout

### 数学公式（Inverted Dropout）

```
训练模式：
  keep_prob = 1 - p
  scale     = 1 / keep_prob
  mask_i    = (rand_i > p) ? 1 : 0
  y_i       = x_i * mask_i * scale

推理模式：
  y_i = x_i          # 恒等映射
```

训练时对保留元素乘 `1/keep_prob` 做补偿，使输出**期望不变**，因此推理时无需再缩放。

### 随机数与种子管理
- 使用 cuRAND 设备端 API（`curand_kernel.h`，header-only）；
- `curand_init(seed, subsequence, offset, &state)`：
  - `subsequence` 用**全局线程 ID**，保证线程间随机序列互不相关；
  - 上层每步传入**递增的 seed**（或固定 seed + 递增 offset）保证每次迭代掩码不同；
  - 固定 `seed` 即可得到**可复现**的掩码（便于调试 / 复现实验）。
- 反向传播需复用同一份 `mask`，故训练模式支持输出 `mask` 数组。

### 调用方式

```cpp
#include "dropout_kernel.cuh"

// 训练：输出随机掩码到 d_mask（反向传播复用）
launch_dropout(d_in, d_out, d_mask, n, /*drop_prob=*/0.3f, /*training=*/true, /*seed=*/step);

// 推理：恒等映射，mask 传 nullptr
launch_dropout(d_in, d_out, nullptr, n, 0.3f, /*training=*/false);
```

---

## 编译与运行

### Windows (有 NVIDIA GPU)

```powershell
cd common/cuda_kernels
cmake -B build -S .
cmake --build build --config Release
.\build\Release\cuda_kernels_demo.exe
```

### macOS / Linux

```bash
cd common/cuda_kernels
cmake -B build -S .
cmake --build build
./build/cuda_kernels_demo
```

> 需要 CUDA Toolkit（兼容 CUDA 11 / 12 / 13）。CMake 会根据 `nvcc` 版本自动选择合适的
> `CMAKE_CUDA_ARCHITECTURES`。设备信息查询使用 `cudaDeviceGetAttribute`，兼容 CUDA 13。

### 预期输出（片段）

```
============================================================
                      性能与验证总结
============================================================
  算子           | 规模               |  GPU耗时(ms) | 验证
  ------------------------------------------------------------
  Softmax        | 512 x 1024         |       0.0xxx | PASS
  LayerNorm      | 512 x 1024         |       0.0xxx | PASS
  GELU(tanh)     | 1048576            |       0.0xxx | PASS
  Dropout        | 1048576            |       0.0xxx | PASS
============================================================
```

---

## 常见优化技巧

1. **数值稳定性**：Softmax 先减 max、LayerNorm 用 `E[x²]-E[x]²`，
   避免上溢 / 除零，是训练能否收敛的关键细节。
2. **warp shuffle 归约**：`__shfl_down_sync` 无需共享内存与 `__syncthreads`，
   warp 内交换寄存器数据，比纯共享内存归约更快，是 Block 内归约的首选。
3. **两级归约**：Block 内 warp 归约 -> 共享内存汇总 -> 首个 warp 再归约，
   兼顾大 Block 与低同步开销。
4. **单趟统计**：LayerNorm 一次遍历同时累加 `Σx` 与 `Σx²`，
   把两遍全局内存读取压缩为一遍，显著提升带宽利用率。
5. **grid-stride 循环**：逐元素算子（GELU / Dropout）用固定 grid + stride 遍历，
   既能覆盖任意规模，又便于复用启动配置。
6. **除法转乘法**：用 `1/sum`、`rsqrtf` 的倒数乘法替代逐元素除法，减少慢速指令。
7. **`__restrict__` 指针**：告知编译器输入输出无别名，利于寄存器缓存与向量化。
8. **掩码复用**：Dropout 训练必须保存 `mask` 供反向使用，前后向共用同一份随机结果。
