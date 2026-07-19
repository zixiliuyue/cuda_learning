# Day 29: 简易深度学习推理引擎

> 🏗️ **综合大项目 · 建议 2-3 天完成**
> 本项目手写 CUDA 算子（Linear+ReLU、Softmax）串起一条完整的前向推理链路，是对 Day 24
> 的深化。建议按「动手挑战」的里程碑推进：**MVP（跑通两层网络）→ 优化（算子融合 + 内存池 + batch）
> → 可视化（端到端流程 + 吞吐/延迟基准）**。

## 学习目标

1. **实现**手写 CUDA 算子库的核心成员：`linear_relu_kernel`（全连接+ReLU 融合）与 `softmax_kernel`
2. **实现**一条完整推理链路 `Linear(784→128) → ReLU → Linear(128→10) → Softmax`，并串联 Kernel 调用
3. **解释**推理引擎的四大组成：算子库、内存池、图优化（算子融合/常量折叠）、精度选择
4. **测量**整条推理链路的 GPU 执行耗时，验证 Softmax 输出为合法概率分布（和为 1）
5. **对比** CUDA 手写引擎与 Python 版「预处理 → 推理 → 后处理」三段式流程，理解端到端延迟与吞吐

---

## 前置依赖

- **Day 24 TensorRT 概念**：本项目是其 FC/ReLU/Softmax 算子的延续与整合
- **Day 06 Tiled 矩阵乘法**：全连接层的核心是矩阵乘
- **Day 09 归约**：Softmax 的 max/sum 归约
- **Day 21 PyTorch 扩展**：理解推理引擎如何服务上层模型

---

## 核心概念

### 1. 推理引擎 = 算子库 + 调度

一个推理引擎的本质是「把网络拆成算子，按图顺序在 GPU 上依次/融合执行」。本例的网络（见 [main.cu](file:///f:/code/cuda_learning/day29_inference_engine/src/main.cu#L61-L62)）：

```
input[784]
   │  Linear1: W1[784×128] + b1   （linear_relu_kernel，含 ReLU）
   ▼
hidden[128]
   │  Linear2: W2[128×10] + b2    （linear_relu_kernel，含 ReLU）
   ▼
logits[10]  ── softmax_kernel ──►  probs[10]
```

参数：`INPUT_SIZE=784`、`HIDDEN_SIZE=128`、`OUTPUT_SIZE=10`、`BLOCK_SIZE=256`。

### 2. Linear + ReLU 融合算子

[linear_relu_kernel](file:///f:/code/cuda_learning/day29_inference_engine/src/main.cu#L21-L35) 每线程算一个输出神经元，末尾 `fmaxf(sum, 0.0f)` 就地完成 ReLU——这是「算子融合」：

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < out_size) {
    float sum = bias[tid];
    for (int i = 0; i < in_size; i++)
        sum += input[i] * weights[i * out_size + tid];
    output[tid] = fmaxf(sum, 0.0f);   // ReLU 与 Linear 融合，省一次显存往返
}
```

### 3. Softmax 输出层

[softmax_kernel](file:///f:/code/cuda_learning/day29_inference_engine/src/main.cu#L37-L51)（单线程教学版）：先求 max 做数值稳定，再 `expf(x-max)` 归一化。

### 4. Kernel 串联调度

主流程按层顺序启动 Kernel，中间结果 `d_hidden` 直接留在显存中传给下一层（见 [main.cu 推理执行](file:///f:/code/cuda_learning/day29_inference_engine/src/main.cu#L98-L109)）：

```cpp
linear_relu_kernel<<<(HIDDEN_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_w1, d_b1, d_hidden, INPUT_SIZE, HIDDEN_SIZE);
linear_relu_kernel<<<(OUTPUT_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_hidden, d_w2, d_b2, d_output, HIDDEN_SIZE, OUTPUT_SIZE);
softmax_kernel<<<1,1>>>(d_output, d_output, OUTPUT_SIZE);
cudaDeviceSynchronize();
```

---

## 文件结构

```
day29_inference_engine/
├── src/main.cu               # linear_relu / softmax 算子 + 两层网络推理
├── python/inference_engine.py # Python 版：预处理→推理(PyTorch/NumPy)→后处理 完整流程 + 吞吐统计
├── CMakeLists.txt            # 构建配置
├── build.bat                 # Windows 编译脚本（生成 inference_engine.exe）
├── build.sh                  # Linux/macOS 编译脚本
└── README.md                 # 本文件
```

> 说明：CUDA 版是两层全连接分类器（`784→128→10`）；Python 版是三层网络（`784→256→128→10`）并演示完整的预处理/后处理与端到端吞吐统计，二者互补。

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：build.bat

```powershell
cd day29_inference_engine
.\build.bat
.\build\Release\inference_engine.exe
```

方式二：手动 CMake

```powershell
cd day29_inference_engine
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\inference_engine.exe
```

### Linux / macOS

```bash
cd day29_inference_engine
bash build.sh
./build/inference_engine
```

### Python 版本

```bash
cd day29_inference_engine/python
python inference_engine.py
```

> 装了 PyTorch 时用 `SimpleModel`（3 层 FC）在 GPU/CPU 上推理并计时；否则用 NumPy 模拟。会分别统计预处理、推理、后处理耗时与端到端吞吐（samples/s）。

---

## 知识点总结

1. **推理引擎四要素**：算子库、内存池（复用中间结果显存）、图优化（算子融合/常量折叠）、精度选择（FP32/FP16/INT8）。
2. **算子融合**：`Linear+ReLU` 合成一个 Kernel，减少中间结果的显存读写与 Kernel 启动开销。
3. **每线程一输出**：全连接层最直观的并行映射，累加 `in_size` 个乘积。
4. **中间结果驻留显存**：层与层之间的张量（`d_hidden`）不回主机，直接在显存传递。
5. **Softmax 数值稳定**：先减最大值再取指数，防止 `expf` 溢出。
6. **端到端延迟**（Python 版）= 预处理 + 推理 + 后处理，吞吐 = batch / 端到端时间。
7. **真实引擎**：TensorRT / ONNX Runtime / OpenVINO 会做量化、蒸馏、剪枝、层融合等进一步优化。

---

## 验收标准

- 程序打印网络结构 `Linear(784->128) -> ReLU -> Linear(128->10) -> Softmax`。
- 打印 `推理引擎执行` 耗时（毫秒级）。
- 打印 10 个输出概率，**每个 ∈ [0,1] 且总和约为 1.0**。
- 末尾打印推理引擎架构四要点，程序正常退出。
- Python 版打印预处理/推理/后处理各阶段耗时、端到端延迟、吞吐（samples/s）与前 5 个样本的预测类别+置信度。

---

## 动手挑战（分阶段里程碑）

### 里程碑 1 — MVP：正确性验证

- 用固定权重（如小随机种子固定）在 CPU 上手写同样的两层前向 + softmax，与 GPU 结果逐元素比较，确认最大误差在容差内（如 `1e-4`），证明算子实现正确。

### 里程碑 2 — 优化：内存池 + batch + 并行 Softmax

- **内存池**：把每层的 `cudaMalloc` 改为程序启动时一次性预分配、层间复用，避免反复分配。
- **batch 推理**：把 `linear_relu_kernel` 扩展为一次处理 `BATCH_SIZE` 个样本（二维索引），对比单样本与批量的吞吐。
- **并行 Softmax**：把单线程 `softmax_kernel` 改为一个 block + 共享内存归约求 max/sum（参考 Day 09）。

### 里程碑 3 — 可视化：端到端基准

- 参考 Python 版，为 CUDA 引擎补上简单的预处理（归一化）和后处理（argmax 取预测类别），打印端到端延迟与吞吐，画一张「batch size vs 吞吐」曲线。
- 进阶：加一个 `Conv2D` 或 `MaxPool` 算子，把网络扩展成小型 CNN。

---

## 常见问题

- **Q：为什么中间结果不拷回主机？** A：层与层之间的张量只在 GPU 上流转，回主机会引入大量 H2D/D2H 传输，是推理性能的大忌；只有最终输出才拷回。
- **Q：单线程 Softmax 会是瓶颈吗？** A：输出只有 10 维时影响很小；但输出维度大（如分类数上千）时应改为并行归约。
- **Q：手写引擎能替代 TensorRT 吗？** A：不能。本项目用于理解底层原理；生产环境应使用经过高度调优、支持量化与自动融合的成熟引擎。

---

## 延伸阅读

- [ONNX Runtime 架构与执行提供者（Execution Providers）](https://onnxruntime.ai/docs/execution-providers/)
- [NVIDIA cuDNN 开发者指南（深度学习算子库）](https://docs.nvidia.com/deeplearning/cudnn/latest/index.html)
