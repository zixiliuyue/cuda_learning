# Day 24: TensorRT 推理优化概念

## 学习目标

1. 理解 **TensorRT** 在推理链路中的定位，以及它相对手写 Kernel 的价值（层融合、精度校准、Kernel 自动调优）
2. 用手写 CUDA Kernel **实现**一个最小推理网络：`FC(784→256) → ReLU → FC(256→10) → Softmax`，对应 TensorRT 底层算子的原理
3. 掌握 `fc_relu_kernel` 中「每线程负责一个输出神经元」的并行映射方式，并**解释**其访存特征
4. **测量**整条推理链路的 GPU 执行耗时，并观察 Softmax 输出为合法概率分布（各元素 ∈ [0,1]、和为 1）
5. **对比** CUDA 手写实现与 Python 端 TensorRT / NumPy 推理流程的差异

> 注意：本项目是**概念演示版**。完整 TensorRT 需要安装 NVIDIA TensorRT SDK 并从 ONNX/engine 文件构建引擎；
> 这里用手写 Kernel 讲清 FC/ReLU/Softmax 这些底层算子如何在 GPU 上执行。

---

## 前置依赖

- **Day 02 向量加法**：线程索引 `blockIdx.x * blockDim.x + threadIdx.x` 与执行配置
- **Day 06 Tiled 矩阵乘法**：全连接层本质是矩阵-向量乘法
- **Day 09 归约**：Softmax 需要求最大值与求和
- **Day 21 PyTorch 扩展 / Day 22 Numba**：理解算子如何服务上层框架

---

## 核心概念

### 1. 推理网络结构

本例是一个针对 28×28（`INPUT_SIZE=784`）输入的两层全连接分类器：

```
input[784]
   │  FC1: W1[784×256] + b1[256]
   ▼
hidden[256]  ── ReLU ──►  (fmaxf(sum, 0))
   │  FC2: W2[256×10] + b2[10]
   ▼
logits[10]  ── Softmax ──►  probs[10]（和为 1）
```

关键宏定义（见 [main.cu](file:///f:/code/cuda_learning/day24_tensorrt/src/main.cu#L17-L20)）：
`BATCH_SIZE=32`、`INPUT_SIZE=784`、`HIDDEN_SIZE=256`、`OUTPUT_SIZE=10`。

### 2. FC + ReLU 融合 Kernel

[fc_relu_kernel](file:///f:/code/cuda_learning/day24_tensorrt/src/main.cu#L25-L39) 把「全连接 + 偏置 + ReLU 激活」融合进一个 Kernel——这正是 TensorRT「层融合」的思想：一次 Kernel 启动完成多步运算，避免中间结果反复读写显存。

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;   // 每线程算一个输出神经元
if (tid < out_size) {
    float sum = bias[tid];
    for (int i = 0; i < in_size; i++)
        sum += input[i] * weights[i * out_size + tid];   // 权重按列访问
    output[tid] = fmaxf(sum, 0.0f);                       // ReLU 融合在末尾
}
```

- **并行映射**：每个线程负责一个输出维度，累加 `in_size` 个乘积。
- **访存特征**：`input[i]` 被所有线程广播读取；`weights[i*out_size+tid]` 相邻线程访问相邻地址，是合并访问。

### 3. Softmax 输出层

[softmax_output_kernel](file:///f:/code/cuda_learning/day24_tensorrt/src/main.cu#L44-L64) 是教学简化版（单线程 `<<<1,1>>>`）：先求 `max_val` 做数值稳定，再 `expf(x - max)`，最后除以总和归一化。

```
p_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

减去 `max` 是为了防止 `expf` 溢出——这是 softmax 的标准数值稳定技巧。

---

## 文件结构

```
day24_tensorrt/
├── src/main.cu               # FC+ReLU / Softmax 手写 Kernel + 推理主流程
├── python/tensorrt_demo.py   # Python 版（trt 可用则演示，否则 NumPy 模拟 Conv→ReLU→FC）
├── CMakeLists.txt            # 构建配置
├── build.bat                 # Windows 编译脚本（生成 tensorrt_demo.exe）
├── build.sh                  # Linux/macOS 编译脚本
└── README.md                 # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：build.bat（内部调用 CMake + MSVC）

```powershell
cd day24_tensorrt
.\build.bat
.\build\Release\tensorrt_demo.exe
```

方式二：手动 CMake

```powershell
cd day24_tensorrt
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\tensorrt_demo.exe
```

### Linux / macOS

```bash
cd day24_tensorrt
bash build.sh
./build/tensorrt_demo
```

### Python 版本

```bash
cd day24_tensorrt/python
python tensorrt_demo.py
```

> 若安装了 `tensorrt`，会打印 TensorRT 版本并说明工作流；否则自动 fallback 到 NumPy
> 模拟 `Conv2D → ReLU → Conv2D → ReLU → Flatten → Linear` 的推理流程并计时。

---

## 知识点总结

1. **TensorRT 定位**：把训练好的模型（ONNX/框架导出）编译为高度优化的推理引擎，降低延迟、提升吞吐。
2. **五大优化技术**（程序末尾也会打印）：层融合、精度校准（FP32→FP16→INT8）、Kernel 自动调优、动态张量内存、多流执行。
3. **全连接层 = 矩阵乘**：每个输出神经元是输入向量与一列权重的点积。
4. **层融合原理**：`fc_relu_kernel` 把 FC 与 ReLU 合并，减少中间结果的显存往返。
5. **每线程一输出**的映射简单直观，但对大矩阵不是最优（未用共享内存/Tensor Core）。
6. **Softmax 数值稳定**：先减最大值再取指数，避免溢出。
7. **概念 vs 工业级**：手写 Kernel 帮助理解底层；真实部署应使用 TensorRT/cuBLAS/cuDNN 等高度调优实现。

---

## 验收标准

- 程序打印设备信息与网络结构 `FC(784->256) -> ReLU -> FC(256->10) -> Softmax`。
- 打印 `推理执行` 耗时（毫秒级，通常远小于 1 ms）。
- 输出 10 个概率值，**每个都在 [0,1] 区间且总和约等于 1.0**（可手动累加验证）。
- 末尾打印五条 TensorRT 优化技术，程序正常退出。
- Python 版能打印推理延迟、输出形状 `(8, 10)` 与 TensorRT 是否可用。

---

## 动手挑战

1. **支持 batch 推理**：当前 `fc_relu_kernel` 只处理单个样本（`input[i]`）。改造为一次处理 `BATCH_SIZE=32` 个样本（二维索引 `[batch, out]`），并对比总耗时。
2. **改进 Softmax 并行度**：现在的 `softmax_output_kernel` 是单线程串行。改成一个 block 内多线程 + 共享内存归约（参考 Day 09）求 max 与 sum。
3. **加一层真实数据验证**：用固定权重（如全 1）和已知输入，手工推导期望输出，验证 Kernel 结果与 CPU 计算一致（容差 1e-4）。

---

## 常见问题

- **Q：为什么手写 Kernel 不一定比 TensorRT 快？** A：TensorRT 会自动选择最优 Kernel、融合层、使用 Tensor Core 并做精度量化；本例的朴素 FC Kernel 未做任何这些优化，仅用于讲原理。
- **Q：Softmax 为什么要减去最大值？** A：`expf` 对较大的输入会溢出成 `inf`，减去 `max(x)` 后最大指数项变成 `exp(0)=1`，结果不变但数值稳定。
- **Q：可执行文件名为什么是 `tensorrt_demo.exe`？** A：由 CMake 目标名决定，`build.bat` 末尾也提示了运行路径 `.\build\Release\tensorrt_demo.exe`。

---

## 延伸阅读

- [TensorRT 官方文档（Developer Guide）](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)
- [TensorRT 层融合与精度优化最佳实践](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
