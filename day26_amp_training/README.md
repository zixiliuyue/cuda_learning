# Day 26: 混合精度训练概念（FP16 vs FP32）

## 学习目标

1. **实现** FP32 与 FP16 两个版本的向量加法 Kernel，理解 `half` 类型与 `__hadd` 半精度加法内建函数
2. 掌握 FP32→FP16 的数据转换 `__float2half`，并**解释** FP16 为何能减半显存与带宽需求
3. **测量**并**对比** FP32 与 FP16 Kernel 在同规模数据（约 800 万元素）下的执行耗时
4. **解释**混合精度训练三大要素：损失缩放（loss scaling）、FP32 主权重、自动混合精度（AMP）
5. 关联 PyTorch `torch.cuda.amp.autocast()` + `GradScaler`，理解框架层如何自动化混合精度

---

## 前置依赖

- **Day 02 向量加法**：两个 Kernel 都基于逐元素加法
- **Day 05 设备查询**：FP16 需要 Compute Capability ≥ 5.3
- **Day 08 占用率**：理解带宽受限型 Kernel 的性能瓶颈

---

## 核心概念

### 1. FP16 vs FP32 内存对比

```
FP32 (float)   : 1 符号位 + 8 指数位 + 23 尾数位  = 4 字节
FP16 (half)    : 1 符号位 + 5 指数位 + 10 尾数位  = 2 字节
                  └─ 范围更小、精度更低，但显存/带宽减半，Tensor Core 可加速
```

对**带宽受限**的算子（如向量加法），数据量减半通常直接带来接近 2× 的吞吐提升。

### 2. 两个对照 Kernel

FP32 版本是普通加法 [vector_add_fp32](file:///f:/code/cuda_learning/day26_amp_training/src/main.cu#L22-L26)；FP16 版本用半精度内建函数 [vector_add_fp16](file:///f:/code/cuda_learning/day26_amp_training/src/main.cu#L31-L38)：

```cpp
__global__ void vector_add_fp16(const half* a, const half* b, half* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        c[tid] = __hadd(a[tid], b[tid]);   // half 专用加法内建函数
}
```

需要包含头文件 `#include <cuda_fp16.h>`。数据规模 `ARRAY_SIZE = 1024*1024*8`（约 800 万），`BLOCK_SIZE = 256`。

### 3. 主机端精度转换

FP16 数据由 FP32 转换而来（见 [main.cu](file:///f:/code/cuda_learning/day26_amp_training/src/main.cu#L81-L87)）：

```cpp
half* h_a16 = (half*)malloc(n * sizeof(half));
for (int i = 0; i < n; i++)
    h_a16[i] = __float2half(h_a[i]);   // FP32 -> FP16
```

### 4. 混合精度训练三要素（程序末尾打印）

```
前向/反向    : 大部分算子用 FP16（快、省显存）
主权重        : 保留一份 FP32 主权重用于参数更新（保精度）
损失缩放      : loss × scale，放大梯度避免 FP16 下溢为 0，更新前再还原
```

这三者结合就是「自动混合精度（AMP）」——框架自动决定哪些算子用 FP16、哪些保持 FP32。

---

## 文件结构

```
day26_amp_training/
├── src/main.cu             # FP32 / FP16 向量加法对照 + 计时
├── python/amp_training.py  # Python 版（PyTorch AMP autocast+GradScaler，否则 NumPy 模拟 FP16/FP32）
├── CMakeLists.txt          # 构建配置
├── build.bat               # Windows 编译脚本（生成 amp_training.exe）
├── build.sh                # Linux/macOS 编译脚本
└── README.md               # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：build.bat

```powershell
cd day26_amp_training
.\build.bat
.\build\Release\amp_training.exe
```

方式二：手动 CMake

```powershell
cd day26_amp_training
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\amp_training.exe
```

### Linux / macOS

```bash
cd day26_amp_training
bash build.sh
./build/amp_training
```

### Python 版本

```bash
cd day26_amp_training/python
python amp_training.py
```

> 装了 PyTorch 且有 CUDA 时，会用 `torch.cuda.amp.autocast()` + `GradScaler` 跑一段 AMP 训练并与 FP32 训练对比耗时/loss；否则用 NumPy 模拟 FP16/FP32 两种精度的前向传播。

---

## 知识点总结

1. **`half` 类型**：2 字节半精度浮点，需 `#include <cuda_fp16.h>`，Compute Capability ≥ 5.3 支持原生运算。
2. **`__hadd` / `__float2half`**：半精度加法内建函数与 FP32↔FP16 转换函数。
3. **带宽收益**：FP16 数据量减半，对带宽受限型 Kernel 直接提升吞吐。
4. **精度代价**：FP16 尾数只有 10 位，动态范围与精度都低于 FP32，累加大量数值时误差累积明显。
5. **损失缩放**：训练中把 loss 乘一个大系数，避免小梯度在 FP16 下变成 0。
6. **FP32 主权重**：优化器保留 FP32 权重副本用于更新，保证收敛精度。
7. **BF16**：指数位与 FP32 相同（范围大）、尾数更少，训练更稳定，常见于新硬件。
8. **PyTorch AMP**：`with autocast():` 自动混合精度前向，配合 `GradScaler` 做损失缩放。

---

## 验收标准

- 程序打印设备信息与数组大小（约 800 万元素）。
- 分别打印 `FP32 向量加法` 与 `FP16 向量加法` 的耗时——理想情况下 FP16 更快（带宽减半）。
- 末尾打印五条混合精度训练要点，程序正常退出。
- Python 版打印 FP32 与 AMP/FP16 的耗时对比、加速比与最终 loss。

---

## 动手挑战

1. **验证 FP16 精度损失**：把 FP16 结果 `__half2float` 转回 FP32，与 FP32 结果逐元素比较，统计最大误差与平均误差，观察 FP16 的精度差距。
2. **半精度向量化**：用 `half2` 类型（一次处理 2 个 half）改写 FP16 Kernel，使每线程处理 2 个元素，测量带宽利用率提升。
3. **加 warmup 与多次平均**：当前只跑一次 Kernel，首次启动含 JIT/上下文开销。加入预热 + 多次循环取平均，得到更稳定的 FP16/FP32 加速比。

---

## 常见问题

- **Q：为什么我的 FP16 没比 FP32 快，甚至更慢？** A：规模太小或未预热时，Kernel 启动开销掩盖了带宽收益；且逐元素单 `half` 未充分利用向量化，试试 `half2` 与更大数据量并多次平均。
- **Q：FP16 会不会算错？** A：结果不是「错」，而是精度更低。对加法这种简单操作误差很小，但深度网络中需靠损失缩放 + FP32 主权重维持收敛。
- **Q：编译报找不到 `half`？** A：确认包含了 `<cuda_fp16.h>`，且目标架构支持半精度（CC ≥ 5.3）。

---

## 延伸阅读

- [NVIDIA 混合精度训练指南（Training with Mixed Precision）](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [PyTorch Automatic Mixed Precision (AMP) 官方文档](https://pytorch.org/docs/stable/amp.html)
