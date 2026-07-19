# Day 1: 环境搭建与第一个 CUDA 程序

## 学习目标

1. 理解 CUDA 程序的基本结构（主机代码 + 设备代码）
2. 掌握 Kernel 函数的语法 `<<<grid, block>>>`
3. 理解线程索引的含义：`threadIdx`, `blockIdx`, `blockDim`
4. 成功编译并运行第一个 GPU 程序
5. 熟悉 `nvcc` 编译流程和 CMake 构建系统

## 核心概念

### CUDA 程序结构

CUDA 程序由两部分组成：
- **主机代码 (Host Code)**: 在 CPU 上执行，负责数据准备、Kernel 启动、结果收集
- **设备代码 (Device Code)**: 在 GPU 上执行，即 Kernel 函数

### Kernel 启动配置

```cpp
kernel_name<<<grid_dim, block_dim>>>(arguments);
```

- `grid_dim`: Grid 中包含多少个 Block
- `block_dim`: 每个 Block 中包含多少个 Thread
- **总线程数** = `grid_dim.x * block_dim.x`（一维情况）

### 线程索引系统

| 变量 | 含义 | 范围 |
|------|------|------|
| `threadIdx.x` | 线程在 Block 内的索引 | `[0, blockDim.x - 1]` |
| `blockIdx.x` | Block 在 Grid 内的索引 | `[0, gridDim.x - 1]` |
| `blockDim.x` | Block 包含的线程数 | 启动时指定 |
| `gridDim.x` | Grid 包含的 Block 数 | 启动时指定 |

**全局线程 ID** = `blockIdx.x * blockDim.x + threadIdx.x`

## 文件说明

| 文件 | 说明 |
|------|------|
| `src/main.cu` | C++/CUDA 主程序 |
| `python/hello_cuda.py` | Python/Numba 等价实现 |
| `CMakeLists.txt` | CMake 构建配置 |
| `build.bat` | Windows 一键编译 |
| `build.sh` | macOS/Linux 一键编译 |

## 编译与运行

### Windows (有 NVIDIA GPU)

```powershell
cd day01_hello_cuda
.\build.bat
.\build\Debug\hello_cuda.exe
```

### macOS / 无 GPU 环境

```bash
cd day01_hello_cuda/python
python hello_cuda.py
```

### Google Colab

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 上传 `python/hello_cuda.py`
3. 在代码单元格中运行：`!python hello_cuda.py`

## 预期输出

```
=============================================================
     Day 1: Hello CUDA — 我的第一个 GPU 程序
=============================================================

[平台信息] 操作系统: Windows | 编译器: MSVC | 架构: x86_64

=================================================================
                     CUDA 设备信息摘要
=================================================================
  设备编号       : 0 / 0
  设备名称       : NVIDIA GeForce RTX 5060 Ti
  计算能力       : 12.0
  ...
=================================================================

[主机] 准备启动 Kernel ...
[主机] Grid 维度 : 4 个 Block
[主机] Block 维度: 8 个 Thread
[主机] 总线程数  : 32 个

[主机] 启动 Kernel <<< 4, 8 >>> ...
------------------------------------------------------------
  [GPU线程] 全局ID=00 | Block=00 | Thread=00 | 你好，CUDA!
  [GPU线程] 全局ID=01 | Block=00 | Thread=01 | 你好，CUDA!
  ...（32行，顺序不确定）...
------------------------------------------------------------
[主机] Kernel 执行完成！
```

## 练习题

1. 修改 `num_blocks` 和 `threads_per_block`，观察输出变化
2. 尝试使用二维 Grid（`dim3 grid(2, 2)`），观察 `blockIdx.y` 的变化
3. 为什么在 Kernel 中使用 `printf` 后需要调用 `cudaDeviceSynchronize()`？

## 常见问题

**Q: 编译报错 "fatal error: cuda_runtime.h: No such file"?**

A: CUDA Toolkit 未安装或未正确配置。检查：
- Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include` 是否存在
- 环境变量 `CUDA_PATH` 是否设置

**Q: 运行时显示 "未检测到 NVIDIA GPU 设备"?**

A: 可能原因：
1. 显卡驱动未安装或版本过旧
2. 使用的是集成显卡而非独立显卡
3. 在远程服务器上运行，需要通过 `nvidia-smi` 检查 GPU 状态

## 进阶阅读

- [NVIDIA CUDA C++ Programming Guide — Program Structure](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#program-structure)
- [CUDA Execution Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-model)

---

## 验收标准

达到以下几点，说明你已掌握本天内容：

1. 程序成功编译运行，先打印出「平台信息」和「CUDA 设备信息摘要」（包含设备名称、计算能力等），说明 GPU 环境配置正确。
2. Kernel 以 `<<< 4, 8 >>>` 启动，屏幕上出现 **32 行** `[GPU线程] 全局ID=xx | Block=xx | Thread=xx | 你好，CUDA!` 输出，即 4 个 Block × 8 个 Thread。
3. 能解释为什么这 32 行的打印顺序每次运行可能不同（线程并行、`printf` 输出顺序不确定），并理解 `全局ID = blockIdx.x * blockDim.x + threadIdx.x` 的计算。
4. 理解若删除 `cudaDeviceSynchronize()`，主机端可能看不到完整的 Kernel 输出。

## 动手挑战

基于 [main.cu](src/main.cu) 中的 `hello_cuda_kernel` 循序渐进地练习：

1. **改变线程规模**：修改 `num_blocks` 和 `threads_per_block`（例如 2×16、8×4），重新运行，验证输出总行数始终等于两者乘积，并观察全局 ID 的连续性。
2. **升级到二维 Grid**：将 `grid_dim` 改为 `dim3 grid_dim(2, 2)`，在 Kernel 中新增打印 `blockIdx.y`，观察二维 Block 索引的变化，理解 `gridDim.y` 的含义。
3. **增加线程计算**：让每个线程在打印前计算并输出 `全局ID 的平方` 或所属 Warp 编号（`global_tid / 32`），体会「每个线程独立执行同一份代码」的 SIMT 模型。
