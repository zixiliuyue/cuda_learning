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
