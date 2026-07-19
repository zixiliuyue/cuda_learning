# Day 30: 阶段总结与进阶路线图

> 🎉 恭喜！你已经完成 CUDA 学习项目的前 30 天。本篇不讲新的 Kernel，而是**回顾**已掌握的
> 知识版图、跑一个**综合基准**验证环境，并给出下一阶段（Day 31+）的**进阶路线**与自检清单。

## 学习目标

1. **回顾**前 30 天的五大阶段主线，梳理自己已掌握与仍薄弱的知识点
2. **运行**综合基准 Kernel（FMA `c = a*b + a`），**测量**有效显存带宽（GB/s）
3. **对比** Python 版中向量加/乘、矩阵乘的 CPU 与 GPU 加速比，理解带宽/计算受限的差异
4. **完成**下方自检清单，确认能独立解释每天的核心概念
5. **规划**进阶方向（底层优化 / 深度学习加速 / 高级话题），衔接 Day 31 之后的内容

---

## 前置依赖

- **Day 01 - Day 29 全部内容**：本篇是对整个第一阶段（1-30 天）的收束

---

## 30 天路线图回顾

程序 [main.cu](file:///f:/code/cuda_learning/day30_summary/src/main.cu#L45-L83) 会完整打印下面五个阶段：

| 阶段 | 天数 | 主题 | 核心能力 |
|------|------|------|----------|
| 一、基础与核心概念 | Day 01-05 | Hello CUDA、向量/矩阵加、转置、设备查询 | 线程层次、执行模型、内存传输、错误处理 |
| 二、内存优化与高级特性 | Day 06-12 | Tiled matmul、合并访问、占用率、归约、流、常量/纹理内存、动态并行 | 共享内存、访存优化、异步执行 |
| 三、算法实战与优化 | Day 13-20 | cuBLAS、cuFFT、图像滤波、排序、Scan、BFS、蒙特卡洛、调优 | 常用库 + 并行算法 + 性能分析 |
| 四、主流框架集成 | Day 21-26 | PyTorch 扩展、Numba、OpenCV CUDA、TensorRT、多 GPU、混合精度 | 与生态框架协作 |
| 五、综合实战 | Day 27-30 | N-Body、光线追踪、推理引擎、总结 | 端到端项目整合 |

---

## 核心内容：综合基准与带宽测量

### 1. FMA 基准 Kernel

[final_benchmark_kernel](file:///f:/code/cuda_learning/day30_summary/src/main.cu#L19-L26) 对每个元素做一次乘加（FMA）：

```cpp
c[tid] = a[tid] * b[tid] + a[tid];   // 融合乘加 (Fused Multiply-Add)
```

数据规模 `TEST_SIZE = 1024*1024`（约 100 万），`BLOCK_SIZE = 256`。

### 2. 有效带宽计算

程序读 2 个数组、写 1 个数组，共 3 份数据传输，据此估算带宽（见 [main.cu](file:///f:/code/cuda_learning/day30_summary/src/main.cu#L117-L119)）：

```
有效带宽 (GB/s) = (3 × bytes) / (1024³) / (耗时秒)
```

这是判断一个**带宽受限型 Kernel** 是否接近硬件峰值的关键指标——把它与 Day 05 查询到的理论带宽对比，就能知道优化空间还有多大。

### 3. Python 版加速比对比

[summary.py](file:///f:/code/cuda_learning/day30_summary/python/summary.py) 会跑向量加法、向量乘法、256×256 矩阵乘法的 CPU（NumPy）vs GPU（Numba）对比，打印加速比表格，直观感受不同运算的 GPU 收益。

---

## 文件结构

```
day30_summary/
├── src/main.cu       # 路线图回顾打印 + FMA 综合基准 + 带宽测量 + 进阶路线
├── python/summary.py # 30 天课程表 + 向量/矩阵运算 CPU vs GPU 基准对比
├── CMakeLists.txt    # 构建配置
├── build.bat         # Windows 编译脚本（生成 summary.exe）
├── build.sh          # Linux/macOS 编译脚本
└── README.md         # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：build.bat

```powershell
cd day30_summary
.\build.bat
.\build\Release\summary.exe
```

方式二：手动 CMake

```powershell
cd day30_summary
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\summary.exe
```

### Linux / macOS

```bash
cd day30_summary
bash build.sh
./build/summary
```

### Python 版本

```bash
cd day30_summary/python
python summary.py
```

---

## 自检清单（能独立讲清则算掌握）

用下面的问题检验自己是否真正吃透了前 30 天：

1. **执行模型**：能说清 grid / block / thread / warp 的层次，以及 `blockIdx*blockDim+threadIdx` 的索引推导吗？（Day 01-03）
2. **内存层级**：全局、共享、常量、纹理、寄存器内存各自的特点与适用场景？（Day 06/11）
3. **合并访问**：什么样的访存是合并的？为什么转置容易触发非合并访问？（Day 04/07）
4. **归约**：并行归约如何避免 warp divergence？共享内存归约的 stride 折半模式？（Day 09）
5. **流与异步**：如何用多流重叠计算与传输？为什么计时前要 `cudaDeviceSynchronize`？（Day 10）
6. **库的使用**：cuBLAS/cuFFT 的列主序、句柄管理、与手写 Kernel 的取舍？（Day 13/14）
7. **框架集成**：PyTorch 自定义算子、Numba `@cuda.jit`、混合精度 AMP 的关键 API？（Day 21/22/26）
8. **综合项目**：N-Body 的 float4 打包与 rsqrtf、光追的每像素并行、推理引擎的算子融合，能各讲一条优化吗？（Day 27-29）

---

## 验收标准

- 程序完整打印五阶段路线图（Day 01-30）。
- 打印 `综合基准 (FMA)` 耗时与 `有效带宽: X.XX GB/s`——带宽应达到硬件理论峰值的一个合理比例。
- 打印进阶路线图三大方向，并显示「恭喜完成」横幅、正常退出。
- Python 版打印 30 天课程表与向量/矩阵运算的 CPU vs GPU 加速比表格。
- 你能对照自检清单，逐条口头解释而不看答案。

---

## 动手挑战（自检性质）

1. **绘制个人带宽曲线**：修改 `TEST_SIZE`（如 2^16 到 2^26 逐步翻倍），记录每个规模下的有效带宽，画成曲线，找到带宽饱和点并解释为什么小数据量达不到峰值带宽。
2. **回顾式重构**：从前 30 天里挑一个你当时没完全理解的项目（如 Day 09 归约或 Day 17 Scan），不看原代码重写一遍 Kernel，再与原实现对比。
3. **规划下一阶段**：根据自检清单里最薄弱的 2-3 项，列出对应的 Day 31+ 学习计划（如 Day 31 Flash Attention、Day 35 Nsight Profiling、Day 39 NCCL）。

---

## 进阶学习路线图

程序末尾打印的三大方向：

1. **CUDA 底层优化**：PTX/SASS 汇编、CUTLASS 模板库、warp-level 原语深入
2. **深度学习加速**：cuDNN / cuTENSOR、Triton 编译器、FlashAttention 实现（见 Day 31）
3. **高级话题**：GPU 驱动开发、RDMA / GPUDirect、MIG（Multi-Instance GPU）

本项目后续天数已覆盖其中不少：Day 31 Flash Attention、Day 32 cuSPARSE、Day 33 CUDA IPC、Day 34 内存池、Day 35 Nsight 分析、Day 39 NCCL AllReduce、Day 40 PyTorch CUDA 扩展等。

---

## 延伸阅读

- [CUDA C++ Best Practices Guide（性能优化权威指南）](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [NVIDIA Nsight Compute / Nsight Systems（性能分析工具）](https://developer.nvidia.com/tools-overview)
