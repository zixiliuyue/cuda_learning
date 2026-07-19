# CUDA 40天学习项目 — 从入门到实战（完整大纲）

> **目标硬件**: NVIDIA RTX 5060 Ti 16GB (Blackwell架构, Compute Capability 12.x, sm_120)
> **支持平台**: Windows 11 + WSL2 / macOS (通过 Python 版本或 Google Colab)
> **CUDA版本**: CUDA 12.8 ~ 13.x（已在 CUDA 13.1 实测全部编译通过）
> **适合人群**: 有一定C/C++基础，希望系统学习GPU并行计算的开发者

---

## 40 天总目标

完成本课程后，你应当能够：

1. **独立编写并调试** CUDA C++ Kernel，熟练掌握执行模型、内存层次与错误处理；
2. **系统性优化** GPU 程序性能——会用共享内存、合并访问、占用率、流并发等手段，并能用 Nsight 定位瓶颈；
3. **调用主流 CUDA 生态库**（cuBLAS/cuFFT/cuSPARSE/Thrust/cuRAND）解决线性代数、信号处理、排序、图算法等实际问题；
4. **将 CUDA 集成进上层框架**——为 PyTorch 编写并优化自定义算子，会用 Numba/OpenCV/TensorRT，理解多 GPU 与混合精度训练；
5. **独立完成综合项目**（N-Body 模拟、光线追踪、推理引擎）并产出性能报告；
6. **掌握 CUDA 的进阶专题与工程化封装**（Flash Attention、内存池、CUDA IPC、pybind11/ctypes/Cython、NCCL）。

> 每一天的目标都服务于以上某条总目标，README 顶部会标注对应关系。

---

## 学习路线图（8 阶段 40 天）

| 阶段 | 天数 | 主题 | 承接关系 |
|------|------|------|----------|
| **第一阶段: 基础入门** | Day 1-5 | CUDA 核心概念与编程模型 | 起点 |
| **第二阶段: 内存优化** | Day 6-12 | 共享内存、合并访问、占用率、原子/归约、流、动态并行 | 在第一阶段"能跑通"的基础上追求"跑得快" |
| **第三阶段: 算法实战** | Day 13-20 | cuBLAS/cuFFT、图像、排序、Scan、图算法、蒙特卡洛、调优 | 用第二阶段的优化手段解决真实算法问题 |
| **第四阶段: 框架集成** | Day 21-26 | PyTorch/Numba/OpenCV/TensorRT/多GPU/AMP | 从"手写 Kernel"上升到"融入工业框架" |
| **第五阶段: 综合项目** | Day 27-30 | N-Body、光线追踪、推理引擎、阶段总结 | 综合运用前 26 天的全部技能 |
| **第六阶段: 进阶专题** | Day 31-35 | Flash Attention、cuSPARSE、CUDA IPC、内存池、Nsight 深度分析 | 深入前沿算法与系统级优化 |
| **第七阶段: Python 封装** | Day 36-38 | pybind11、ctypes、Cython 三种 CUDA↔Python 封装方式 | 把 CUDA 能力交付给 Python 生态 |
| **第八阶段: 扩展实战** | Day 39-40 | NCCL 多GPU通信、PyTorch CUDA Extension 完整实战 | 面向真实生产的收尾实战 |

---

## 第一阶段：CUDA基础与核心概念 (Day 1-5)

> **阶段目标**：搭好环境，理解 GPU 并行的三大基石——执行模型、内存模型、错误处理——并建立性能测量意识。

### Day 1: 环境搭建与第一个CUDA程序
- **目标**: 搭建开发环境，跑通第一个 CUDA 程序，理解 `nvcc` 编译流程
- **实战项目**: Hello CUDA — 在 GPU 上打印线程 ID

### Day 2: CUDA执行模型与线程层次
- **目标**: 深入理解 Grid、Block、Thread 三级结构与全局索引计算
- **实战项目**: 向量加法（Vector Add）— 一维并行

### Day 3: CUDA内存模型与数据传输
- **目标**: 掌握 GPU 内存层次、`cudaMalloc/cudaMemcpy`、统一内存
- **实战项目**: 矩阵加法（Matrix Addition）— 二维并行 + 内存管理

### Day 4: CUDA错误处理与调试基础
- **目标**: 建立健壮的 CUDA 调试能力（错误码、宏封装、Kernel 内 printf）
- **实战项目**: 带完整错误处理的矩阵转置（Matrix Transpose）

### Day 5: 设备查询与性能基准
- **目标**: 读懂 GPU 硬件参数（SM、共享内存、带宽），建立性能意识
- **实战项目**: GPU 硬件信息查询与报告生成工具

---

## 第二阶段：内存优化与高级特性 (Day 6-12)

> **阶段目标**：把第一阶段"能跑"的程序优化到"跑得快"，围绕内存层次与并发展开。

### Day 6: 共享内存（Shared Memory）优化
- **目标**: 掌握共享内存与 Tiling，减少全局内存访问
- **实战项目**: 共享内存优化的矩阵乘法（Tiled Matrix Multiply）

### Day 7: 全局内存合并访问（Coalesced Access）
- **目标**: 理解并优化全局内存访问模式（合并 vs 非合并）
- **实战项目**: 内存访问模式对比实验（stride 1/2/32/随机）

### Day 8: 寄存器与局部内存 / 占用率优化
- **目标**: 理解寄存器限制、溢出代价、占用率（Occupancy）与 `__launch_bounds__`
- **实战项目**: 占用率分析工具与优化实验

### Day 9: 原子操作与并行归约
- **目标**: 掌握原子操作、树形归约、共享内存归约、Warp Shuffle 归约
- **实战项目**: 并行归约求和（对比 atomic / shared / warp-shuffle 三种实现）

### Day 10: 流与异步执行
- **目标**: 掌握 CUDA 流的异步执行与"拷贝-计算"重叠、Event 计时
- **实战项目**: 多流并行的向量运算流水线

### Day 11: 常量内存与纹理内存
- **目标**: 了解常量内存的广播缓存与纹理内存的硬件插值/缓存
- **实战项目**: 使用常量内存优化查表操作（LUT）

### Day 12: 动态并行与CUDA Graphs
- **目标**: 了解 Kernel 中启动 Kernel、以及 CUDA Graphs 降低启动开销
- **实战项目**: 递归分治算法的动态并行实现

---

## 第三阶段：算法实战与优化技巧 (Day 13-20)

> **阶段目标**：调用官方库并手写经典并行算法，把第二阶段的优化手段用到真实问题上。

### Day 13: 矩阵运算库实战 — cuBLAS
- **目标**: 掌握 cuBLAS 矩阵乘法，与手写 Tiled Kernel 对比性能
- **实战项目**: cuBLAS 大规模矩阵乘法性能对比

### Day 14: 快速傅里叶变换 — cuFFT
- **目标**: 掌握 GPU 加速的 1D/2D FFT 与频域滤波
- **实战项目**: FFT 正变换/逆变换与频域处理

### Day 15: 图像处理算法 — 卷积与滤波
- **目标**: 实现 GPU 加速的 2D 卷积、高斯模糊、Sobel 边缘检测
- **实战项目**: GPU 图像滤波（高斯模糊 + Sobel）

### Day 16: 排序与搜索算法
- **目标**: 掌握 GPU 并行排序（Bitonic Sort）与 Thrust 库
- **实战项目**: 大规模数据并行排序

### Day 17: 前缀和（Scan / Prefix Sum）
- **目标**: 掌握 Blelloch 并行 Scan（上扫/下扫两阶段）
- **实战项目**: 并行前缀和与应用（直方图/流压缩）

### Day 18: 图算法 — BFS
- **目标**: 掌握图的 CSR 表示与并行 BFS 前沿扩展策略
- **实战项目**: GPU 并行 BFS 遍历大规模图

### Day 19: 随机数生成与蒙特卡洛方法
- **目标**: 掌握 cuRAND、每线程种子管理、蒙特卡洛积分
- **实战项目**: 蒙特卡洛估算 Pi + Black-Scholes 期权定价

### Day 20: 性能分析与调优综合
- **目标**: 系统化调优方法论（Roofline、Warp Stall、迭代优化）
- **实战项目**: 对向量加法 Kernel 进行多轮端到端调优（naive→unroll→…）

---

## 第四阶段：主流框架集成与项目 (Day 21-26)

> **阶段目标**：从"手写 Kernel"上升到"融入工业框架"。注意本阶段工具链切换较大，Day 21 是关键过渡点。

### Day 21: PyTorch CUDA扩展（C++ / CUDA）
- **目标**: 为 PyTorch 编写自定义 CUDA 算子（cpp_extension / ATen / 自动求导）
- **实战项目**: 自定义 PyTorch CUDA 算子

### Day 22: Numba — Python中的CUDA编程
- **目标**: 用 `@cuda.jit` 在 Python 中写 Kernel，理解与原生 CUDA 的对应关系
- **实战项目**: Numba GPU 加速算法演示

### Day 23: OpenCV CUDA模块
- **目标**: 使用 `cv::cuda::GpuMat` 与 CUDA 算子做 GPU 视觉处理
- **实战项目**: GPU 加速的图像处理流水线

### Day 24: NVIDIA TensorRT 推理优化
- **目标**: 理解 TensorRT 工作流、ONNX 导入、FP16/INT8 量化
- **实战项目**: TensorRT 加速推理概念演示

### Day 25: Multi-GPU编程基础
- **目标**: 掌握 `cudaSetDevice`、P2P 访问、多 GPU 数据/模型并行策略
- **实战项目**: 多 GPU 并行矩阵运算

### Day 26: CUDA与深度学习训练优化
- **目标**: 理解混合精度训练（AMP）、Kernel Fusion、FlashAttention 原理（原理层，实现见 Day 31）
- **实战项目**: FP16 vs FP32 计算对比

---

## 第五阶段：综合实战与拓展 (Day 27-30)

> **阶段目标**：综合运用前 26 天的技能完成完整项目。Day 27-29 均为**多日项目**，建议每个预留 2-3 天，按 MVP→优化→可视化 分里程碑推进。

### Day 27: 项目一 — GPU加速物理模拟（N-Body）【建议 2-3 天】
- **目标**: 综合应用共享内存 Tiling 优化 O(n²) 全对力计算
- **交付物**: N-Body 模拟程序 + 性能报告

### Day 28: 项目二 — GPU加速光线追踪（简易版）【建议 2-3 天】
- **目标**: 实现基础 GPU 光线追踪（几何求交、材质、抗锯齿）
- **交付物**: GPU 光线追踪渲染器

### Day 29: 项目三 — GPU加速深度学习推理引擎【建议 2-3 天】
- **目标**: 手写 Conv2D/ReLU/MaxPool/Linear/Softmax 算子，组装推理引擎
- **交付物**: 可运行的简易推理引擎

### Day 30: 阶段总结与进阶路线图
- **目标**: 回顾前 30 天成果，汇总性能基准，规划后 10 天进阶方向
- **交付物**: 学习总结报告

---

## 第六阶段：进阶专题 (Day 31-35)

> **阶段目标**：深入前沿算法与系统级优化，是全课程难度最高的部分，强依赖第二、三阶段的内存优化与归约基础。

### Day 31: Flash Attention 原理与实现（教学简化版）
- **目标**: 掌握 Tiling + Online Softmax，把注意力显存从 O(N²) 降到 O(N)
- **前置依赖**: Day 6 Tiling、Day 9 归约、Day 26 注意力原理
- **实战项目**: 标准注意力 / Flash / float4 向量化 三个 Kernel 对比

### Day 32: cuSPARSE — 稀疏矩阵运算
- **目标**: 掌握稀疏矩阵存储格式（CSR/COO）与 cuSPARSE SpMV/SpMM
- **前置依赖**: Day 13 cuBLAS、Day 18 CSR 图表示
- **实战项目**: 稀疏矩阵-向量乘法（SpMV）

### Day 33: CUDA IPC — 进程间共享显存
- **目标**: 掌握 `cudaIpcGetMemHandle` 在多进程间共享 GPU 显存
- **前置依赖**: Day 3 内存模型
- **实战项目**: 跨进程共享显存演示

### Day 34: 自定义 GPU 内存池
- **目标**: 理解频繁 `cudaMalloc/cudaFree` 的开销，实现内存池复用显存
- **前置依赖**: Day 3 内存模型、Day 29 推理引擎内存管理
- **实战项目**: 自定义 GPU 内存池分配器

### Day 35: Nsight 深度性能分析
- **目标**: 熟练使用 Nsight Systems/Compute 做时间线与 Kernel 级深度剖析
- **前置依赖**: Day 20 调优方法论
- **实战项目**: 对典型 Kernel 做 Nsight 深度分析

---

## 第七阶段：Python 封装 (Day 36-38)

> **阶段目标**：把 CUDA 能力交付给 Python 生态。三天是"同一目标（Python 调用 CUDA）的三种技术路线"，重在**横向对比**其易用性/性能/适用场景，而非重复学习。

### Day 36: pybind11 封装 CUDA
- **目标**: 用 pybind11 把 CUDA Kernel 暴露为 Python 模块
- **实战项目**: pybind11 CUDA 扩展

### Day 37: ctypes 调用 CUDA 动态库
- **目标**: 把 CUDA Kernel 导出为 C 风格 API，用 ctypes 调用
- **实战项目**: ctypes 加载 CUDA 动态库

### Day 38: Cython + CUDA 混合编程
- **目标**: 用 Cython 包装 CUDA C 风格函数
- **实战项目**: Cython CUDA 封装

> **三种封装方式对比**（学习本阶段时请填写并对照）：
>
> | 方式 | 易用性 | 性能开销 | 类型安全 | 适用场景 |
> |------|--------|----------|----------|----------|
> | pybind11 | 高（C++ 风格） | 低 | 强 | C++ 项目、需要 STL/类绑定 |
> | ctypes | 中（纯 Python，无需编译扩展） | 中（每次调用有封送开销） | 弱 | 快速原型、已有 .dll/.so |
> | Cython | 中（需学 .pyx 语法） | 低 | 中 | 需与现有 Python/C 代码深度融合 |

---

## 第八阶段：扩展实战 (Day 39-40)

> **阶段目标**：面向真实生产场景收尾——分布式通信与工业级 PyTorch 算子。

### Day 39: NCCL 多 GPU AllReduce 通信实战
- **目标**: 掌握 NCCL 集合通信原语（AllReduce），理解数据并行训练底层
- **前置依赖**: Day 25 多 GPU 基础
- **说明**: NCCL 官方仅支持 Linux，Windows 上编译为 CPU 模拟版本
- **实战项目**: 多 GPU AllReduce 通信

### Day 40: PyTorch CUDA Extension 完整实战
- **目标**: 完整实现带前向/反向的 PyTorch 自定义算子（ReLU、LayerNorm）并与 PyTorch 交叉验证
- **前置依赖**: Day 21 PyTorch 扩展、Day 26 AMP
- **交付物**: 可 `pip install` 的 PyTorch CUDA 扩展（forward + backward + 数值校验）

---

## 公共基础设施

### `common/` 目录

| 目录/文件 | 功能 |
|-----------|------|
| `include/cuda_utils.h` | CUDA 错误检查宏 (`CUDA_SAFE_CALL`)、设备信息查询 |
| `include/timer.h` | 基于 CUDA Event 的高精度 GPU 计时器 |
| `include/math_utils.h` | 数组初始化、结果验证、浮点比较 |
| `include/platform.h` | 跨平台检测宏 (Windows/macOS/Linux) |
| `cuda_kernels/softmax_kernel.cuh` | 高频基础 Kernel：Softmax 前向（warp-level 规约） |
| `cuda_kernels/layer_norm_kernel.cuh` | 高频基础 Kernel：Layer Normalization |
| `cuda_kernels/gelu_kernel.cuh` | 高频基础 Kernel：GELU 激活函数 |
| `cuda_kernels/dropout_kernel.cuh` | 高频基础 Kernel：Dropout（cuRAND 掩码） |

### 每天的标准文件结构

```
dayXX_topic/
├── README.md              # 当天学习内容说明（目标/原理/图解/验收标准/动手挑战）
├── src/main.cu            # C++/CUDA 源代码（带详细中文注释）
├── python/*.py            # Python 版本（Numba/PyTorch，macOS 友好）
├── CMakeLists.txt         # 独立构建配置
├── build.bat              # Windows 一键编译
└── build.sh               # macOS/Linux 一键编译
```

### 验收与自检

- 每天 README 末尾提供**「验收标准」**，明确"看到什么输出 / 达到什么加速比"算掌握；
- 每个阶段末尾提供**「阶段自检 Checklist」**，用于自评是否可进入下一阶段；
- `tests/` 目录提供 pytest 自动化用例，可 `python -m pytest tests/ -v` 验证（无 GPU 自动跳过）。

---

## 学习建议

1. **每天投入**: 2-3 小时学习 + 1 小时实践（Day 27-29 综合项目需多日）
2. **先原理后代码**: 先理解并行思想，再读/写代码
3. **性能意识**: 从 Day 1 就养成测量习惯
4. **完成动手挑战**: 每天 README 的「动手挑战」是从"读代码"到"写代码"的关键
5. **记笔记**: 在每天 README 记录踩坑，配合 `COMMON_PITFALLS.md`

---

## 参考资源

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [Programming Massively Parallel Processors (书)](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)

---

*CUDA 40-Day Learning Project | RTX 5060 Ti 16GB (sm_120) | CUDA 12.8~13.x*
