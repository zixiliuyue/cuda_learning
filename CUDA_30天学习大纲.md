# CUDA 30天学习项目 — 从入门到实战

> **目标硬件**: NVIDIA RTX 5060 Ti 16GB (Blackwell架构, Compute Capability 12.x)  
> **支持平台**: Windows 11 + WSL2 / macOS (通过 Google Colab 或 CPU fallback)  
> **CUDA版本**: CUDA 12.8+  
> **适合人群**: 有一定C/C++基础，希望系统学习GPU并行计算的开发者

---

## 项目结构总览

```
cuda_30days/
├── docs/                          # 学习文档和笔记
├── day01-day05/                   # 第一阶段：CUDA基础与核心概念
├── day06-day12/                   # 第二阶段：内存优化与高级特性
├── day13-day20/                   # 第三阶段：算法实战与优化技巧
├── day21-day26/                   # 第四阶段：主流框架集成与项目
├── day27-day30/                   # 第五阶段：综合实战与拓展
├── common/                        # 公共工具库（计时、校验、跨平台兼容）
├── scripts/                       # 自动化脚本（Windows/macOS）
├── requirements.txt               # Python依赖
└── README.md                      # 项目说明
```

---

## 第一阶段：CUDA基础与核心概念 (Day 1-5)

### Day 1: 环境搭建与第一个CUDA程序
- **目标**: 搭建开发环境，跑通第一个CUDA程序
- **内容**:
  - Windows: 安装CUDA Toolkit 12.8, Visual Studio 2022, Nsight
  - macOS: 配置Google Colab环境，或安装CUDA驱动（外置eGPU）
  - 编写 `hello_cuda.cu` — 最基础的CUDA程序
  - 理解 `.cu` 文件编译流程：`nvcc` 编译器工作原理
  - 跨平台编译脚本：`build.bat` (Windows) / `build.sh` (macOS/Linux)
- **实战项目**: Hello CUDA — 在GPU上打印线程ID

### Day 2: CUDA执行模型与线程层次
- **目标**: 深入理解Grid、Block、Thread三级结构
- **内容**:
  - Grid → Block → Thread 的层次关系图解
  - `dim3` 类型与维度配置
  - 线程索引计算：`blockIdx`, `threadIdx`, `blockDim`, `gridDim`
  - 一维/二维/三维线程组织的适用场景
  - 线程ID全局索引计算公式推导
- **实战项目**: 向量加法（Vector Add）— 一维并行

### Day 3: CUDA内存模型与数据传输
- **目标**: 掌握GPU内存层次和数据传输机制
- **内容**:
  - 主机内存（Host Memory）vs 设备内存（Device Memory）
  - `cudaMalloc`, `cudaFree`, `cudaMemcpy` 详解
  - 内存传输的三种方向：`cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice`
  - 内存带宽概念与传输开销分析
  - 统一内存（Unified Memory）初探：`cudaMallocManaged`
- **实战项目**: 矩阵加法（Matrix Addition）— 二维并行 + 内存管理

### Day 4: CUDA错误处理与调试基础
- **目标**: 建立健壮的CUDA程序调试能力
- **内容**:
  - CUDA错误码与 `cudaGetErrorString`
  - 宏封装：安全的CUDA API调用封装
  - `cudaDeviceSynchronize` 的作用与使用场景
  - Nsight Systems / Nsight Compute 工具入门
  - 使用 `printf` 在Kernel中调试（限制与技巧）
- **实战项目**: 带完整错误处理的矩阵转置（Matrix Transpose）

### Day 5: 设备查询与性能基准
- **目标**: 了解你的GPU硬件特性，建立性能意识
- **内容**:
  - `cudaGetDeviceProperties` 获取设备信息
  - 解析RTX 5060 Ti 16GB的关键参数：
    - SM数量、每个SM的CUDA核心数
    - 共享内存大小、寄存器数量
    - 显存带宽、L2缓存大小
    - Compute Capability 12.x 的新特性
  - 峰值算力理论计算（TFLOPS）
  - 编写通用设备信息查询工具
- **实战项目**: GPU硬件信息查询与报告生成工具

---

## 第二阶段：内存优化与高级特性 (Day 6-12)

### Day 6: 共享内存（Shared Memory）优化
- **目标**: 掌握共享内存的使用，减少全局内存访问
- **内容**:
  - 共享内存的物理位置与访问速度
  - `__shared__` 关键字与动态/静态分配
  - Bank Conflict原理与避免策略
  - 线程同步：`__syncthreads()` 的作用与陷阱
  - Tiling技术：分块计算减少全局内存访问
- **实战项目**: 共享内存优化的矩阵乘法（Tiled Matrix Multiply）

### Day 7: 全局内存合并访问（Coalesced Access）
- **目标**: 理解并优化全局内存访问模式
- **内容**:
  - 全局内存的物理结构：DRAM、缓存行
  - 什么是合并访问？为什么重要？
  - 行优先 vs 列优先的内存布局影响
  - 使用结构体时的内存对齐（`__align__`）
  - 只读数据的 `__restrict__` 与 `const` 优化提示
- **实战项目**: 内存访问模式对比实验（合并 vs 非合并）

### Day 8: 寄存器与局部内存优化
- **目标**: 理解寄存器限制和溢出影响
- **内容**:
  - 寄存器文件的结构与分配策略
  - 寄存器溢出到局部内存的性能代价
  - 占用率（Occupancy）概念与计算
  - Launch Bounds：`__launch_bounds__` 的使用
  - 编译器优化等级：`-O3`, `-use_fast_math`
- **实战项目**: 占用率分析工具与优化实验

### Day 9: 原子操作与同步原语
- **目标**: 掌握多线程间的同步与原子操作
- **内容**:
  - 原子操作：`atomicAdd`, `atomicMin`, `atomicMax`, `atomicCAS`
  - 原子操作的性能代价与使用限制
  - 归约（Reduction）算法：并行求和/求最大最小值
  - 树形归约与共享内存归约
  - Warp级别原语：`__shfl_down_sync`, `__ballot_sync`
- **实战项目**: 并行归约求和（Parallel Reduction Sum）

### Day 10: 流与异步执行
- **目标**: 掌握CUDA流的异步执行与重叠计算
- **内容**:
  - CUDA流（Stream）的概念与创建
  - 默认流（Stream 0）vs 显式流
  - 主机-设备数据传输与Kernel执行的重叠
  - 事件（Event）与精确计时
  - 多流并行执行的最佳实践
- **实战项目**: 多流并行的向量运算流水线

### Day 11: 常量内存与纹理内存
- **目标**: 了解特殊内存类型的使用场景
- **内容**:
  - 常量内存（Constant Memory）：缓存机制与适用场景
  - `__constant__` 关键字与 `cudaMemcpyToSymbol`
  - 纹理内存（Texture Memory）：硬件插值与缓存
  - CUDA Array与纹理对象绑定
  - RTX 50系新特性：纹理内存的改进
- **实战项目**: 使用常量内存优化查表操作（LUT）

### Day 12: 动态并行与CUDA Graphs
- **目标**: 了解高级执行模型
- **内容**:
  - 动态并行（Dynamic Parallelism）：Kernel中启动Kernel
  - `cudaDeviceSynchronize` 在动态并行中的使用
  - CUDA Graphs：预定义执行流程减少启动开销
  - Graph的捕获、实例化与执行
  - 适用场景与性能收益分析
- **实战项目**: 递归分治算法的动态并行实现

---

## 第三阶段：算法实战与优化技巧 (Day 13-20)

### Day 13: 矩阵运算库实战 — cuBLAS
- **目标**: 掌握NVIDIA官方线性代数库
- **内容**:
  - cuBLAS简介与API层次（Lv1/Lv2/Lv3）
  - 矩阵乘法 `cublasSgemm` / `cublasDgemm`
  - 行优先 vs 列优先的转换处理
  - 与手动实现的CUDA Kernel性能对比
  - Batched运算与Strided运算
- **实战项目**: 使用cuBLAS实现大规模矩阵乘法并对比性能

### Day 14: 快速傅里叶变换 — cuFFT
- **目标**: 掌握GPU加速的信号处理
- **内容**:
  - FFT算法原理回顾（蝶形运算）
  - cuFFT库API：`cufftPlan1d`, `cufftExecC2C`
  - 1D/2D/3D FFT与批量FFT
  - 实数FFT的优化（R2C, C2R）
  - 频域滤波应用
- **实战项目**: 图像频域滤波（高斯模糊/锐化）

### Day 15: 图像处理算法 — 卷积与滤波
- **目标**: 实现GPU加速的图像处理算法
- **内容**:
  - 2D卷积的CUDA实现
  - 边界处理策略（零填充、复制边缘、镜像）
  -  separable filter（可分离滤波器）优化
  - 共享内存Tiling在图像处理中的应用
  - 使用Texture Memory进行图像采样
- **实战项目**: GPU加速的高斯模糊、边缘检测（Sobel算子）

### Day 16: 排序与搜索算法
- **目标**: 掌握GPU并行排序算法
- **内容**:
  - 并行排序算法概览：Bitonic Sort, Radix Sort, Merge Sort
  - Thrust库简介：`thrust::sort`, `thrust::reduce`
  - 自定义Comparator与设备向量
  - 并行二分搜索与Lower Bound
  - 与STL算法的对比
- **实战项目**: 大规模数据并行排序与Top-K搜索

### Day 17: 前缀和（Scan / Prefix Sum）
- **目标**: 掌握并行前缀和算法
- **内容**:
  - 前缀和的定义与Inclusive/Exclusive区别
  - 串行算法 vs Blelloch并行Scan
  - 两步扫描算法：上扫（Up-sweep）与下扫（Down-sweep）
  - Thrust的 `thrust::inclusive_scan`
  - 应用场景：直方图计算、流压缩
- **实战项目**: 并行直方图计算与图像直方图均衡化

### Day 18: 图算法 — BFS与最短路径
- **目标**: GPU加速的图算法基础
- **内容**:
  - 图的CSR（Compressed Sparse Row）表示
  - 并行BFS：前沿（Frontier）扩展策略
  - 工作负载均衡问题
  - 原子操作在图算法中的使用
  - 使用cuGRAPH库简介
- **实战项目**: GPU并行BFS遍历大规模图

### Day 19: 随机数生成与蒙特卡洛方法
- **目标**: 掌握GPU并行随机数生成
- **内容**:
  - cuRAND库：`curandGenerateUniform`, `curandGenerateNormal`
  - 伪随机数 vs 准随机数（Sobol序列）
  - 每个线程独立的随机数种子管理
  - 蒙特卡洛积分原理
  - 蒙特卡洛在金融中的应用（期权定价）
- **实战项目**: GPU加速的蒙特卡洛估算Pi值与期权定价

### Day 20: 性能分析与调优综合
- **目标**: 系统化地进行CUDA程序调优
- **内容**:
  - Nsight Compute指标解读：
    - 内存吞吐量分析
    - 计算吞吐量分析
    - 占用率与Warp Stall原因
  - Roofline模型：分析计算受限 vs 内存受限
  - 常见性能瓶颈与解决策略
  - 迭代优化方法论
- **实战项目**: 对一个复杂Kernel进行端到端调优

---

## 第四阶段：主流框架集成与项目 (Day 21-26)

### Day 21: PyTorch CUDA扩展（C++ / CUDA）
- **目标**: 为PyTorch编写自定义CUDA算子
- **内容**:
  - PyTorch C++扩展机制：`torch.utils.cpp_extension`
  - 使用 `ATen` 张量库
  - 编写自定义CUDA Kernel并注册为PyTorch Op
  - 自动求导支持
  - 使用JIT编译与预编译对比
- **实战项目**: 自定义PyTorch CUDA算子 — 快速Softmax实现

### Day 22: Numba — Python中的CUDA编程
- **目标**: 使用Python直接编写CUDA Kernel
- **内容**:
  - Numba的 `@cuda.jit` 装饰器
  - Numba CUDA编程模型与原生CUDA的对应关系
  - 设备数组管理：`cuda.to_device`, `copy_to_host`
  - Numba内置的GPU算法：`cuda.reduce`, `cuda.scan`
  - Numba与CuPy的结合使用
- **实战项目**: 使用Numba实现GPU加速的K-Means聚类

### Day 23: OpenCV CUDA模块
- **目标**: GPU加速的计算机视觉处理
- **内容**:
  - OpenCV CUDA模块编译与配置
  - `cv::cuda::GpuMat` — GPU上的图像数据
  - 常用CUDA算子：滤波、形态学、颜色空间转换
  - CUDA流与OpenCV的集成
  - 自定义CUDA Kernel与OpenCV的结合
- **实战项目**: GPU加速的实时视频处理流水线

### Day 24: NVIDIA TensorRT 推理优化
- **目标**: 学习深度学习模型GPU推理加速
- **内容**:
  - TensorRT简介与工作流程
  - ONNX模型导入与TensorRT Engine构建
  - 精度校准：FP32, FP16, INT8量化
  - 动态Shape与优化Profile
  - TensorRT与PyTorch/TensorFlow的集成
- **实战项目**: 使用TensorRT加速ResNet图像分类推理

### Day 25: Multi-GPU编程基础
- **目标**: 掌握多GPU并行编程
- **内容**:
  - 多GPU系统架构与PCIe拓扑
  - `cudaSetDevice` 与设备上下文切换
  - Peer-to-Peer内存访问
  - NCCL库简介：多GPU通信原语
  - 数据并行 vs 模型并行策略
- **实战项目**: 多GPU并行矩阵运算

### Day 26: CUDA与深度学习训练优化
- **目标**: 理解深度学习中的CUDA优化技术
- **内容**:
  - 混合精度训练（Automatic Mixed Precision, AMP）
  - `torch.cuda.amp` 与 `GradScaler`
  - CUDA Kernel Fusion技术
  - FlashAttention算法原理与实现
  - 显存优化：梯度检查点、ZeRO
- **实战项目**: 使用AMP训练一个小型CNN并对比速度

---

## 第五阶段：综合实战与拓展 (Day 27-30)

### Day 27: 项目一 — GPU加速物理模拟（N-Body）
- **目标**: 综合应用CUDA优化技术
- **内容**:
  - N-Body问题：万有引力模拟
  - O(n²) 直接计算法的CUDA实现
  - 共享内存Tiling优化
  - Barnes-Hut算法的GPU实现思路
  - 实时可视化（OpenGL/CUDA Interop）
- **交付物**: 完整的N-Body模拟程序，含性能报告

### Day 28: 项目二 — GPU加速光线追踪（简易版）
- **目标**: 实现基础GPU光线追踪
- **内容**:
  - 光线追踪基本原理
  - CUDA中的递归与迭代实现
  - 加速结构：BVH（Bounding Volume Hierarchy）
  - 纹理采样与材质模型
  - 抗锯齿与后处理
- **交付物**: GPU光线追踪渲染器（支持基本几何体）

### Day 29: 项目三 — GPU加速深度学习推理引擎
- **目标**: 从零构建简易推理引擎
- **内容**:
  - 推理引擎架构设计
  - 手写CUDA Kernel实现关键算子：
    - Conv2D, ReLU, MaxPool, Linear, Softmax
  - 内存池管理与算子融合
  - 读取ONNX模型并执行推理
  - 与PyTorch推理结果对比验证
- **交付物**: 可运行的简易推理引擎，支持MNIST/CIFAR推理

### Day 30: 项目总结与进阶路线图
- **目标**: 总结所学，规划进阶方向
- **内容**:
  - 30天学习成果回顾
  - 完整项目代码整理与文档编写
  - 性能基准测试汇总报告
  - 进阶方向推荐：
    - CUDA底层优化（PTX/SASS）
    - CUTLASS模板库
    - NVIDIA cuDNN/cuTensor
    - Triton编译器
    - GPU驱动开发
  - 社区资源与学习路径推荐
- **交付物**: 完整的学习总结报告与代码仓库

---

## 公共基础设施设计

### `common/` 目录内容

```
common/
├── cuda_utils.h / cuda_utils.cu     # CUDA错误检查、设备查询宏
├── timer.h / timer.cu               # GPU事件计时器（支持多平台）
├── math_utils.h                     # 数学工具：随机初始化、精度比较
├── io_utils.h / io_utils.cpp        # 文件读写、图像加载
├── platform.h                       # 平台检测宏（Windows/macOS/Linux）
└── CMakeLists.txt                   # 统一CMake构建配置
```

### 跨平台支持方案

| 平台 | CUDA支持方案 | 构建工具 |
|------|-------------|---------|
| Windows 11 | 本地CUDA Toolkit 12.8 + MSVC | `build.bat` + CMake |
| Windows + WSL2 | WSL2中安装CUDA Toolkit | `build.sh` + CMake |
| macOS (Intel) | 外置eGPU + CUDA驱动 | `build.sh` + CMake |
| macOS (Apple Silicon) | Google Colab (免费T4 GPU) | Jupyter Notebook |
| macOS (通用) | CPU fallback模式 (用于编译验证) | `build.sh` + CMake |

### 每个项目的标准文件结构

```
dayXX_topic/
├── README.md              # 当天学习内容说明
├── main.cu                # 主程序入口
├── kernel.cu / kernel.cuh # CUDA Kernel实现
├── host_utils.cpp/.h      # 主机端辅助函数
├── Makefile / CMakeLists.txt  # 构建配置
├── build.bat              # Windows一键编译
├── build.sh               # macOS/Linux一键编译
├── test_data/             # 测试数据
└── results/               # 运行结果与性能报告
```

### 编译系统

- **CMake**: 跨平台统一构建
- **一键脚本**: `build.bat` (Windows) / `build.sh` (macOS/Linux)
- **Python辅助**: `run.py` 用于运行测试、收集性能数据

---

## RTX 5060 Ti 16GB 硬件参数速查

| 参数 | 数值 |
|------|------|
| 架构 | Blackwell |
| Compute Capability | 12.x |
| 显存 | 16 GB GDDR6 |
| 显存带宽 | ~448 GB/s (预计) |
| CUDA核心数 | ~4608 (预计) |
| 张量核心 | 第5代 (FP8/FP4支持) |
| 光线追踪核心 | 第4代 |
| TDP | 180W (预计) |
| PCIe | PCIe 5.0 x16 |

> 注：RTX 5060 Ti为虚构/未来型号，基于Blackwell架构特性设计。如实际硬件有差异，部分参数需调整。

---

## 学习建议

1. **每天投入时间**: 建议2-3小时学习 + 1小时实践
2. **先理解原理**: 不要急于抄代码，先理解GPU并行计算的原理
3. **性能意识**: 从Day 1就养成测量性能的习惯
4. **多平台测试**: 如有条件，在Windows和Colab上都跑一遍
5. **记笔记**: 每个Day的README.md中记录遇到的问题和解决方案
6. **社区交流**: 遇到问题可参考NVIDIA开发者论坛、Stack Overflow

---

## 参考资源

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [CUDA by Example (书)](https://developer.nvidia.com/cuda-example-introduction-general-purpose-gpu-programming)
- [Programming Massively Parallel Processors (书)](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0)
- [NVIDIA On-Demand Training](https://www.nvidia.com/en-us/on-demand/)

---

*Generated for CUDA 30-Day Learning Project | RTX 5060 Ti 16GB | Windows & macOS*
