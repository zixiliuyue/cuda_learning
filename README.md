# CUDA 40天学习项目 — 从入门到实战

> **目标硬件**: NVIDIA RTX 5060 Ti 16GB (Blackwell架构, Compute Capability 12.x, sm_120)
> **支持平台**: Windows 11 + WSL2 / macOS (通过 Python 版本或 Google Colab)
> **CUDA版本**: CUDA 12.8 ~ 13.x（已在 CUDA 13.1 实测全部编译通过）
> **适合人群**: 有一定C/C++基础，希望系统学习GPU并行计算的开发者

---

## 编译验收状态

全部 **40 天项目**已在 **CUDA 13.1 + Visual Studio 2022 + RTX 5060 Ti (sm_120)** 环境下
**实测编译通过**。新增的基础 Kernel 库和 NCCL 项目均编译通过。
你可以用内置脚本一键复现验证：

```powershell
# CUDA C++ 编译验证（无需 CMake，自动定位 MSVC）
python build_all.py

# Python 单元测试（含 Day 7/9/10 新增测试）
python -m pytest tests/ -v

# 新手踩坑指南
code COMMON_PITFALLS.md

# Python 语法全量检查
python -c "import glob,os;[compile(open(f,encoding='utf-8').read(),f,'exec') or print(f'[OK] {os.path.basename(f)}') for f in glob.glob('day*/python/*.py')]"
```

> **CUDA 13 兼容性说明**：CUDA 13 相比 12.x 有两处破坏性变化，本项目已全部适配：
> 1. **架构**：CUDA 13 移除了 `sm_50~sm_70`，最低支持 `sm_75`。CMake 已按 nvcc 版本自动选择架构。
> 2. **API**：`cudaDeviceProp` 移除了 `memoryClockRate`、`memoryBusWidth`、
>    `singleToDoublePrecisionPerfRatio` 字段，代码改用 `cudaDeviceGetAttribute()` 查询（跨版本兼容）。

---

## 项目结构

```
cuda_40days/
├── .vscode/                 # VS Code 调试/编译配置（launch.json, tasks.json）
├── common/                  # 公共工具库 + 高频基础Kernel（softmax/layernorm/gelu/dropout）
├── tests/                   # Python 单元测试（pytest, 28 个用例）
├── day01_hello_cuda/        # Day 1: 环境搭建与第一个CUDA程序
├── ...
├── day40_pytorch_cuda_extension/  # Day 40: PyTorch CUDA Extension 完整实战
├── COMMON_PITFALLS.md       # 新手常见踩坑文档
├── build_all.py             # 一键编译验证脚本
├── requirements.txt         # Python 依赖清单
├── .gitignore               # Git 忽略规则
└── README.md                 # 本文件
```

每个 `dayXX` 目录都是一个**完全独立的小项目**，包含：
- `README.md` — 当天学习内容的详细说明
- `CMakeLists.txt` — 独立构建配置
- `build.bat` / `build.sh` — 一键编译脚本
- `src/main.cu` — C++/CUDA 源代码（带详细中文注释）
- `python/*.py` — Python 版本（Numba/PyTorch，macOS友好）

---

## 快速开始

### Windows 11 (推荐)

**前提条件**:
- NVIDIA GPU（如 RTX 5060 Ti）
- CUDA Toolkit 12.8+ ([下载](https://developer.nvidia.com/cuda-downloads))
- Visual Studio 2022 Community
- CMake 3.18+

**一键编译运行 Day 1**:
```powershell
cd day01_hello_cuda
.\build.bat
.\build\Debug\hello_cuda.exe
```

**VS Code 一键调试**:
> 打开项目后按 F5 即可编译并调试当前 .cu 文件。
> 需要安装推荐扩展：C/C++ Extension Pack + Python。

**全局构建所有项目**:
```powershell
python build_all.py
```

### macOS

macOS 自 10.14 后不再原生支持 NVIDIA GPU。提供两种运行方式：

**方式一: Python 版本（推荐）**
```bash
pip install -r requirements.txt
cd day01_hello_cuda/python
python hello_cuda.py
```

**方式二: Google Colab**
每个 `python/` 目录中的脚本可直接在 Colab 中运行，获得免费 T4 GPU。

---

## 学习路线图

| 阶段 | 天数 | 主题 |
|------|------|------|
| **第一阶段: 基础入门** | Day 1-5 | CUDA 核心概念与编程模型 |
| **第二阶段: 内存优化** | Day 6-12 | 共享内存、合并访问、流、动态并行 |
| **第三阶段: 算法实战** | Day 13-20 | cuBLAS/cuFFT、排序、Scan、图算法、蒙特卡洛 |
| **第四阶段: 框架集成** | Day 21-26 | PyTorch/Numba/OpenCV/TensorRT/多GPU/AMP |
| **第五阶段: 综合项目** | Day 27-30 | N-Body、光线追踪、推理引擎、综合总结 |
| **第六阶段: 进阶专题** | Day 31-35 | Flash Attention、cuSPARSE、CUDA IPC、内存池、Nsight 分析 |
| **第七阶段: Python 封装** | Day 36-38 | pybind11、ctypes、Cython 三种 CUDA Python 封装方式 |
| **第八阶段: 扩展实战** | **Day 39-40** | **NCCL 多GPU通信、PyTorch CUDA Extension 完整实战** |

> 📖 完整的每日目标、阶段承接关系与 40 天总目标见 [CUDA_40天学习大纲.md](CUDA_40天学习大纲.md)。
> ✅ 每个阶段结束时用 [STAGE_CHECKLISTS.md](STAGE_CHECKLISTS.md) 自评是否可进入下一阶段。
> 每个 `dayXX/README.md` 都包含「学习目标 / 核心概念 / 验收标准 / 动手挑战」，全部 40 天已补齐。
> 🚀 想进一步走向 AI Infra（大模型训练/推理/部署）？见进阶扩展路线 [CUDA_进阶大纲_Day41-90.md](CUDA_进阶大纲_Day41-90.md)（Triton / LLM 推理 / 模型压缩 / 分布式训练 / 通信存储）。

---

## 公共工具库 (`common/`)

| 目录/文件 | 功能 |
|-----------|------|
| `include/cuda_utils.h` | CUDA 错误检查宏 (`CUDA_SAFE_CALL`)、设备信息查询 |
| `include/timer.h` | 基于 CUDA Event 的高精度 GPU 计时器 |
| `include/math_utils.h` | 数组初始化、结果验证、浮点比较 |
| `include/platform.h` | 跨平台检测宏 (Windows/macOS/Linux) |
| `cuda_kernels/softmax_kernel.cuh` | 高频基础 Kernel：Softmax 前向（warp-level 规约） |
| `cuda_kernels/layer_norm_kernel.cuh` | 高频基础 Kernel：Layer Normalization |
| `cuda_kernels/gelu_kernel.cuh` | 高频基础 Kernel：GELU 激活函数（精确版 + 近似版） |
| `cuda_kernels/dropout_kernel.cuh` | 高频基础 Kernel：Dropout（cuRAND 掩码） |

---

## 常见问题

**Q: macOS 上没有 NVIDIA GPU，能学习 CUDA 吗？**
A: 可以！全部 40 天项目均提供 Python 版本（Numba/PyTorch/CuPy），语法与 CUDA C++ 高度对应。也可以通过 Google Colab 免费使用 T4 GPU。

**Q: 没有 RTX 5060 Ti 怎么办？**
A: 项目支持所有 Compute Capability 5.0+ 的 NVIDIA GPU（GTX 900 系列及以上）。代码会自动适配你的 GPU。

**Q: 编译报错 "nvcc not found"？**
A: 确保 CUDA Toolkit 已安装，且 `nvcc` 在系统 PATH 中。Windows 安装后可能需要重启终端。

**Q: 如何运行单元测试？**
A: `pip install pytest numba numpy` 然后 `python -m pytest tests/ -v`。无 GPU 时测试自动跳过。

**Q: 新手常踩哪些坑？**
A: 参考 `COMMON_PITFALLS.md`，涵盖编译、运行时、性能、调试、PyTorch 扩展五大类共 21 个常见问题及解决方案。

**Q: NCCL 在 Windows 上能用吗？**
A: NCCL 官方仅支持 Linux。Windows 上 Day 39 会编译为 CPU 模拟版本。如需真实多 GPU 通信，请使用 WSL2 或 Linux 系统。

---

## 参考资源

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA by Example](https://developer.nvidia.com/cuda-example-introduction-general-purpose-gpu-programming)
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [Cython Documentation](https://cython.readthedocs.io/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [PyTorch CUDA Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)

---

*Generated for CUDA 40-Day Learning Project*