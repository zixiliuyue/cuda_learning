# CUDA 30天学习项目 — 从入门到实战

> **目标硬件**: NVIDIA RTX 5060 Ti 16GB (Blackwell架构, Compute Capability 12.x)  
> **支持平台**: Windows 11 + WSL2 / macOS (通过 Python 版本或 Google Colab)  
> **CUDA版本**: CUDA 12.8+  
> **适合人群**: 有一定C/C++基础，希望系统学习GPU并行计算的开发者

---

## 项目结构

```
cuda_30days/
├── common/              # 公共工具库（错误检查、计时器、数学工具）
├── day01_hello_cuda/    # Day 1: 环境搭建与第一个CUDA程序
├── day02_vector_add/    # Day 2: 向量加法 — CUDA执行模型
├── day03_matrix_add/    # Day 3: 矩阵加法 — 内存模型
├── day04_matrix_transpose/  # Day 4: 矩阵转置 — 错误处理与调试
├── day05_device_query/  # Day 5: GPU设备查询与性能基准
├── ...                  # Day 6-30
├── CMakeLists.txt       # 全局构建入口
└── README.md            # 本文件
```

每个 `dayXX` 目录都是一个**完全独立的小项目**，包含：
- `README.md` — 当天学习内容的详细说明
- `CMakeLists.txt` — 独立构建配置
- `build.bat` / `build.sh` — 一键编译脚本
- `src/` — C++/CUDA 源代码（带详细中文注释）
- `python/` — Python 版本（Numba/PyTorch，macOS友好）

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

**全局构建所有项目**:
```powershell
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### macOS

macOS 自 10.14 后不再原生支持 NVIDIA GPU。提供两种运行方式：

**方式一: Python 版本（推荐）**
```bash
cd day01_hello_cuda/python
python hello_cuda.py
```

**方式二: Google Colab**
每个 `python/` 目录中的 `.ipynb` 笔记本可直接在 Colab 中运行，获得免费 T4 GPU。

**方式三: 本地编译验证（无GPU运行）**
```bash
cd day01_hello_cuda
./build.sh
./build/hello_cuda
# 会提示无 GPU，但代码可编译通过
```

---

## 学习路线图

| 阶段 | 天数 | 主题 |
|------|------|------|
| **第一阶段** | Day 1-5 | CUDA 基础与核心概念 |
| **第二阶段** | Day 6-12 | 内存优化与高级特性 |
| **第三阶段** | Day 13-20 | 算法实战与优化技巧 |
| **第四阶段** | Day 21-26 | 主流框架集成与项目 |
| **第五阶段** | Day 27-30 | 综合实战与拓展 |

---

## 公共工具库 (`common/`)

所有项目共享的辅助代码：

| 头文件 | 功能 |
|--------|------|
| `cuda_utils.h` | CUDA 错误检查宏 (`CUDA_SAFE_CALL`)、设备信息查询 |
| `timer.h` | 基于 CUDA Event 的高精度 GPU 计时器 |
| `math_utils.h` | 数组初始化、结果验证、浮点比较 |
| `platform.h` | 跨平台检测宏 (Windows/macOS/Linux) |

---

## 常见问题

**Q: macOS 上没有 NVIDIA GPU，能学习 CUDA 吗？**  
A: 可以！每个项目都提供了 Python 版本（Numba/PyTorch），语法与 CUDA C++ 高度对应。也可以通过 Google Colab 免费使用 T4 GPU。

**Q: 没有 RTX 5060 Ti 怎么办？**  
A: 项目支持所有 Compute Capability 5.0+ 的 NVIDIA GPU（GTX 900 系列及以上）。代码会自动适配角你的 GPU。

**Q: 编译报错 "nvcc not found"？**  
A: 确保 CUDA Toolkit 已安装，且 `nvcc` 在系统 PATH 中。Windows 安装后可能需要重启终端。

---

## 参考资源

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA by Example](https://developer.nvidia.com/cuda-example-introduction-general-purpose-gpu-programming)
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)

---

*Generated for CUDA 30-Day Learning Project*
