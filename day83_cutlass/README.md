# Day 83: CUTLASS 入门 — GEMM 模板库

> 🧱 接前 40 天大纲遗留的「用 CUTLASS 写 GEMM」项。Day6 你手写过 Tiling 矩阵乘、Day13 用过 cuBLAS，本天看 NVIDIA 开源的 **CUTLASS** 如何用「分层 tiling + C++ 模板组合」把高性能 GEMM 拆成可复用积木。

## 学习目标

1. **描述** CUTLASS 的三层 tile：ThreadBlock → Warp → Thread/Instruction
2. **解释** 主循环软件流水（双缓冲 + `cp.async`）如何隐藏访存延迟
3. **复现** 分层分块 GEMM 的概念（NumPy），验证与朴素/torch 结果一致
4. **读懂** CUTLASS C++ 使用骨架，知道换精度/架构只需改模板参数

---

## 前置依赖

- **Day 6 Tiling 矩阵乘**：CUTLASS 的 ThreadBlock tile 就是它的工业化版本
- **Day 13 cuBLAS**：CUTLASS 是「可定制的 cuBLAS」，能融合尾操作
- **Day 7 合并访存 / 共享内存**：理解 GMEM→SMEM→REG 的搬运层级

---

## 核心概念

### 分层 tiling（三层 + 主循环流水）

```
C[M,N] = A[M,K] × B[K,N]

ThreadBlock Tile  128×128，沿 K 分 32 循环   GMEM → SMEM
Warp Tile          64×64，block 内各 warp     SMEM → Register
Instruction Tile   16×8×16，Tensor Core MMA   Register → Tensor Core
```

- **软件流水**：算当前 Ktile 时，用 `cp.async` 异步预取下一个 Ktile（双/多缓冲），把访存延迟藏在计算后面。这是 CUTLASS 高性能的关键。
- **数据搬运层级**：`GMEM(慢/大) → SMEM(快) → Register(更快) → Tensor Core(最快)`。

### 模板组合

CUTLASS 把「怎样高效做 GEMM」参数化，你像搭积木一样指定：元素类型 / 布局 / 三层 Tile Shape / 目标架构（SM70/80/90）/ 流水级数 / Epilogue（`alpha*AB+beta*C`、加偏置、激活）。换精度或架构，**只改模板参数**，编译器实例化出专用 kernel。

### 本天的 Python 复现

脚本用 NumPy 写了一个 `tiled_gemm`，用三层嵌套循环 + 沿 K 分块累加**复现结构**（不追求性能），并统计各层 tile 数量，验证与 `torch.matmul` / 朴素三重循环**数值完全一致**——说明 tiling 只改变计算顺序、不改变数学结果。

---

## 文件结构

```
day83_cutlass/
├── python/cutlass_intro.py   # 分层 tiling 讲解 + NumPy 复现 + C++ 骨架
├── build.sh                  # Linux/WSL2 运行脚本
├── build.bat                 # Windows 运行脚本
└── README.md                 # 本文件
```

> CUTLASS 本体是 C++ 模板库（GPU 编译运行）。本天**原理为主**，用 NumPy 复现概念；真实使用见下方 C++ 骨架。

---

## 编译与运行

### 单机（无需 GPU，理解概念）

```bash
cd day83_cutlass
python python/cutlass_intro.py
```

### 真实 CUTLASS（Linux + NVIDIA GPU）

```bash
git clone https://github.com/NVIDIA/cutlass
nvcc -I cutlass/include -arch=sm_80 gemm.cu -o gemm   # 见脚本内 C++ 骨架
# 或用 Python 封装：
pip install nvidia-cutlass
```

### Windows

```powershell
cd day83_cutlass
.\build.bat
```

---

## 知识点总结

1. **三层 tile**：ThreadBlock / Warp / Instruction，对应 GMEM/SMEM/Register/TensorCore。
2. **软件流水**：`cp.async` + 多缓冲，计算与访存重叠，隐藏延迟。
3. **模板组合**：换精度/布局/架构/Epilogue 只改模板参数，复用同一套高性能框架。
4. **Epilogue 融合**：GEMM 后直接做偏置/激活/缩放，省去额外 kernel 与访存。
5. **tiling 不改数学**：分块只重排计算顺序，结果与朴素 GEMM 一致。

---

## 验收标准

- 程序打印三层 tile 结构与数据搬运层级，你能说出每层数据在哪级存储。
- NumPy `tiled_gemm` 与 `torch.matmul` / 朴素三重循环最大误差 ≈ 0，显示「一致」。
- 你能指出改成 int8 或 Hopper(SM90) 时，CUTLASS 骨架里要改哪几个模板参数。

---

## 动手挑战

1. **改 Tile 形状**：把 `block_tile` 改成 (16,16,8) / (48,48,16)，观察各层 tile 数量变化。
2. **非整除**：用不能被 tile 整除的尺寸（如 M=100），验证边界 `min(...)` 处理正确。
3. **加 Epilogue**：在 `tiled_gemm` 结果上做 `alpha*C + beta*C0`，模拟 LinearCombination。
4. **上真机**：clone CUTLASS，编译 profiler `tools/profiler`，跑 `cutlass_profiler --operation=Gemm` 看不同 tile 配置的实测性能。

---

## 常见问题

- **Q：有了 cuBLAS 为什么还要 CUTLASS？**
  A：cuBLAS 是黑盒库；CUTLASS 开源可定制，能把激活/偏置/量化**融合进 GEMM 的 Epilogue**，还能实现自定义数据类型与融合 kernel，是很多推理引擎的底座。

- **Q：Python 里能直接用 CUTLASS 吗？**
  A：可以，`pip install nvidia-cutlass` 提供 Python 接口，但仍需 NVIDIA GPU 与 CUDA 环境。本天无 GPU 时用 NumPy 理解结构即可。

---

## 延伸阅读

- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUTLASS 官方文档 — GEMM 分层结构](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
