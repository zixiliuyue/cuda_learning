# Day 41: Triton 入门 — 编程模型 vs CUDA

> 🧩 **第九阶段（现代算子编程）开篇**。前 40 天你用 CUDA C++ 手写 kernel，本阶段学习业界写融合算子的主流工具 **Triton**——它让你用 Python 就能写出接近手工优化的 GPU kernel。

## 学习目标

1. **解释** Triton 的 program/BLOCK 编程模型与 CUDA thread/block 模型的对应关系
2. **实现** Triton 版向量加法与行 Softmax，并与 PyTorch 结果交叉验证
3. **对比** Triton 的 `mask` 边界处理与 CUDA 的 `if (tid < n)` 写法
4. **理解** 为什么 Triton 不用手写共享内存与 `__syncthreads()` 也能高效

---

## 前置依赖

- **Day 2 向量加法**：本天的 Triton 向量加法与之一一对照
- **Day 31 Flash Attention / common Softmax kernel**：本天的 Triton Softmax 对照其 CUDA 版
- **Day 9 归约**：Softmax 里的 `tl.max` / `tl.sum` 本质是块内归约，理解 Day9 有助于体会 Triton 帮你做了什么

---

## 核心概念

### Triton 编程模型：以「块」为思考单位

CUDA 让你思考**单个线程**处理**单个元素**；Triton 让你思考**一个 program**处理**一整块（BLOCK_SIZE 个）元素**。

```
CUDA（每线程一元素）                 Triton（每 program 一块）
  tid = blockIdx.x*blockDim.x          pid = tl.program_id(0)
      + threadIdx.x                    offs = pid*BLOCK + arange(0,BLOCK)
  if (tid < n)                         mask = offs < n
      c[tid] = a[tid] + b[tid];        x = tl.load(a+offs, mask)
                                       tl.store(c+offs, x+y, mask)
```

| 概念 | CUDA | Triton |
|------|------|--------|
| 并行单位 | thread（单元素） | program（数据块） |
| 块编号 | `blockIdx.x` | `tl.program_id(0)` |
| 块内下标 | `threadIdx.x` | `tl.arange(0, BLOCK)` |
| 边界处理 | `if (tid < n)` | `mask = offsets < n` |
| 访存 | 指针 + 手动索引 | `tl.load / tl.store(..., mask=)` |
| 共享内存 | 显式 `__shared__` + `__syncthreads()` | 编译器自动管理片上缓存 |
| 启动 | `kernel<<<grid, block>>>()` | `kernel[grid](..., BLOCK=...)` |

### 融合思想（本阶段主线）

Softmax kernel 里，**一次把整行读入片上**，减最大值、`exp`、求和、相除全部在片上完成，只写回一次。这就是「算子融合」——减少全局内存往返，是后续 Fused LayerNorm / Flash Attention 的基础。

---

## 文件结构

```
day41_triton_intro/
├── python/triton_intro.py   # Triton 向量加法 + Softmax，含无 GPU 原理演示
├── build.sh                 # Linux/WSL2 运行脚本
├── build.bat                # Windows 运行脚本（提示用 WSL2）
└── README.md                # 本文件
```

> 本阶段以 Triton（Python）为主，不再有 `src/main.cu`。Triton 官方主要支持 **Linux**，Windows 用户建议 **WSL2 + NVIDIA GPU**。

---

## 编译与运行

### Linux / WSL2（推荐，有 NVIDIA GPU）

```bash
cd day41_triton_intro
pip install triton torch    # 需 NVIDIA GPU
python python/triton_intro.py
```

### Windows

```powershell
cd day41_triton_intro
.\build.bat
```

> Windows 原生对 Triton 支持有限。脚本会检测环境：有 GPU+Triton 则跑真实 kernel，否则用 NumPy 演示等价原理（不伪造结果）。

---

## 知识点总结

1. **program vs thread**：Triton 的一个 program 处理一整块数据，粒度比 CUDA thread 粗。
2. **mask 机制**：`offsets < n_elements` 一次性处理越界，替代 CUDA 的分支边界检查。
3. **`tl.load / tl.store`**：带 mask 的批量访存，编译器自动向量化。
4. **块内归约**：`tl.max` / `tl.sum` 由编译器生成高效归约，无需手写。
5. **自动片上管理**：无需手写 `__shared__` 与 `__syncthreads()`，降低出错率。
6. **grid 配置**：用 `triton.cdiv(n, BLOCK)` 向上取整算 program 数量。

---

## 验收标准

- 程序打印 CUDA↔Triton 对照表，你能逐行解释每个概念的对应关系。
- 有 GPU 时：向量加法与 Softmax 的「与 PyTorch 最大误差」分别 < 1e-5 / 1e-4，显示「通过 ✓」。
- 无 GPU 时：NumPy 演示中 Softmax 每行和≈1，你能对照 `softmax_kernel` 讲清片上融合流程。
- 你能说出 Triton 相比 CUDA C++ 在写融合算子时的两点优势。

---

## 动手挑战

1. **改 BLOCK_SIZE**：把向量加法的 `BLOCK_SIZE` 从 1024 改为 256 / 512 / 2048，观察结果是否仍正确，思考它对 program 数量的影响。
2. **加一个 Triton kernel**：仿照 `add_kernel` 写一个逐元素乘法 `mul_kernel`，并与 `x * y` 交叉验证。
3. **融合一步**：把 Softmax kernel 改成「Softmax 后再乘以一个缩放系数 scale」，在片上一并完成，体会融合减少了几次全局访存。

---

## 常见问题

- **Q：Windows 能用 Triton 吗？**
  A：官方主要支持 Linux。Windows 建议用 WSL2；本天脚本在无 Triton 时会用 NumPy 演示原理，不影响你理解逻辑。

- **Q：Triton 一定比手写 CUDA 慢吗？**
  A：不一定。Triton 编译器的自动调优（下一天 Day42 会讲）常能达到接近甚至超过朴素手写 CUDA 的性能，而开发成本低得多。

---

## 延伸阅读

- [Triton 官方教程 — Vector Add / Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [OpenAI Triton 介绍博客](https://openai.com/research/triton)
