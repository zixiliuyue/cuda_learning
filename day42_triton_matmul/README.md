# Day 42: Triton 矩阵乘与自动调优（@triton.autotune）

> 🧩 **第九阶段（现代算子编程）第 2 天**。Day41 学了 Triton 的编程模型，本天写出真正实用的 **分块矩阵乘（GEMM）**，并掌握 Triton 的杀手锏 **`@triton.autotune`**——让编译器自动搜出最优的 BLOCK 配置，把 Day6 手写 Tiled 里的“手动调参”自动化。

## 学习目标

1. **实现** Triton 分块矩阵乘 kernel：每个 program 计算 C 的一个 `BLOCK_M×BLOCK_N` 子块，沿 K 维分块累加
2. **掌握** `@triton.autotune`：给一批候选 `Config`，编译器自动跑真实输入择优并缓存
3. **理解** `tl.dot`、片上 float32 累加器、group-major 重排对 L2 命中的意义
4. **对比** Triton GEMM 与 Day6 手写 Tiled CUDA、Day13 cuBLAS 的开发成本差异
5. **验证** 与 `torch.matmul` 交叉验证正确性

---

## 前置依赖

- **Day 6 Tiled 矩阵乘**：本天 Triton GEMM 与之逐点对照（共享内存 tile ↔ `tl.load`+编译器管理）
- **Day 13 cuBLAS**：作为“性能天花板”参照
- **Day 9 归约**：K 维累加本质是沿收缩维的归约
- **Day 41 Triton 入门**：program/BLOCK/mask 模型是本天基础

---

## 核心概念

### 分块矩阵乘：每个 program 算 C 的一个子块

```
  C[M,N] = A[M,K] @ B[K,N]

      N 维 →                     每个 program 负责 C 的一个
    ┌───┬───┬───┐                BLOCK_M × BLOCK_N 子块：
  M │C00│C01│C02│                沿 K 维循环，一次读入 A 的一条
  维 ├───┼───┼───┤                行块 + B 的一条列块做 tl.dot，
  ↓ │C10│C11│C12│                累加到片上 float32 accumulator。
    └───┴───┴───┘
```

### CUDA Tiled(Day6) ↔ Triton 对照

| 步骤 | CUDA Tiled (Day6) | Triton (本天) |
|------|-------------------|---------------|
| 子块划分 | `blockIdx.{x,y}` | `tl.program_id` + group 重排 |
| 片上缓存 | 手写 `__shared__` A/B tile | `tl.load` 到片上，编译器管理 |
| 同步 | `__syncthreads()` | 自动（`num_stages` 软件流水线） |
| 块内乘加 | 手写三重循环累加 | `tl.dot`（可走 Tensor Core） |
| 边界 | `if` 判断 | `mask` |
| 累加精度 | 手动用 float | `tl.zeros(..., float32)` accumulator |
| 配置搜索 | 手动改 `TILE` 重编译测 | **`@triton.autotune` 自动搜** |

### @triton.autotune 工作机制

```
@triton.autotune(
    configs=[Config({BLOCK_M,BLOCK_N,BLOCK_K,GROUP_M}, num_warps, num_stages), ...],
    key=["M","N","K"],   # 这些参数变化时才重新搜索，否则复用缓存
)
@triton.jit
def matmul_kernel(...): ...
```

- 第一次遇到某组 `(M,N,K)`：把每个候选 `Config` 都真实跑一遍计时，选最快的；
- 之后同样 shape 直接复用缓存的最优配置，**零额外开销**；
- 免去了 CUDA 里“改 TILE→重编译→测速”的手工循环。

### group-major 重排（L2 优化）

朴素的 `pid → (pid_m, pid_n)` 行主序映射会让并发 program 访问分散的 A 行 / B 列。按 `GROUP_M` 分组重排后，相邻 program 复用同一批已在 L2 里的 A/B 数据，提升缓存命中率——这是官方 GEMM 教程的经典技巧。

---

## 文件结构

```
day42_triton_matmul/
├── python/triton_matmul.py  # Triton 分块 GEMM + autotune，含无 GPU 原理演示
├── build.sh                 # Linux/WSL2 运行脚本
├── build.bat                # Windows 运行脚本
└── README.md                # 本文件
```

> 本阶段以 Triton（Python）为主。Triton 官方主要支持 **Linux**，Windows 用户建议 **WSL2 + NVIDIA GPU**。

---

## 编译与运行

### Linux / WSL2（推荐，有 NVIDIA GPU）

```bash
cd day42_triton_matmul
pip install triton torch    # 需 NVIDIA GPU
python python/triton_matmul.py
```

### Windows

```powershell
cd day42_triton_matmul
.\build.bat
```

> 有 GPU+Triton 则跑真实 GEMM+autotune，并打印选中的最优配置；否则用 NumPy 演示“沿 K 维分块累加”的等价原理（不伪造结果）。

---

## 知识点总结

1. **子块并行**：一个 program 负责 C 的一个 `BLOCK_M×BLOCK_N` 块，grid 大小 = `cdiv(M,BM)*cdiv(N,BN)`。
2. **K 维分块累加**：循环里 `tl.load` A/B 的 tile，`tl.dot` 乘加进 float32 accumulator，最后转回目标 dtype。
3. **`tl.dot`**：块级矩阵乘，自动利用 Tensor Core（fp16/bf16 输入、fp32 累加）。
4. **`@triton.autotune`**：候选 Config 自动择优 + 按 `key` 缓存，替代手工调参。
5. **`num_warps` / `num_stages`**：分别控制并行度与软件流水线深度，是重要的调优旋钮。
6. **group-major 重排**：提升 L2 命中，是 GEMM 达到高性能的关键细节。

---

## 验收标准

- 程序打印 CUDA↔Triton GEMM 对照表，你能解释每一行的对应关系。
- 有 GPU 时：Triton GEMM 与 `torch.matmul` 的最大误差在 fp16 合理范围内，显示「通过 ✓」，并打印 autotune 选中的最优 `Config`。
- 无 GPU 时：NumPy 分块累加结果与直接 `a@b` 误差≈0，你能对照 `matmul_kernel` 讲清 K 维分块累加流程。
- 你能说出 `@triton.autotune` 相比 Day6 手工调 TILE 的两点优势。

---

## 动手挑战

1. **改配置**：往 `configs` 里加/删几组 `Config`（如更大的 `BLOCK_K`），观察 `best_config` 是否变化。
2. **换 shape**：把 `(M,N,K)` 改成非 2 的幂（如 `500,700,300`），确认 mask 边界处理正确。
3. **加 bias**：让 kernel 在写回前对 C 子块加一个 bias 向量（融合 `C = A@B + bias`），体会算子融合。
4. **性能对比**：用 `triton.testing.do_bench` 对比 Triton GEMM 与 `torch.matmul(cuBLAS)` 的 TFLOPS。

---

## 常见问题

- **Q：fp16 误差为什么比向量加法大很多？**
  A：GEMM 沿 K 维累加了几百个乘积，fp16 精度低，即便用 fp32 accumulator，与参考实现的舍入路径不同也会有可见误差；只要在合理范围（远小于结果量级）即算正确。

- **Q：第一次运行为什么卡一下？**
  A：`autotune` 在首次遇到新 shape 时会逐个 Config 试跑计时，属正常。相同 shape 后续调用直接复用缓存。

- **Q：Windows 能跑吗？**
  A：官方主要支持 Linux，建议 WSL2；无 Triton 时脚本用 NumPy 演示原理，不影响理解。

---

## 延伸阅读

- [Triton 官方教程 — Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton autotune API 文档](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
- 对照本仓库 Day6（Tiled GEMM）、Day13（cuBLAS）
