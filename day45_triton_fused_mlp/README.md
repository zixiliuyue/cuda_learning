# Day 45: 算子融合进阶 — Fused MLP / GELU+Linear

> 🧩 **第九阶段（现代算子编程）**。Day43 学了 Fused Softmax/LayerNorm 的融合思想，本天把融合用到 Transformer FFN 最常见的一步 `y = GELU(x @ W + b)`，把「矩阵乘 → 加 bias → GELU」压进一个 Triton kernel，减少中间张量落地。

## 学习目标

1. **理解** Transformer FFN（前馈网络）中的融合机会：`Linear + Bias + GELU`
2. **实现** Triton 版 Fused GEMM+Bias+GELU kernel，与 PyTorch 分步计算交叉验证
3. **说清** 融合后省掉了几个中间张量、减少了几次显存往返
4. **掌握** 在 GEMM 结果块仍在片上寄存器时，直接做逐元素后处理的写法

---

## 前置依赖

- **Day 42 Triton 矩阵乘**：本天的融合基于分块 GEMM（`tl.dot` + 沿 K 累加）
- **Day 43 Fused Softmax / LayerNorm**：本天延续「一次读入、片上完成多步」的融合思想
- **Day 6 Tiling**：分块矩阵乘的 tiling 思路与之一脉相承

---

## 核心概念

### FFN 中的融合机会

Transformer 的前馈网络（FFN）核心是两次线性 + 一次激活：

```
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
                └────────────┘
              本天聚焦的一步：Linear + Bias + GELU
```

### 融合前 vs 融合后

朴素 PyTorch 分步会产生多个 `[M,N]` 中间张量，每个都要写回再读出：

```
【朴素分步】                              显存往返（针对 [M,N]）
  tmp1 = x @ W        写 tmp1             1 写
  tmp2 = tmp1 + b     读 tmp1 / 写 tmp2   1 读 + 1 写
  y    = GELU(tmp2)   读 tmp2 / 写 y      1 读 + 1 写
  ── 额外中间张量：tmp1, tmp2（2 个 [M,N]）

【Triton 融合】
  x@W 结果块仍在片上 → +b → GELU → 只写回 y
  ── 额外中间张量：0 个；针对 [M,N] 只有 1 次写
```

| 维度 | 朴素分步 | Triton 融合 |
|------|----------|-------------|
| 额外中间张量 | tmp1、tmp2 两个 | 0 个 |
| `[M,N]` 显存写 | ~3 次（tmp1/tmp2/y） | 1 次（y） |
| `[M,N]` 显存读 | ~2 次（tmp1/tmp2） | 0 次 |
| kernel 启动 | 3 次 | 1 次 |

> **为什么值得**：`+bias` 和 `GELU` 都是访存受限（memory-bound）操作，本身算得很快，瓶颈全在显存往返。把它们融进 GEMM 后几乎「免费」，张量越大收益越明显。

### GELU 的 tanh 近似

kernel 内用与 PyTorch `approximate='tanh'` 一致的公式，在片上完成：

```
GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
```

---

## 文件结构

```
day45_triton_fused_mlp/
├── python/triton_fused_mlp.py   # Fused GEMM+Bias+GELU，含无 GPU 原理演示
├── build.sh                     # Linux/WSL2 运行脚本
├── build.bat                    # Windows 运行脚本（提示用 WSL2）
└── README.md                    # 本文件
```

> 本阶段以 Triton（Python）为主。Triton 官方主要支持 **Linux**，Windows 用户建议 **WSL2 + NVIDIA GPU**。

---

## 编译与运行

### Linux / WSL2（推荐，有 NVIDIA GPU）

```bash
cd day45_triton_fused_mlp
pip install triton torch    # 需 NVIDIA GPU
python python/triton_fused_mlp.py
```

### Windows

```powershell
cd day45_triton_fused_mlp
.\build.bat
```

> Windows 原生对 Triton 支持有限。脚本会检测环境：有 GPU+Triton 则跑真实融合 kernel，否则用 NumPy 演示等价原理（不伪造结果）。

---

## 知识点总结

1. **融合三步走**：GEMM 结果块留在片上 → 加 bias → GELU → 只写回一次。
2. **省中间张量**：融合掉 tmp1、tmp2，减少约 2 次写 + 2 次读的显存往返。
3. **memory-bound 后处理**：bias/GELU 融进 compute-bound 的 GEMM 后近乎免费。
4. **片上 GELU**：用 `tl.math.tanh` 实现 tanh 近似，与 PyTorch 数值对齐。
5. **分块累加**：沿 K 维 `BLOCK_K` 分块，用 `tl.dot` 累加到 `acc` 寄存器。

---

## 验收标准

- 程序打印「融合前 vs 融合后」的中间张量与显存往返对照表，你能讲清省了什么。
- 有 GPU 时：融合 kernel 与 PyTorch 分步计算的最大误差 < 1e-3，显示「通过 ✓」。
- 无 GPU 时：NumPy 分步 vs 融合语义误差 < 1e-5（数学等价），你能对照 `fused_mlp_kernel` 讲清片上后处理流程。
- 你能说出为什么 bias/GELU 融进 GEMM 后几乎不增加耗时。

---

## 动手挑战

1. **加 Dropout**：在 GELU 后再融合一个 `Fused Bias+Dropout`（训练态用随机 mask），思考推理态如何跳过。
2. **两层 FFN**：把 `GELU(x@W1+b1) @ W2 + b2` 的第二次 Linear 也接上，观察还能否进一步融合。
3. **换激活**：把 GELU 换成 SiLU / ReLU，只改片上那一行，体会融合框架的复用性。
4. **测收益**：用 `torch.cuda.Event` 对比融合 vs 分步的耗时，验证张量越大融合收益越明显。

---

## 常见问题

- **Q：为什么误差阈值用 1e-3 而不是 1e-5？**
  A：`tl.dot` 在 GPU 上可能用 TF32/累加顺序不同，加上 GELU 的近似，float32 下 1e-3 属正常范围。

- **Q：融合一定更快吗？**
  A：对访存受限的后处理（bias/GELU）几乎总是更快；但若后处理很重（compute-bound），收益会变小。

---

## 延伸阅读

- [Triton 官方教程 — Fused Softmax / Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [PyTorch `torch.nn.functional.gelu`（tanh 近似）文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html)
