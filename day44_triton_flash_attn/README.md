# Day 44: Triton Flash Attention（tiled + online softmax）

> 🧩 **第九阶段（现代算子编程）第 4 天**，也是本阶段的“集大成”。Day31 你用 **CUDA C++ 手写**了 Flash Attention（共享内存、`__syncthreads()`、online softmax 回溯、`float4` 向量化，数百行）；本天用 **Triton 重写**同一算法，核心逻辑只需约几十行，直观感受 Triton 在写复杂融合算子上的简洁性。

## 学习目标

1. **实现** Triton Flash Attention 前向：tiled 遍历 K/V + online softmax，**从不构造完整 N×N 分数矩阵**
2. **理解** online softmax 三个运行状态 `m`（running max）/ `l`（running sum）/ `acc`（输出累加器）如何增量更新
3. **掌握** `alpha = exp(m_old - m_new)` 回溯修正旧状态的数值稳定技巧
4. **验证** 与 PyTorch 标准注意力交叉验证
5. **对照** Day31 CUDA 手写版，量化“简洁性”差异

---

## 前置依赖

- **Day 31 Flash Attention（CUDA 版）**：本天与之逐点对照，务必先理解 online softmax 原理
- **Day 42 Triton GEMM**：`tl.dot` 分块矩阵乘是本天算 `Q·K^T` 与 `P·V` 的基础
- **Day 43 Fused Softmax/LayerNorm**：片上归约（`tl.max`/`tl.sum`）是 online softmax 的组成部分

---

## 核心概念

### 标准注意力 vs Flash Attention

```
  标准注意力（O(N²) 显存）：
    Q[N,d] × K^T ─▶ S[N,N] ─▶ softmax ─▶ ×V ─▶ O[N,d]
                    ^^^^^^ 完整分数矩阵，N 越大平方增长 → 显存瓶颈

  Flash Attention（O(N) 显存）：每个 Q 行维护 m / l / acc，沿 K/V 分块遍历：
    for 每个 K/V tile:
        S_blk = Q · K_tile^T * scale        # 只算一小块，绝不存整行
        m_new = max(m, rowmax(S_blk))
        alpha = exp(m - m_new)              # 回溯修正旧的 l 与 acc
        p     = exp(S_blk - m_new)
        l     = l*alpha + rowsum(p)
        acc   = acc*alpha + p · V_tile
    O = acc / l                             # 收尾归一化
```

### online softmax 为什么正确

softmax 需要“全局最大值”做数值稳定，但分块遍历时还没见到所有分数。技巧是：
维护当前 running max `m`，每来一个新块就更新 `m_new`；由于指数底数变了，用
`alpha = exp(m_old - m_new)` 把**之前累计的 `l` 和 `acc` 一并按比例缩放**，
等价于事后统一用 `m_new` 重算——数学上与标准 softmax 完全等价。

### Day31 CUDA 手写版 vs Triton 版

| 项目 | CUDA (Day31) | Triton (本天) |
|------|--------------|---------------|
| K/V 分块缓存 | 手写 `__shared__` + 手动加载 | `tl.load`（编译器管片上） |
| 同步 | 手写 `__syncthreads()` | 自动 |
| m/l 回溯修正 | 手写标量循环维护 | 向量化，几行 |
| 向量化访存 | 手写 `float4`（128 bit） | 编译器自动向量化 |
| 边界处理 | 手写 `if` | `mask` / `tl.where` |
| 代码量 | 数百行 | 约几十行 |

> 算法（online softmax）完全一样，Triton 把访存/同步/向量化的“脏活”交给编译器。

---

## 文件结构

```
day44_triton_flash_attn/
├── python/triton_flash_attn.py  # Triton Flash Attention，含无 GPU 原理演示
├── build.sh                     # Linux/WSL2 运行脚本
├── build.bat                    # Windows 运行脚本
└── README.md                    # 本文件
```

> 本天为教学简化版（单头、无 causal mask、无反向），聚焦讲清 tiled + online softmax。
> Triton 官方主要支持 **Linux**，Windows 用户建议 **WSL2 + NVIDIA GPU**。

---

## 编译与运行

### Linux / WSL2（推荐，有 NVIDIA GPU）

```bash
cd day44_triton_flash_attn
pip install triton torch    # 需 NVIDIA GPU
python python/triton_flash_attn.py
```

### Windows

```powershell
cd day44_triton_flash_attn
.\build.bat
```

> 有 GPU+Triton 则跑真实 Flash Attention 并与 PyTorch 标准注意力交叉验证；
> 否则用 NumPy 同时跑「标准注意力」与「online softmax 分块」两条路径，
> 验证二者数学等价（误差≈0），不伪造结果。

---

## 知识点总结

1. **分块遍历 K/V**：每个 program 负责一块 Q 行，沿 N 维分块，`BLOCK_M×BLOCK_N` 的小分数块即用即弃。
2. **三个运行状态**：`m`（max）、`l`（sum）、`acc`（加权 V 累加器）构成 online softmax 的全部记忆。
3. **回溯修正**：`alpha = exp(m_old - m_new)` 缩放旧的 `l`/`acc`，保证与全局 softmax 等价。
4. **O(N) 显存**：从不物化 N×N 分数矩阵，是长序列注意力可行的关键。
5. **`tl.dot` 复用**：`Q·K^T` 和 `P·V` 都用块级矩阵乘，可走 Tensor Core。
6. **Triton 简洁性**：同一算法比 CUDA 手写少一个数量级的代码，且不易出错。

---

## 验收标准

- 程序打印标准注意力 vs Flash Attention 原理图与 CUDA↔Triton 对照表。
- 有 GPU 时：Triton Flash Attention 与 PyTorch 标准注意力最大误差在 fp16 合理范围（< 1e-2），显示「通过 ✓」。
- 无 GPU 时：NumPy 的 online softmax 分块结果与标准注意力误差≈0，你能解释为何二者等价。
- 你能说出 online softmax 的 `alpha` 回溯修正在做什么，以及 Flash Attention 为何是 O(N) 显存。
- 你能对照 Day31 指出 Triton 版省去了哪些手写代码。

---

## 动手挑战

1. **causal mask**：在 kernel 里加因果掩码（只允许 query 看到 `<=` 自己位置的 key），验证与 `torch` 的 causal 注意力一致。
2. **多头/批处理**：把 grid 扩展到 `(cdiv(N,BLOCK_M), num_heads*batch)`，用额外的 program 维处理多头。
3. **调 BLOCK**：改 `BLOCK_M`/`BLOCK_N`，或加 `@triton.autotune`（回顾 Day42）自动择优。
4. **性能对比**：用 `triton.testing.do_bench` 对比 Triton Flash Attention 与 `F.scaled_dot_product_attention` 在长序列下的耗时/显存。

---

## 常见问题

- **Q：fp16 误差为什么比 LayerNorm 大？**
  A：注意力涉及两次矩阵乘 + softmax，fp16 累加误差叠加；kernel 内部用 fp32 accumulator 已尽量稳，误差在 < 1e-2 属正常。

- **Q：为什么要 `tl.where(n_mask, s, -inf)`？**
  A：最后一个 K/V tile 可能有 padding 列，把越界分数设为 `-inf`，`exp(-inf)=0`，不影响 softmax。

- **Q：和 Day31 CUDA 版结果会完全一样吗？**
  A：数学等价，但浮点累加顺序不同 + 精度不同，会有微小差异；只要与标准注意力一致即正确。

---

## 延伸阅读

- [Triton 官方教程 — Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [FlashAttention 论文（Dao et al., 2022）](https://arxiv.org/abs/2205.14135)
- 对照本仓库 Day31（CUDA 手写 Flash Attention）
