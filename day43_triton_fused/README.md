# Day 43: Triton 融合算子 — Fused Softmax / LayerNorm

> 🧩 **第九阶段（现代算子编程）第 3 天**。Day41 用 Softmax 初尝“融合”，本天把 **LayerNorm** 写成一个 fused kernel：一次读入整行，片上完成 `mean → var → normalize → scale/shift` 全流程，只写回一次。这正是 Transformer 里最常被融合的算子之一。

## 学习目标

1. **实现** Triton fused layernorm kernel：每个 program 负责一行，片上完成全部计算
2. **理解** “融合”为何能省访存：对比未融合（多 kernel、中间结果落地）与融合（1 次读、1 次写）
3. **掌握** 片上归约求 mean/var、`tl.where` 处理 mask、`rstd = 1/sqrt(var+eps)` 的数值写法
4. **验证** 与 `torch.nn.functional.layer_norm` 交叉验证
5. **对照** 同样思想的 Fused Softmax，体会二者共性

---

## 前置依赖

- **Day 41 Fused Softmax**：本天 LayerNorm 沿用“每个 program 一行 + 片上归约”的结构
- **Day 9 归约**：mean/var 本质是行内求和归约，`tl.sum` 由编译器生成高效实现
- **common/ 的 CUDA Softmax**：作为“手写 CUDA 融合”的对照

---

## 核心概念

### 融合思想：LayerNorm 为例

```
  未融合（eager 典型多 kernel）：
    x ─▶[求mean]─▶[求var]─▶[归一化]─▶[仿射]─▶ y
        每步都在全局内存来回读写中间结果，访存往返多。

  融合（Triton，一个 kernel 一行）：
    x ─▶┌──────────────────────────────────┐─▶ y
        │ 读入整行到片上                     │
        │  mean → var → (x-mean)*rstd*w + b  │  全在片上完成
        │ 只写回一次                         │
        └──────────────────────────────────┘
```

| 项目 | 未融合(多 kernel) | Triton Fused |
|------|------------------|--------------|
| 全局读 | 多次 | 整行 1 次 |
| 全局写 | 多次(中间落地) | 结果 1 次 |
| kernel launch | 多次 | 1 次 |
| 片上复用 | 无(每步重读) | mean/var/归一化共享同一份片上数据 |

### LayerNorm 公式

```
mean = Σx / N
var  = Σ(x - mean)^2 / N        # 有偏方差，与 torch / kernel 一致
y    = (x - mean) / sqrt(var + eps) * weight + bias
```

### 关键实现细节

- **一次读入整行**：`BLOCK_SIZE = next_power_of_2(n_cols)`，`mask = cols < n_cols` 处理越界。
- **mask 修正统计**：越界位置 `other=0.0` 读入，但用 `tl.where(mask, x-mean, 0.0)` 保证不污染方差；除数用真实 `n_cols`。
- **float32 计算**：即使输入 fp16，也在片上转 fp32 求统计，保证数值稳定。
- **只写回一次**：`tl.store(y_start + cols, y, mask=mask)`。

---

## 文件结构

```
day43_triton_fused/
├── python/triton_fused.py   # Fused LayerNorm + Fused Softmax，含无 GPU 原理演示
├── build.sh                 # Linux/WSL2 运行脚本
├── build.bat                # Windows 运行脚本
└── README.md                # 本文件
```

> 本阶段以 Triton（Python）为主。Triton 官方主要支持 **Linux**，Windows 用户建议 **WSL2 + NVIDIA GPU**。

---

## 编译与运行

### Linux / WSL2（推荐，有 NVIDIA GPU）

```bash
cd day43_triton_fused
pip install triton torch    # 需 NVIDIA GPU
python python/triton_fused.py
```

### Windows

```powershell
cd day43_triton_fused
.\build.bat
```

> 有 GPU+Triton 则跑真实 fused kernel 并与 PyTorch 交叉验证；否则用 NumPy 演示等价原理（不伪造结果）。

---

## 知识点总结

1. **每个 program 一行**：行内所有计算共享同一份片上数据，天然适合融合。
2. **片上归约求统计**：`tl.sum` 求和 → mean/var，无需手写归约。
3. **mask 的双重作用**：既控制越界访存，也用 `tl.where` 保证统计量不被 padding 污染。
4. **数值稳定**：fp32 累加、`var + eps` 再开方，避免除零与精度损失。
5. **一次读一次写**：融合的本质收益是减少全局内存往返与 kernel launch 次数。
6. **通用模式**：Softmax / LayerNorm / RMSNorm 都能套用“一行一 program + 片上多步”模板。

---

## 验收标准

- 程序打印“未融合 vs 融合”对照图与对照表，你能解释融合省了哪些访存。
- 有 GPU 时：fused layernorm 与 `F.layer_norm` 最大误差 < 1e-3，fused softmax 与 `torch.softmax` < 1e-4，均显示「通过 ✓」。
- 无 GPU 时：NumPy 演示中归一化后每行均值≈0、方差≈1，softmax 每行和≈1，你能对照 `layernorm_kernel` 讲清片上融合流程。
- 你能说出“为什么融合能减少访存往返”的两点理由。

---

## 动手挑战

1. **RMSNorm**：仿照 layernorm kernel 写一个 RMSNorm（只除以 `sqrt(mean(x^2)+eps)`，无减均值），与手写参考对照。
2. **反向传播**：查资料实现 layernorm 的反向 kernel（对 x、weight、bias 的梯度），体会为何反向更复杂。
3. **融合更多**：把 LayerNorm 后再接一个逐元素激活（如 GELU）融进同一 kernel，减少一次全局往返。
4. **性能测量**：用 `triton.testing.do_bench` 对比 fused layernorm 与 `F.layer_norm` 的带宽利用率。

---

## 常见问题

- **Q：方差为什么用有偏（除 N 而非 N-1）？**
  A：LayerNorm 定义与 PyTorch 实现都用有偏方差（除以 N），这样才能与 `F.layer_norm` 对齐。

- **Q：n_cols 不是 2 的幂会怎样？**
  A：`BLOCK_SIZE` 取 `next_power_of_2(n_cols)`，多出的列由 mask 屏蔽，不影响结果。

- **Q：非常大的行（如 n_cols 上万）能一次读入吗？**
  A：单 program 片上容量有限，超大行需分块归约（多趟）。本例针对常见 hidden size（几百~几千）。

---

## 延伸阅读

- [Triton 官方教程 — Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton 官方教程 — LayerNorm](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- 对照本仓库 Day41（Fused Softmax）、common/ 的 CUDA Softmax
