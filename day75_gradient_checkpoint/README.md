# Day 75: 梯度检查点 / 激活重计算（Gradient Checkpointing）

> 💾 **分布式训练之显存换算力**。反向传播要用前向激活；层数一多，激活显存就爆了。梯度检查点前向只存少量「检查点」，反向时从检查点重新前向「重算」中间激活——用约 +33% 计算换来激活显存从 O(N) 降到 O(√N)。本天手写实现并与 `torch.utils.checkpoint` 对拍。

## 学习目标

1. **解释** 为什么反向传播需要前向激活，默认显存为何是 O(N)
2. **实现** 手写「存全部激活」vs「梯度检查点（重算）」两种前向/反向
3. **验证** 两种方式的前向输出与梯度完全一致
4. **量化** 显存节省与计算开销的权衡，理解 seg_len ≈ √N 最优

---

## 前置依赖

- **Day 26 AMP 混合精度**：另一种省显存手段，可与检查点叠加
- **Day 40 PyTorch CUDA 扩展 / 反向传播**：理解前向激活如何用于反向
- **Day 71-74**：并行策略省的是「跨卡」显存，检查点省的是「单卡内」激活显存，互补

---

## 核心概念

### 为什么默认显存是 O(N)

对 `y = f(x @ W)`，反向需要前向的输入 `x`（或输出 `y`）来算梯度。N 层网络默认把每层激活都存下 → 显存 O(N)。

```
前向: x0 →[L0]→ x1 →[L1]→ x2 → ... → xN   (全部 x_i 驻留显存)
反向: 直接用存好的 x_i
```

### 梯度检查点：只存检查点，反向重算

```
前向: 每 seg_len 层设一个检查点，只存段边界激活，段内丢弃
反向: 到某段时，从最近检查点「重新前向」恢复段内激活，再反向
```

- **显存**：O(N) → O(√N)（取 seg_len ≈ √N 时）
- **算力**：段内多做一次前向，总计算约 +33%

### seg_len 的权衡

| seg_len | 检查点数 | 激活显存 | 重算开销 |
|---------|---------|---------|---------|
| 1 | N | 最大（几乎不省） | 无 |
| √N | √N | **最优** | 适中 |
| N | 1 | 最小 | 最大（整网重算） |

---

## 文件结构

```
day75_gradient_checkpoint/
├── python/gradient_checkpoint.py   # 手写 存激活 vs 重计算 + torch.checkpoint 对照 + 显存统计
├── build.sh
├── build.bat
└── README.md
```

> 显存用「保存的激活元素总数」作为可复现代理指标（不依赖真实显存计量）。有 torch 时额外用 `torch.utils.checkpoint` 做真实重计算对拍梯度；无 torch 用 numpy 手写，同样跑通。

---

## 编译与运行

### Linux / WSL2

```bash
cd day75_gradient_checkpoint
python python/gradient_checkpoint.py
```

### Windows

```powershell
cd day75_gradient_checkpoint
.\build.bat
```

---

## 知识点总结

1. **激活显存**：反向依赖前向激活，默认存全部 → O(N)。
2. **重计算**：前向只存检查点，反向从检查点重新前向恢复段内激活。
3. **显存-算力权衡**：显存降到 O(√N)，计算约 +33%。
4. **seg_len ≈ √N** 时显存-算力最优。
5. **等价性**：重计算与存全部激活得到完全相同的梯度（误差 0）。
6. **torch.utils.checkpoint**：`use_reentrant=False` 是推荐用法，与手写结果一致。

---

## 验收标准

- 「存全部」与「检查点」两种方式的前向输出与梯度误差 < 1e-4（本例为 0）。
- 显存统计显示激活显存显著下降（本例 3072 → 768，省 75%）。
- seg_len 扫描表体现「间隔越大越省显存」的单调趋势。
- 有 torch 时，`torch.utils.checkpoint` 梯度与手写一致，显示「通过 ✓」。

---

## 动手挑战

1. **改层数**：把 `num_layers` 改为 36，取 seg_len = 6 (≈√36)，验证显存-算力最优点。
2. **测真实显存**：有 GPU 时用 `torch.cuda.max_memory_allocated()` 对比开/关检查点的真实峰值显存。
3. **嵌套检查点**：对已用检查点的段再做一层检查点，观察显存能否进一步降到 O(logN) 量级。

---

## 常见问题

- **Q：重算会不会导致梯度不准？**
  A：不会。重算得到的激活与原前向完全相同（本例误差 0），梯度也完全一致。

- **Q：检查点一定要开吗？**
  A：不一定。显存够用时开它反而拖慢训练。它是「显存受限」时才划算的换算力手段。

- **Q：为什么 torch 建议 `use_reentrant=False`？**
  A：新版实现对 RNG、无输入梯度等边界情况更健壮，是官方推荐默认。

---

## 延伸阅读

- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html)
- [DeepSpeed Activation Checkpointing](https://www.deepspeed.ai/tutorials/megatron/)
