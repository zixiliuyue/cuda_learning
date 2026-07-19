# Day 70: 张量并行（Tensor Parallel, Megatron-LM 风格）

> 🧩 当**单层权重**大到单卡放不下（如巨大的 FFN / Attention 投影），数据并行也救不了——因为每卡仍要存完整一层。张量并行（TP）把**一个算子的计算本身**切到多卡协作完成。本天手写 Column / Row Parallel Linear，并验证结果与单卡完全一致。

## 学习目标

1. **实现** Column Parallel Linear（按输出维切 W，前向 `all_gather`）
2. **实现** Row Parallel Linear（按输入维切 W 与 X，前向 `all_reduce`）
3. **组合** Megatron MLP：`Column → GELU → Row`，理解两层间为何无需通信
4. **验证** 张量并行结果 == 单卡完整 Linear/MLP 结果（误差 < 1e-9）

---

## 前置依赖

- **Day 67 集合通信**：Column 用 `all_gather`，Row 用 `all_reduce`
- **Day 68/69 数据并行 / ZeRO**：TP 与它们正交，可组合成 3D 并行
- **矩阵乘与线性层**：`Y = X @ W^T + b`，W 形状 `(out_dim, in_dim)`

---

## 核心概念

### Column Parallel Linear（按输出维/列切）

把 `W`（`out_dim × in_dim`）按**行（即输出维）**切成 N 份，每 rank 算出 `Y` 的一段列：

```
W = [W0]    rank0: Y0 = X @ W0^T   (Y 的前半列)
    [W1]    rank1: Y1 = X @ W1^T   (Y 的后半列)

  X ─┬─> [rank0 W0] ─> Y0 ┐
     └─> [rank1 W1] ─> Y1 ┴─ all_gather(按列拼接) ─> 完整 Y
```

- **前向**：`all_gather` 沿特征维拼接。
- **反向**：对输入 X 的梯度需要 `all_reduce`（求和）。

### Row Parallel Linear（按输入维/行切）

把 `W` 按**列（输入维）**切、`X` 也按列切，每 rank 算一个**部分和**：

```
X = [X0 | X1],  W = [W0 | W1]
  X0 ─> [rank0 W0] ─> Y0 (部分和) ┐
  X1 ─> [rank1 W1] ─> Y1 (部分和) ┴─ all_reduce(SUM) ─> 完整 Y
```

- **前向**：`all_reduce(SUM)` 把部分和相加。
- **偏置 b**：只在聚合后加**一次**（否则会被加 N 次）。

### Megatron MLP：Column → GELU → Row（精髓）

```
X ──Column Parallel──> 各 rank 拿到隐藏维的一段分片
                       │（本地直接做 GELU，无需通信！）
                       ▼
   ──Row Parallel───> 各 rank 部分和 ──all_reduce──> Y
```

**关键洞察**：第一层 Column Parallel 的输出，正好是隐藏维的一个分片；第二层 Row Parallel 的输入也按隐藏维切分。二者**天然对齐**，所以两层之间**无需任何通信**，整个 MLP 仅末尾 **1 次 all_reduce**。这就是 Megatron-LM 把通信降到最低的设计精髓。

---

## 文件结构

```
day70_tensor_parallel/
├── python/tensor_parallel.py   # Column/Row Parallel Linear + Megatron MLP + 正确性验证
├── build.sh                    # Linux/WSL2 运行脚本
├── build.bat                   # Windows 运行脚本
└── README.md                   # 本文件
```

---

## 编译与运行

### Windows

```powershell
cd day70_tensor_parallel
.\build.bat
```

### Linux / WSL2

```bash
cd day70_tensor_parallel
bash build.sh
```

### 真实张量并行（2 卡）

```bash
torchrun --nproc_per_node=2 tp.py   # 参考脚本末尾骨架
```

> 有 torch 用 torch 生成数据 + numpy 稳定计算，无 torch 纯 numpy，均 `exit code 0`、稳定退出。

---

## 知识点总结

1. **Column Parallel**：切输出维 → 前向 `all_gather`。
2. **Row Parallel**：切输入维（W 与 X 同切）→ 前向 `all_reduce(SUM)`。
3. **偏置处理**：Row Parallel 的 bias 聚合后加一次，别重复加。
4. **Megatron MLP**：Column→Row 组合让中间无通信，全 MLP 仅 1 次 all_reduce。
5. **前向/反向对偶**：Column 前向 all_gather、反向 all_reduce；Row 前向 all_reduce、反向 all_gather（互为对偶）。
6. **与 DP/ZeRO 正交**：TP 切「算子内计算」，DP 切「数据」，ZeRO 切「状态」，三者组合即 3D 并行（Day 74）。

---

## 验收标准

- 验证 1/2/3 的最大误差均 < 1e-9，显示「通过 ✓」，总校验「全部通过 ✓」。
- 你能画出 Column 与 Row Parallel 的数据流与通信原语。
- 你能解释 Megatron MLP 中两层为何无需通信。
- 程序 `exit code 0`，无卡死。

---

## 动手挑战

1. **改 world_size**：把 `world_size=2` 改为 4，确保 `out_dim` / `in_dim` / `hidden` 能被整除，验证仍全部通过。
2. **反向通信**：为 Column Parallel 补上「对输入 X 的梯度需 all_reduce」这一步，用数值梯度验证。
3. **Attention 并行**：仿照 MLP，把多头注意力的 QKV 投影做成 Column Parallel、输出投影做成 Row Parallel（每个 head 天然是一列分片）。
4. **真跑 2 卡**：在 Linux/WSL2 用 `torchrun --nproc_per_node=2`（gloo 也行）跑起脚本末尾骨架。

---

## 常见问题

- **Q：为什么 Column 用 all_gather 而 Row 用 all_reduce？**
  A：Column 每卡算出**不同的列**（互补），拼起来即完整 → all_gather；Row 每卡算出**同形状的部分和**，相加才完整 → all_reduce。

- **Q：张量并行能跨机吗？**
  A：能但不建议。TP 通信频繁且在关键路径上，通常限制在**单机多卡**（NVLink 高带宽）内；跨机更适合流水线并行（Day 71）。

- **Q：本天为什么用 numpy 计算即使有 torch？**
  A：为保证跨平台稳定、避免多进程卡死，统一用 numpy float64 走确定性逻辑；数据可由 torch 生成，切分/通信语义与真实 TP 完全一致。

---

## 延伸阅读

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Reducing Activation Recomputation in Large Transformer Models (Sequence Parallel)](https://arxiv.org/abs/2205.05198)
