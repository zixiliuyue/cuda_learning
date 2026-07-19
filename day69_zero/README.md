# Day 69: ZeRO-1/2/3 显存优化

> 💾 训练大模型的最大瓶颈不是算力而是**显存**。DDP 里每张卡都存一份完整的参数/梯度/优化器状态，冗余惊人。ZeRO（Zero Redundancy Optimizer，DeepSpeed 提出）把这些状态按 rank 切分，让单卡显存随卡数近线性下降。本天以**原理 + 显存对比表**为主。

## 学习目标

1. **拆解** 混合精度 Adam 训练的「每参数 16 字节」显存账本
2. **区分** ZeRO-1 / ZeRO-2 / ZeRO-3 分别切分了什么、省了多少
3. **计算** 给定模型规模与卡数时，各级的单卡显存占用
4. **理解** ZeRO-3 用「参数按需 all_gather」换取最大省显存的通信代价权衡

---

## 前置依赖

- **Day 67 集合通信**：ZeRO 的切分/聚合依赖 `reduce_scatter` 与 `all_gather`
- **Day 68 DDP**：ZeRO 是在 DDP 基础上「消除状态冗余」的增强
- **Adam 优化器**：理解 Adam 需要存 fp32 主权重 + 一阶动量 m + 二阶方差 v

---

## 核心概念

### 每参数 16 字节账本（混合精度 Adam）

设参数量为 Ψ，采用 fp16 计算 + fp32 主权重：

| 部分 | 精度 | 字节/参数 | 可切分？ |
|------|------|-----------|----------|
| 参数 | fp16 | 2 | 仅 ZeRO-3 |
| 梯度 | fp16 | 2 | ZeRO-2/3 |
| 优化器 · fp32 主参数 | fp32 | 4 | ZeRO-1/2/3 |
| 优化器 · 动量 m | fp32 | 4 | ZeRO-1/2/3 |
| 优化器 · 方差 v | fp32 | 4 | ZeRO-1/2/3 |
| **合计** | | **16Ψ** | |

> 优化器状态占 12/16 = **75%**，所以 ZeRO-1 先切它，收益最大。

### 三级切分（ASCII 图）

```
              参数 P      梯度 G      优化器状态 O(=12Ψ)
DDP           完整        完整        完整          → 16Ψ
ZeRO-1        完整        完整        切分 1/N      → 4Ψ + 12Ψ/N
ZeRO-2        完整        切分 1/N    切分 1/N      → 2Ψ + 14Ψ/N
ZeRO-3        切分 1/N    切分 1/N    切分 1/N      → 16Ψ/N
```

### 显存公式

- **DDP**：`16Ψ`（每卡完整）
- **ZeRO-1**：`2Ψ + 2Ψ + 12Ψ/N`
- **ZeRO-2**：`2Ψ + (2Ψ + 12Ψ)/N`
- **ZeRO-3**：`16Ψ/N`（随卡数近线性下降）

经典例子（论文常引）：7.5B 参数，64 卡，ZeRO-3 单卡仅需约 **1.75 GB**（DDP 需 ~112 GB）。

### 通信代价权衡

- **ZeRO-1/2**：通信量与 DDP **同量级**（`reduce_scatter` + `all_gather`），几乎免费省显存。
- **ZeRO-3**：前向、反向都要临时 `all_gather` 参数分片，通信量约 **1.5 倍**，换取最大省显存。

---

## 文件结构

```
day69_zero/
├── python/zero.py    # 状态切分数据结构演示 + 各级单卡显存对比表 + 公式自洽校验
├── build.sh          # Linux/WSL2 运行脚本
├── build.bat         # Windows 运行脚本
└── README.md         # 本文件
```

---

## 编译与运行

### Windows

```powershell
cd day69_zero
.\build.bat
```

### Linux / WSL2

```bash
cd day69_zero
bash build.sh
```

### 真实 ZeRO（DeepSpeed）

```bash
pip install deepspeed
deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json
```

> 本天为纯显存原理计算，不依赖 GPU，`exit code 0`。

---

## 知识点总结

1. **16 字节账本**：fp16 参数 2 + fp16 梯度 2 + 优化器 12（fp32 主参/m/v）。
2. **ZeRO-1 切优化器状态**：最先切、收益最大（占 75%）。
3. **ZeRO-2 再切梯度**：显存进一步下降，通信量不变。
4. **ZeRO-3 再切参数**：单卡显存 → 16Ψ/N，代价是前向/反向 all_gather 参数。
5. **Offload**：ZeRO 还可把状态/参数 offload 到 CPU 内存甚至 NVMe，进一步省显存。
6. **与 TP/PP 正交**：ZeRO 属数据并行家族，可与张量/流水线并行组合成 3D 并行。

---

## 验收标准

- 程序打印 3 组（模型规模 × 卡数）的 DDP/ZeRO-1/2/3 单卡显存对比表。
- 「显存公式自洽」校验显示「通过 ✓」（DDP=16Ψ，ZeRO-3=16Ψ/N）。
- 你能解释为什么 ZeRO-1 先切优化器状态，以及 ZeRO-3 的额外通信代价。
- 程序 `exit code 0`，无卡死。

---

## 动手挑战

1. **换模型规模**：给 `print_table` 传 175（GPT-3 175B），看 DDP 单卡需多少 GB（会爆显存），再看 ZeRO-3 + 更多卡如何降下来。
2. **纯 fp32 训练**：把账本改成 fp32（参数4+梯度4+优化器 8?），重算各级显存。
3. **加激活显存**：本天未计入激活（activation）。查资料补上「激活 ≈ batch×seq×hidden×层数×常数」一项，讨论为什么激活要靠**梯度检查点**（Day 75）而非 ZeRO 来省。
4. **实测 DeepSpeed**：在有 GPU 环境用 stage 1/2/3 各跑一次，用 `nvidia-smi` 观察显存差异。

---

## 常见问题

- **Q：ZeRO 和张量并行（TP）区别？**
  A：ZeRO 属**数据并行**，切的是「状态」，计算逻辑不变；TP 切的是「单个算子的权重与计算」。两者正交可组合。

- **Q：ZeRO-3 会不会很慢？**
  A：会有额外 all_gather 开销，但通过通信-计算重叠（`overlap_comm`）与预取，实际吞吐损失通常可控，换来的是能训练原本放不下的模型。

- **Q：本天为什么没真跑多卡？**
  A：ZeRO 的核心是「显存布局」这一数学/工程问题，用公式即可精确刻画；真实性能需多卡，脚本给出了 DeepSpeed 配置骨架供实操。

---

## 延伸阅读

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054)
- [DeepSpeed ZeRO 官方文档](https://www.deepspeed.ai/tutorials/zero/)
- [ZeRO-Offload / ZeRO-Infinity](https://arxiv.org/abs/2104.07857)
