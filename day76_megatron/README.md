# Day 76: 认识 Megatron-LM 架构

> 🌐 **第十二阶段（分布式训练）**。Megatron-LM 是大模型训练的事实标准之一，本天聚焦其灵魂——**张量并行（Tensor Parallel, TP）**：把 Transformer 层里的巨型权重矩阵按行/列切到多张 GPU，用极少的通信换来显存与算力的横向扩展。

## 学习目标

1. **解释** Megatron 如何切分 MLP（列并行 → 行并行）与 MHA（QKV 按 head 列并行、输出投影 Wo 行并行）
2. **理解** 前向/反向的两个通信算子 `f` 与 `g`，以及「一层 TP 块只需 1 次 all-reduce」的通信优势
3. **验证** 张量并行与不切分的完整计算在**数值上完全等价**（单进程模拟多 rank）
4. **说清** 为什么切 head 而非切 hidden 内部，能保持注意力的数学结构

---

## 前置依赖

- **Day 70 张量并行**：本天是其在真实框架 Megatron 中的落地形态
- **Day 39 NCCL AllReduce**：TP 的核心通信原语就是 all-reduce
- **Day 48 Transformer Decoder / Day 31 Flash Attention**：需理解 MHA / MLP 的结构

---

## 核心概念

### MLP 的切法：列并行 → 行并行

`Y = GELU(X·A)·B`，其中 `A: [h,4h]`，`B: [4h,h]`。

- **A 列并行**（沿输出维 4h 切）：每个 rank 只算 `4h/world` 列。GELU 是逐元素运算，切开后各算各的仍正确，**无需通信**。
- **B 行并行**（沿输入维 4h 切）：每个 rank 得到一个「部分和」，块末做**一次 all-reduce** 求和拼回。

列并行的输出可直接喂给行并行的输入，**中间零通信**。

### MHA 的切法：按 head 切

`Q=X·Wq, K=X·Wk, V=X·Wv; O=Attention(Q,K,V)·Wo`。

- **Wq/Wk/Wv 列并行**：按 **head** 切，每个 rank 拿 `nheads/world` 个头，完整算自己那些头的注意力（注意力天然按头独立）。
- **Wo 行并行**：各 rank 出部分和，最后 all-reduce 求和。

### 通信算子 f 与 g

| 算子 | 前向 | 反向 |
|------|------|------|
| `f`（输入侧） | 恒等/复制 | all-reduce 输入梯度 |
| `g`（输出侧） | all-reduce 部分和 | 恒等 |

**口诀**：一层 TP 块 = 前向 1 次 all-reduce(g) + 反向 1 次 all-reduce(f)。

---

## 文件结构

```
day76_megatron/
├── python/megatron.py   # MLP/MHA 张量并行的数值等价验证（单进程模拟多 rank）
├── build.sh             # Linux/WSL2 运行脚本
├── build.bat            # Windows 运行脚本
└── README.md            # 本文件
```

> 本天为原理与数值验证，统一用 **numpy 单进程模拟** `world_size` 个 rank，把权重真的切开分别计算再手动 all-reduce。单机单卡起 `torch.distributed + mp.spawn` 易卡死，而 TP 的数学等价性用单进程即可完整复现。

---

## 编译与运行

### Linux / WSL2

```bash
cd day76_megatron
python python/megatron.py
```

### Windows

```powershell
cd day76_megatron
.\build.bat
```

> 脚本在任何环境（有无 GPU）下都能稳定运行并以 exit code 0 退出。

---

## 知识点总结

1. **列并行 / 行并行**：按输出维切=列并行，按输入维切=行并行。
2. **组合零通信**：列并行接行并行，中间不通信，块末一次 all-reduce。
3. **按 head 切注意力**：保持数学结构不被破坏。
4. **f/g 算子**：TP 的前向/反向各只有一次 all-reduce，通信极省。
5. **数值等价**：切分并行与完整计算的误差在浮点精度级（~1e-15）。

---

## 验收标准

- MLP 演示中「与完整计算的最大误差」< 1e-9，显示「等价 ✓」。
- MHA 演示中「与完整 MHA 的最大误差」< 1e-9，显示「等价 ✓」。
- 你能画出 f/g 两个通信点，并说出各自前向/反向做什么。
- 你能解释为什么列并行 → 行并行的组合中间不需要通信。

---

## 动手挑战

1. **改 world_size**：把 MLP 的 `world_size` 改成 8（需保证 4h 可整除），观察误差是否仍在浮点级。
2. **加偏置**：给列并行 Linear 加上 bias，思考 bias 该放在哪个 rank、行并行 all-reduce 后会不会被重复累加。
3. **反向推演**：手写一遍 MLP 反向传播，标出 f 算子在哪一步做 all-reduce。

---

## 常见问题

- **Q：为什么不直接把 hidden 维平均切给每张卡？**
  A：注意力必须以「完整的头」为单位计算，切碎单个头的内部维度会破坏 QK^T 的语义。所以 Megatron 按 head 切。

- **Q：TP 和数据并行（DDP）冲突吗？**
  A：不冲突，可组合成 2D/3D 并行（见 Day74）。TP 切模型、DP 切数据。

---

## 延伸阅读

- [Megatron-LM 论文：Efficient Large-Scale LM Training on GPU Clusters](https://arxiv.org/abs/2104.04473)
- [NVIDIA/Megatron-LM 官方仓库](https://github.com/NVIDIA/Megatron-LM)
