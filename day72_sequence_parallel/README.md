# Day 72: 序列并行 / 上下文并行（Sequence / Context Parallel）

> 🧩 **分布式训练之序列维切分**。当上下文长到 128K token，单卡放不下这么长序列的激活与 attention 中间量。本天学习沿「序列长度」维度切分，并解决 attention 需要看到全序列的难题——AllGather KV 与 Ring Attention。

## 学习目标

1. **解释** 序列并行如何沿 seq 维把 Q/K/V 切散到多个 rank
2. **理解** 为什么 attention 是序列并行的难点（每个 query 要看全部 key/value）
3. **实现** 两种通信方案：AllGather KV 与 Ring Attention（online-softmax 增量累加）
4. **验证** 序列并行注意力结果 == 完整序列注意力

---

## 前置依赖

- **Day 31 Flash Attention**：Ring Attention 本质是 Flash 的分块 online-softmax 在序列维度上的分布式版
- **Day 39 NCCL AllReduce / AllGather**：AllGather KV 的通信原语基础
- **Day 48 Transformer Decoder**：attention 结构

---

## 核心概念

### 沿序列维切分

```
完整序列 S=8 切到 4 rank：
  rank0: tok0,1   rank1: tok2,3   rank2: tok4,5   rank3: tok6,7
  每 rank 只持有 S/N 个 token 的 Q/K/V
```

与数据并行（切 batch）、张量并行（切权重）不同，序列并行切的是**单条样本的 seq 维**。

### attention 难点与两种解法

每个 query 需要与**全部** token 的 key/value 交互，但 K/V 被切散：

| 方案 | 做法 | 通信 | 峰值显存 |
|------|------|------|---------|
| **AllGather KV** | 各 rank 收齐完整 K/V，本地算 | O(N)，一次收齐 | 需存完整 K/V |
| **Ring Attention** | K/V 块环形流动，online-softmax 增量累加 | 每步传 1 块 | O(1)，最省 |

### Ring Attention 与 online-softmax

Ring Attention 让 K/V 块在 rank 间「环形」传递，每收到一块就用运行最大值 `m` 和归一化和 `l` 增量融合：

```
m_new = max(m_run, m_blk)
alpha = exp(m_run - m_new)          # 重缩放旧累加
acc   = acc * alpha + exp(scores - m_new) @ V_block
l_run = l_run * alpha + sum(exp(scores - m_new))
```

无需一次性构造完整 [S, S] 的 attention 矩阵，这正是超长上下文训练的关键。

---

## 文件结构

```
day72_sequence_parallel/
├── python/sequence_parallel.py   # AllGather / Ring 两种方案 + 对拍完整 attention
├── build.sh
├── build.bat
└── README.md
```

> 真实序列并行需多卡通信。单机单卡下脚本用**单进程模拟多 rank**：用列表存各 rank 分片，显式模拟 AllGather 与 Ring 通信，最后与完整 attention 对拍。

---

## 编译与运行

### Linux / WSL2

```bash
cd day72_sequence_parallel
python python/sequence_parallel.py
```

### Windows

```powershell
cd day72_sequence_parallel
.\build.bat
```

---

## 知识点总结

1. **切 seq 维**：每 rank 持 S/N 个 token 的 Q/K/V，分摊长序列显存。
2. **attention 难点**：query 需看全序列，K/V 却被切散。
3. **AllGather KV**：简单、通信大、需存完整 K/V。
4. **Ring Attention**：环形传 K/V + online-softmax，峰值显存 O(1)。
5. **数值等价**：两种方案结果都 == 完整 attention（误差 < 1e-4）。

---

## 验收标准

- 两种方案与完整 attention 的最大误差都 < 1e-4，显示「通过 ✓」。
- 你能解释为什么 attention 是序列并行的难点。
- 你能说出 Ring Attention 相比 AllGather 在峰值显存上的优势。

---

## 动手挑战

1. **改 rank 数**：把 `num_ranks` 从 4 改为 2 / 8，确认两种方案仍与参考一致。
2. **加因果掩码**：给 attention 加上 causal mask，思考 Ring 传递时哪些块可跳过（load balancing）。
3. **打印每步累加量**：在 Ring 循环里打印 `m_run / l_run`，观察 online-softmax 如何逐块收敛。

---

## 常见问题

- **Q：序列并行和张量并行冲突吗？**
  A：不冲突，常一起用。Megatron 的 sequence parallel 就与 TP 组合，进一步省激活显存。

- **Q：Ring Attention 顺序会影响结果吗？**
  A：不会。online-softmax 的融合对块顺序不敏感，最终结果与完整 attention 一致。

---

## 延伸阅读

- [Ring Attention 论文](https://arxiv.org/abs/2310.01889)
- [Megatron Sequence Parallelism](https://arxiv.org/abs/2205.05198)
- [Flash Attention（online softmax 原理）](https://arxiv.org/abs/2205.14135)
