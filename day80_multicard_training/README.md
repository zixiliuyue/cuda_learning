# Day 80: 阶段项目 — 多卡训练一个小模型

> 🏁 **第十二阶段（分布式训练）收官**。整合本阶段所学，用**单进程模拟多 rank 的 DDP** 把数据并行的核心闭环端到端跑通：切数据 → 本地反向 → 梯度 all-reduce 求平均 → 同步更新，并做 1/2/4 rank 的扩展性分析。

## 学习目标

1. **实现** 一个完整的数据并行训练闭环（DDP 的四步）
2. **理解** 梯度 all-reduce 求平均 ≡ 扩大有效 batch 的单卡训练（数学等价）
3. **验证** 不同 rank 数下最终 loss 一致（并行不改变收敛数学）
4. **分析** 扩展性：吞吐、加速比、并行效率，理解为什么 N 卡 < N 倍加速

---

## 前置依赖

- **Day 68 DDP / Day 78 通信重叠 / Day 79 profiling**：本天的直接整合对象
- **Day 39 NCCL AllReduce**：梯度同步的核心原语
- **Day 26 AMP**：真实多卡训练常叠加混合精度

---

## 核心概念

### DDP 单步训练闭环

```
① 切数据：全局 batch 切成 N 份，rank r 拿第 r 份（各卡数据不同）
② 本地算：每个 rank 用相同参数各自前向+反向 → 得到梯度 g_r
③ all-reduce：g = (g_0+...+g_{N-1}) / N（对所有 rank 求平均）
④ 同步更新：每个 rank 用同一个 g 更新 → 各卡参数始终一致
```

**数学等价**：平均各 rank 梯度 = 把 N 份数据拼成大 batch 在单卡上算。所以 **DDP ≡ 大 batch 单卡训练**。

### 扩展性（为什么 N 卡 < N 倍）

- **通信开销**：梯度 all-reduce 时间不随算力线性下降；
- **不可并行部分（Amdahl）**：数据加载、优化器 step 等；
- ⇒ **并行效率 = 实际加速比 / N < 1**，卡越多通常效率越低。

---

## 文件结构

```
day80_multicard_training/
├── python/multicard_training.py   # 单进程模拟多 rank DDP 端到端训练 + 扩展性报告
├── build.sh                       # Linux/WSL2 运行脚本
├── build.bat                      # Windows 运行脚本
└── README.md                      # 本文件
```

> 有 torch 用 autograd，无则 numpy 手写反向。两条路径都用**单进程模拟** `world_size` 个 rank 的 DDP。单机单卡起 `torch.distributed + mp.spawn` 易卡死，而 DDP 的数学本质单进程即可完整、稳定复现。

---

## 编译与运行

### Linux / WSL2

```bash
cd day80_multicard_training
python python/multicard_training.py
```

### Windows

```powershell
cd day80_multicard_training
.\build.bat
```

### 真实多卡（有多张 GPU 时）

```bash
torchrun --nproc_per_node=4 your_ddp_script.py
# 脚本内用 DistributedDataParallel(model) 包裹模型，DDP 自动做梯度 all-reduce 与重叠
```

---

## 知识点总结

1. **四步闭环**：切数据 / 本地反向 / all-reduce 求平均 / 同步更新。
2. **梯度平均**：除以 world_size，等价于大 batch 单卡。
3. **参数一致**：初始一致 + 每步用同一平均梯度更新 ⇒ 各卡永远一致。
4. **扩展性指标**：吞吐(samples/s)、加速比、并行效率。
5. **Amdahl 上限**：通信与串行部分限制了并行加速比。

---

## 验收标准

- 端到端训练完成，1/2/4 rank 下最终 loss **完全一致**（数学等价的证据）。
- 打印扩展性报告，含吞吐、加速比、并行效率三列。
- 你能解释「梯度求平均为什么等价于大 batch 单卡」。
- 你能说出真实多卡里并行效率 < 100% 的两个原因。

---

## 动手挑战

1. **改全局 batch**：把 global_batch 改成 512，观察各 rank 数下 loss 是否仍一致。
2. **换 all-reduce 为求和**：把「除以 world_size」去掉，观察 loss 变化，理解为什么必须求平均。
3. **接真实 DDP**：在有多卡的机器上，用 `torchrun` + `DistributedDataParallel` 重写本例，对比真实吞吐扩展性。

---

## 常见问题

- **Q：为什么单进程模拟里吞吐没有随 rank 线性上升？**
  A：单进程只是把 N 份计算摊在同一设备上串行做，没有真正的并行硬件。它验证的是**数学正确性与闭环**，真实并行加速需要多卡。

- **Q：DDP 里各卡的 BatchNorm 统计量怎么办？**
  A：需要 SyncBatchNorm 跨卡同步统计量，否则各卡 BN 不一致。本例用的 MLP 无 BN，不涉及。

---

## 延伸阅读

- [PyTorch DDP 官方教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torchrun 弹性启动文档](https://pytorch.org/docs/stable/elastic/run.html)

---

## 🏁 第十二阶段回顾（Day 67-80：分布式训练）

本阶段从「单卡装不下 / 训不快」出发，系统走完分布式训练的四大并行维度与工程优化。核心一句话：**并行 = 把 数据/模型/计算 切开分到多卡，再用集合通信把结果缝合**。

| Day | 主题 | 一句话核心 |
|-----|------|-----------|
| 67 | 分布式基础：进程组与集合通信 | world/rank/backend，all-reduce 等原语是一切并行的地基 |
| 68 | 数据并行 DDP | 切数据、各卡本地梯度、all-reduce 求平均 ≡ 大 batch |
| 69 | ZeRO-1/2/3 显存优化 | 逐级分片 优化器状态→梯度→参数，显存换通信 |
| 70 | 张量并行 TP | Column/Row 并行 Linear，切单层权重矩阵 |
| 71 | 流水线并行 PP | 按层切分 + micro-batch 调度，减小气泡 |
| 72 | 序列/上下文并行 | 沿序列维切分，攻克超长上下文 |
| 73 | MoE 并行与专家路由 | 稀疏激活 + 专家分卡，扩容量不增计算 |
| 74 | 混合并行（3D 并行） | DP × TP × PP 组合，撑起超大模型 |
| 75 | 梯度检查点/激活重计算 | 丢激活、反向重算，显存换算力 |
| 76 | 认识 Megatron-LM 架构 | TP 切 MLP/MHA，一块只需 1 次 all-reduce（f/g 算子） |
| 77 | 认识 FSDP | 参数/梯度/优化器全分片，用时 all-gather 用完 reshard |
| 78 | 通信与计算重叠优化 | 梯度分桶 + 异步 all-reduce，藏进反向计算 |
| 79 | 训练 profiling 体系 | 分阶段计时定位 计算/通信/显存 瓶颈 |
| 80 | 阶段项目：多卡训练小模型 | 端到端 DDP 闭环 + 扩展性分析（本天） |

**并行维度速记**：
- **DP（数据并行）**：切数据，模型副本冗余 → DDP / FSDP(分片)
- **TP（张量并行）**：切单层权重 → Megatron
- **PP（流水线并行）**：切层 → micro-batch 流水
- **MoE**：切专家；**混合并行**：以上组合成 3D 并行

**显存优化谱系**：ZeRO-1/2/3 → FSDP（全分片）→ 梯度检查点（激活重算）。

**工程优化**：通信/计算重叠、梯度分桶、profiling 定位瓶颈。

至此第十二阶段（分布式训练）完成。你已建立从**单卡 → 多卡 → 大模型**的完整并行心智模型，下一阶段（Day 81-90）进入**通信·存储·工程化**。
