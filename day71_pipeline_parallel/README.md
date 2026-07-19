# Day 71: 流水线并行（Pipeline Parallel）

> 🚀 **第十一阶段（分布式训练）开篇**。当模型大到单卡放不下时，除了张量并行，另一条路是把模型「按层」切开分到多张卡上——这就是流水线并行。本天讲清 stage 切分、micro-batch 调度（GPipe / 1F1B）与流水线气泡。

## 学习目标

1. **解释** 流水线并行如何把模型「按层」切分到多个 stage，激活/梯度如何在 stage 间传递
2. **区分** GPipe 的「F-then-B」调度与 Megatron 的「1F1B」调度，及其显存差异
3. **推导** 流水线气泡（bubble）的来源与公式 `bubble = (p-1)/(m+p-1)`
4. **验证** 多 stage 串联前向的数值正确性

---

## 前置依赖

- **Day 25 多 GPU**：理解多卡协作的基础
- **Day 39 NCCL AllReduce**：stage 间的点对点通信（send/recv）与集合通信是分布式训练底座
- **Day 48 Transformer Decoder**：流水线并行最常切分的就是 Transformer 的 decoder 层

---

## 核心概念

### 按层切分（Layer Partition）

流水线并行（PP）与张量并行（TP）的关键区别：

| 维度 | 张量并行 TP | 流水线并行 PP |
|------|------------|--------------|
| 切什么 | 单层「内部」的权重矩阵 | 层与层「之间」 |
| 通信 | 每层都要 AllReduce | 只在 stage 边界 send/recv 激活 |
| 通信频率 | 高（层内） | 低（stage 边界） |
| 适合 | 单层很大 | 层数很多 |

```
    GPU0          GPU1          GPU2          GPU3
   [L0 L1]  -->  [L2 L3]  -->  [L4 L5]  -->  [L6 L7]
   前向 激活 activation 向右传递；反向 梯度 gradient 向左传递
```

### micro-batch 与调度

把一个 mini-batch 拆成 m 个 micro-batch，让它们错峰流过流水线，各 stage 尽量同时忙碌。

- **GPipe / F-then-B**：先做完所有 micro 的前向，再统一做反向。需同时保存全部 micro 的前向激活 → **显存大**。
- **1F1B（Megatron）**：预热后进入「一次前向、一次反向」交替的稳定期。激活显存只需约 stage 数份 → **显存友好**，是主流选择。

### 流水线气泡（Bubble）

流水线的「填充 + 排空」阶段，头尾 stage 会空闲，这段浪费叫气泡：

```
bubble_fraction = (p - 1) / (m + p - 1)      p=stage数, m=micro数
```

micro-batch 越多，气泡被摊得越薄，效率越高（但调度开销上升，需折中）。

---

## 文件结构

```
day71_pipeline_parallel/
├── python/pipeline_parallel.py   # stage 切分 + 调度时间线 + 气泡分析 + 数值验证
├── build.sh                      # Linux/WSL2 运行脚本
├── build.bat                     # Windows 运行脚本
└── README.md                     # 本文件
```

> 本天以原理为主。真实多 stage 流水线需要多卡，单机单卡下脚本用**单进程模拟**多 stage（激活用张量/数组在函数间传递），调度时间线用逻辑推演 + ASCII 图展示，保证无 GPU 也能 `python` 直接跑通。

---

## 编译与运行

### Linux / WSL2

```bash
cd day71_pipeline_parallel
python python/pipeline_parallel.py
```

### Windows

```powershell
cd day71_pipeline_parallel
.\build.bat
```

> 脚本会自动探测环境：多 GPU / 单 GPU / 仅 CPU（torch 或 numpy），无论哪种都能跑完并给出数值验证。

---

## 知识点总结

1. **PP vs TP**：PP 切层间、通信少；TP 切层内、通信多。
2. **激活/梯度传递**：前向激活向后传，反向梯度向前传，只在 stage 边界通信。
3. **GPipe**：F-then-B，实现简单但激活显存大。
4. **1F1B**：前反向交替，激活显存降到 ~stage 数份，主流方案。
5. **气泡公式**：`(p-1)/(m+p-1)`，增大 m 可摊薄气泡。
6. **串联等价性**：多个线性 stage 串联 == 单个等价大层，可用于对拍验证。

---

## 验收标准

- 程序打印 GPipe 与 1F1B 两张调度时间线，你能指出哪里是气泡。
- 气泡表随 micro 数增大，效率单调上升，你能用公式解释。
- 数值验证「流水线前向 vs 参考」最大误差 < 1e-3，显示「通过 ✓」。
- 你能说出 PP 与 TP 的两点区别，以及 1F1B 为什么比 GPipe 省显存。

---

## 动手挑战

1. **改 stage 数**：把 `num_stages` 从 4 改为 2 / 8，观察气泡占比如何变化。
2. **画你自己的时间线**：手动推演 3 stage、4 micro 的 1F1B 排布，与程序输出核对。
3. **加激活函数**：给每个 stage 的 `forward` 加 ReLU，思考「串联等价大层」的对拍还成立吗？为什么？

---

## 常见问题

- **Q：单卡为什么不能真跑流水线并行？**
  A：PP 的意义是多卡分摊模型；单卡上没有 stage 间真实通信，只能模拟调度逻辑与前向数值。

- **Q：1F1B 气泡比 GPipe 小吗？**
  A：两者气泡量级相同（都是 (p-1)/(m+p-1)），1F1B 的核心优势是**激活显存**大幅下降。

---

## 延伸阅读

- [GPipe 论文](https://arxiv.org/abs/1811.06965)
- [Megatron-LM 1F1B 调度](https://arxiv.org/abs/2104.04473)
- [PyTorch PiPPy（Pipeline Parallelism）](https://github.com/pytorch/PiPPy)
