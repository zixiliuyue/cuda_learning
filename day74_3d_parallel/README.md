# Day 74: 混合并行 —— 3D 并行（DP + TP + PP）

> 🌐 **分布式训练之集大成**。训练千亿参数模型时，单一并行维度都不够用。工业界（Megatron-LM / DeepSpeed）把数据并行、张量并行、流水线并行组合成「3D 并行」。本天演示如何把 W 张 GPU 划分成 (dp, tp, pp) 网格，打印每张卡的 rank 坐标与职责，并对比通信量。

## 学习目标

1. **解释** DP / TP / PP 三种并行分别切什么、用什么通信原语
2. **实现** 把 W 张 GPU 映射到 (pp_rank, dp_rank, tp_rank) 三维坐标
3. **识别** 每张卡的 TP 组 / DP 组 / PP 组通信伙伴
4. **对比** 不同 (dp, tp, pp) 划分方案的通信量权衡

---

## 前置依赖

- **Day 71 流水线并行**：PP 维度
- **Day 25 多 GPU / TP 概念**：TP 维度（层内切权重）
- **Day 39 NCCL AllReduce**：DP 的梯度 AllReduce、TP 的层内 AllReduce

---

## 核心概念

### 三个维度各切什么

| 维度 | 切什么 | 通信原语 | 通信频率 |
|------|--------|---------|---------|
| **TP** 张量并行 | 单层内部权重矩阵 | 层内 AllReduce | 最高（每层多次） |
| **PP** 流水线并行 | 层与层之间 | stage send/recv | 低（stage 边界） |
| **DP** 数据并行 | 不同 batch | 梯度 AllReduce | 每步一次（可重叠） |

### rank 坐标映射

`dp * tp * pp = world_size`。常见排布让 TP 变化最快、PP 最慢：

```
world_rank = pp_rank*(dp*tp) + dp_rank*tp + tp_rank
```

每张卡的 `(pp_rank, dp_rank, tp_rank)` 坐标决定它负责哪些层、属于哪些通信组。

### 摆放经验法则

- **TP 放机内**：通信最频繁，靠 NVLink 高带宽；
- **PP 可跨节点**：通信少，只在 stage 边界传激活；
- **DP 放最外层**：每步一次梯度 AllReduce，可与反向计算重叠隐藏。

---

## 文件结构

```
day74_3d_parallel/
├── python/hybrid_parallel.py   # rank 坐标映射 + 通信组识别 + 通信量对比
├── build.sh
├── build.bat
└── README.md
```

> 本天为**纯逻辑推演**，不依赖真实多卡，无 GPU 甚至无 PyTorch 也能完整跑通。真实 3D 并行由 Megatron-LM / DeepSpeed 编排。

---

## 编译与运行

### Linux / WSL2

```bash
cd day74_3d_parallel
python python/hybrid_parallel.py
```

### Windows

```powershell
cd day74_3d_parallel
.\build.bat
```

---

## 知识点总结

1. **3D 并行**：DP × TP × PP，三者乘积 = GPU 总数。
2. **rank 坐标**：`(pp_rank, dp_rank, tp_rank)` 决定职责与通信伙伴。
3. **通信组**：同 TP 组做层内 AllReduce，同 DP 组做梯度 AllReduce，同 PP 组做 send/recv。
4. **通信频率**：TP > DP > PP，故 TP 放机内、PP 可跨节点。
5. **划分权衡**：TP 度数大→层内通信重；DP 度数大→梯度通信重但可重叠。

---

## 验收标准

- 程序打印 8 卡在 (2,2,2) 下的完整 rank 坐标表与各卡职责。
- 正确列出 rank0 的 TP / DP / PP 三个通信组成员。
- 通信量对比表能看出「TP 越大通信越重」的趋势。
- 你能解释为什么 TP 通常只放机内。

---

## 动手挑战

1. **改 GPU 总数**：把 `world` 改为 16 或 32，看 `factorizations` 列出的方案如何增多。
2. **改排布顺序**：把映射改成「PP 变化最快」，观察通信组成员如何变化。
3. **加显存维度**：为每种划分估算单卡显存占用（参数 + 激活），与通信量一起权衡。

---

## 常见问题

- **Q：为什么不全用 DP 就好？**
  A：DP 每张卡都要放下完整模型。模型太大时单卡放不下，必须用 TP/PP 切分参数。

- **Q：(dp,tp,pp) 怎么选最优？**
  A：无通用最优，取决于模型大小、单卡显存、互联带宽。一般先用 TP 填满机内，再用 PP 跨节点，剩下的给 DP。

---

## 延伸阅读

- [Megatron-LM 3D 并行](https://arxiv.org/abs/2104.04473)
- [DeepSpeed 3D Parallelism](https://www.deepspeed.ai/tutorials/pipeline/)
- [Efficient Large-Scale Training 综述](https://arxiv.org/abs/2205.05198)
