# Day 81: NCCL 通信算法 — Ring / Tree AllReduce

> 🌐 **第十三阶段（通信·存储·工程化）开篇**。Day39 你见过 NCCL AllReduce 的「语义」，本天深入到 NCCL **内部到底怎么把 N 张卡的数据求和再发回去**——这是分布式训练通信性能的核心。

## 学习目标

1. **实现** Ring AllReduce 的两阶段（reduce-scatter + all-gather）并验证正确性
2. **实现** Tree AllReduce（reduce 上行 + broadcast 下行）
3. **对比** 朴素 / Ring / Tree 三种算法的通信步数与每 rank 发送量
4. **理解** 为什么 Ring 是「带宽最优」：每 rank 发送量 2(N-1)/N·D 不随卡数膨胀

---

## 前置依赖

- **Day 39 NCCL AllReduce**：本天是它的算法内幕展开
- **Day 25 多 GPU**：AllReduce 是数据并行梯度同步的核心原语
- **Day 9 归约**：reduce-scatter 本质是分布式的分段归约

---

## 核心概念

### AllReduce 语义

N 个 rank，每个持有向量 x_i，AllReduce(sum) 后**每个 rank 都得到** `x_0 + x_1 + ... + x_{N-1}`。数据并行训练里，它用来同步各卡的梯度。

### Ring AllReduce（NCCL 默认，带宽最优）

把每个 rank 的向量切成 N 个 chunk，rank i 只与右邻 `(i+1)%N` 通信：

```
阶段一 reduce-scatter（N-1 步）：
  每步把某个 chunk 的部分和发给右邻、收左邻的另一个 chunk 累加，
  N-1 步后 rank i 手里的第 (i+1)%N 号 chunk = 该 chunk 的全局和。

阶段二 all-gather（N-1 步）：
  把「已是全局和的 chunk」沿环传一圈，人人集齐所有 chunk。

     rank0 ──chunk──▶ rank1 ──chunk──▶ rank2
       ▲                                  │
       └──────────────chunk───────────────┘
```

每 rank 收/发各 `2(N-1)/N · D` 个元素——**随 N 增大趋于常数 2D**，故带宽最优。

### Tree AllReduce（延迟最优）

以二叉树组织 rank：reduce 上行汇聚到根、broadcast 下行广播。步数仅 `O(log N)`，**延迟低**，适合小数据/大规模集群，但每步传完整 D，带宽利用不如 Ring。NCCL 会按消息大小/拓扑在 Ring / Tree / CollNet 间自动选择。

### 三种算法对比

| 算法 | 通信步数 | 每 rank 发送量 | 特点 |
|------|----------|----------------|------|
| 朴素 | O(N) | (N-1)·D | 简单，带宽随 N 恶化 |
| Ring | 2(N-1) | 2(N-1)/N·D ≈ 2D | 带宽最优，NCCL 默认 |
| Tree | 2·log₂N | ~log₂N/N·D | 延迟最优，适合小数据 |

---

## 文件结构

```
day81_nccl_algorithms/
├── python/nccl_algorithms.py   # Ring/Tree/朴素 AllReduce 单机模拟 + 对比
├── build.sh                    # Linux/WSL2 运行脚本
├── build.bat                   # Windows 运行脚本
└── README.md                   # 本文件
```

> 本脚本**无需 GPU**：用 NumPy 在单机模拟 N 个 rank 的显存与逐步收发，算法逻辑与 NCCL 一致。

---

## 编译与运行

### 单机（无需 GPU，推荐先跑通）

```bash
cd day81_nccl_algorithms
python python/nccl_algorithms.py
```

### 真实多卡（Linux + 多张 NVIDIA GPU）

```bash
torchrun --nproc_per_node=4 your_train.py   # NCCL 后端自动选 Ring/Tree
export NCCL_ALGO=Ring   # 或 Tree，可强制指定算法
export NCCL_DEBUG=INFO   # 打印 NCCL 实际选择的算法与拓扑
```

### Windows

```powershell
cd day81_nccl_algorithms
.\build.bat
```

---

## 知识点总结

1. **reduce-scatter + all-gather**：Ring AllReduce = 这两个原语拼起来。
2. **带宽最优性**：Ring 每 rank 发送 2(N-1)/N·D，N 大时趋于 2D，与卡数解耦。
3. **延迟 vs 带宽**：Tree 步数 O(log N) 延迟低；Ring 带宽高。小数据用 Tree、大数据用 Ring。
4. **NCCL 自动选择**：由消息大小、拓扑（NVLink/PCIe/网络）决定，可用 `NCCL_ALGO` 覆盖。
5. **切 chunk**：环形算法把张量切成 N 份并行流动，才能打满带宽。

---

## 验收标准

- 程序打印三种算法对比表，你能解释 Ring 两阶段各做了什么。
- 三种算法「与标准和最大误差」均 ≈ 0，显示「全部一致 ✓」。
- 实测 Ring 平均每 rank 发送量 ≈ 2(N-1)/N·D（N=8,D=1000 时约 1750）。
- 你能说出何时用 Ring、何时用 Tree。

---

## 动手挑战

1. **改 rank 数**：把 `num_ranks` 改为 4 / 16 / 32，观察 Ring 发送量是否逼近 2D。
2. **非整除切分**：把 `dim` 改成不能被 N 整除的数（如 1001），验证 `np.array_split` 仍正确。
3. **实现 AllGather**：仿照 Ring 的第二阶段，单独实现一个 Ring AllGather 并验证。
4. **加计时**：给每步加入模拟延迟（步数×α + 数据量×β），画出 Ring vs Tree 的总耗时随数据大小的交叉点。

---

## 常见问题

- **Q：为什么 Ring 不是发送量最小，却说带宽最优？**
  A：「带宽最优」指每 rank 发送量不随 N 膨胀（趋于 2D），能持续打满链路；Tree 发送量更小但步数带来延迟，小数据才划算。

- **Q：单机模拟和真实 NCCL 有何差异？**
  A：语义与算法完全一致；差异在于真实 NCCL 走 NVLink/网卡并行流水、有拓扑感知与 kernel 融合。本模拟聚焦「算法正确性 + 通信量」。

---

## 延伸阅读

- [NCCL 官方文档 — Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [Baidu Ring AllReduce 经典博客](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)
