# Day 73: MoE 并行与专家路由（Mixture of Experts）

> 🧠 **分布式训练之稀疏专家**。MoE 用「稀疏激活」换「巨大参数量」：一层里放很多专家，但每个 token 只走 top-k 个，因此参数量可上千亿而单 token 计算量可控。本天实现简化 MoE 层：gating 路由、dispatch、专家计算、combine，并统计负载均衡。

## 学习目标

1. **解释** MoE 如何用稀疏激活在不成比例增加计算量的前提下扩大参数量
2. **实现** gating 网络的 top-k 路由（token → 专家）
3. **实现** dispatch → expert compute → combine 的完整 MoE 前向
4. **理解** expert parallel（专家分布多 rank）与负载均衡问题

---

## 前置依赖

- **Day 45 Fused MLP**：专家本身就是 MLP
- **Day 39 NCCL / All-to-All**：expert parallel 的 dispatch/combine 靠 All-to-All 通信
- **Day 52 采样 / top-k**：gating 的 top-k 选择

---

## 核心概念

### 稀疏激活

```
token → gating 打分 → 选 top-k 专家 → 只算这 k 个专家 → 按权重合并
```

E 个专家，每 token 只走 k 个，激活比例 k/E。参数量随 E 增长，但单 token FLOPs 只随 k 增长。

### MoE 前向四步

| 步骤 | 说明 |
|------|------|
| **gating** | `logits = x @ gate_w` → softmax → top-k → 对 top-k 权重重新归一化 |
| **dispatch** | 按路由结果把 token 分组发给各专家（多 rank 时用 All-to-All） |
| **expert compute** | 每个专家对分到的 token 做 MLP |
| **combine** | 各专家输出乘 gating 权重后加回对应 token |

### Expert Parallel 与负载均衡

专家分布到多 rank，每 rank 持 E/R 个。若 gating 总把 token 送给少数专家，会造成部分 rank 过载、部分空闲：

- 用**变异系数 CV = std/mean** 衡量负载均衡（越小越均衡）；
- 训练时加**负载均衡辅助损失**促使路由均匀；
- 常配合 capacity factor 限制单专家最大 token 数（超出则 drop 或 overflow）。

---

## 文件结构

```
day73_moe/
├── python/moe.py   # gating top-k + dispatch + expert compute + combine + 负载统计
├── build.sh
├── build.bat
└── README.md
```

> 真实 expert parallel 需多卡 All-to-All。单机单卡下脚本用**单进程模拟**：用列表表示各 rank 持有的专家，分组 dispatch 模拟 token 分发，最后与逐 token 参考实现对拍。

---

## 编译与运行

### Linux / WSL2

```bash
cd day73_moe
python python/moe.py
```

### Windows

```powershell
cd day73_moe
.\build.bat
```

---

## 知识点总结

1. **稀疏激活**：每 token 只走 top-k/E 专家，参数量与计算量解耦。
2. **gating**：softmax 打分 + top-k + 权重重归一化。
3. **dispatch/combine**：分组分发与加权合并，多 rank 靠 All-to-All。
4. **expert parallel**：专家切到多 rank，各 rank 只算分到的 token。
5. **负载均衡**：CV 衡量均衡度，需辅助损失 + capacity 约束。
6. **数值等价**：分组 dispatch 前向 == 逐 token 参考（误差 < 1e-4）。

---

## 验收标准

- 程序打印每个专家的负载条形图与变异系数 CV。
- 分组 dispatch 前向与逐 token 参考的最大误差 < 1e-4，显示「通过 ✓」。
- 你能解释 MoE 为什么能「大参数量、小计算量」。
- 你能说出负载不均会带来什么问题、如何缓解。

---

## 动手挑战

1. **改 top-k**：把 `k` 从 2 改为 1（Switch Transformer 风格）或 4，观察负载与结果。
2. **加负载均衡损失**：实现 `aux_loss = E * sum(f_i * P_i)`（f=分派比例，P=平均门控概率），打印其值。
3. **加 capacity**：给每个专家设最大 token 数，超出的 token 直接 drop，观察对结果的影响。

---

## 常见问题

- **Q：top-k 里为什么要对权重重新归一化？**
  A：softmax 是在全部专家上归一的，只取 top-k 后权重和 < 1，需重新归一让加权合并保持尺度。

- **Q：负载不均只是效率问题吗？**
  A：不仅效率。过载专家的 token 可能被 drop，影响精度；空闲 rank 浪费算力。均衡是 MoE 工程核心难点。

---

## 延伸阅读

- [Switch Transformer（top-1 MoE）](https://arxiv.org/abs/2101.03961)
- [GShard（MoE 并行）](https://arxiv.org/abs/2006.16668)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
