# Day 68: 数据并行 DDP（Distributed Data Parallel）

> 🚀 数据并行是**最常用**的分布式训练范式，也是面试高频。核心一句话：模型复制多份、数据切分多份、梯度 all_reduce 平均、参数保持同步。本天用可运行代码**数学验证**「DDP 等价于单卡大 batch」。

## 学习目标

1. **描述** DDP 的完整工作流程：数据切分 → 各自前向反向 → 梯度 all_reduce 平均 → 同步更新
2. **证明并验证** 「N 个 rank 的平均局部梯度 == 单卡大 batch 梯度」
3. **理解** 为什么每个 rank 的 loss 要对分片内样本取均值，梯度才能正确平均
4. **掌握** `DistributedDataParallel` + `DistributedSampler` 的真实用法骨架

---

## 前置依赖

- **Day 67 集合通信**：DDP 的梯度同步就是一次 `all_reduce(SUM)` 再除以 world_size
- **自动微分基础**：理解前向/反向、梯度的含义
- **线性层与 MSE loss**：本天用最小线性回归验证等价性

---

## 核心概念

### DDP 工作流程（ASCII 图）

```
大 batch (B 个样本)
    │  按 rank 切分（DistributedSampler）
    ├──> rank0: 分片0 ─前向─反向─> 局部梯度 g0
    ├──> rank1: 分片1 ─前向─反向─> 局部梯度 g1
    ├──> ...
    └──> rankN: 分片N ─前向─反向─> 局部梯度 gN
                                  │
                  all_reduce(SUM) / N  （梯度平均）
                                  │
            所有 rank 得到同一份平均梯度 ḡ
                                  │
            各自 optimizer.step()（参数始终一致）
```

### 数学保证：DDP == 单卡大 batch

设单卡在 B 个样本上的 loss 取均值：

```
单卡:  loss = (1/B) Σ_i loss_i        → grad = (1/B) Σ_i g_i
rank r 只有 B/N 个样本，loss_r = (N/B) Σ_{i∈r} loss_i
       → 局部梯度 g_r = (N/B) Σ_{i∈r} g_i
平均:  (1/N) Σ_r g_r = (1/N) Σ_r (N/B) Σ_{i∈r} g_i = (1/B) Σ_i g_i   ✓
```

**结论**：梯度平均后与单卡大 batch 完全相等，DDP 只是把计算分摊到多卡，不改变优化轨迹。

> ⚠️ 关键细节：每个 rank 的 loss 必须对**分片内样本取均值**（而非 sum），最后梯度用**平均**（而非 sum）聚合，两者配合才严格等价。这也是 DDP 默认行为。

### 单进程模拟多 rank

真实多进程（`mp.spawn`）在 Windows/无 GPU 环境易卡死。本天用**单进程**：
- `run_torch_demo`：用真实 autograd 分别算各 rank 局部梯度，再 `stack().mean()` 模拟 all_reduce 平均；
- `run_training_loop_numpy`：跑 5 步训练，逐步打印单卡 vs DDP 的 loss 与参数差异（应恒为 ~1e-16）。

---

## 文件结构

```
day68_ddp/
├── python/ddp.py     # 梯度平均等价性验证 + 多步训练同步演示 + 真实 DDP 骨架
├── build.sh          # Linux/WSL2 运行脚本
├── build.bat         # Windows 运行脚本
└── README.md         # 本文件
```

---

## 编译与运行

### Windows

```powershell
cd day68_ddp
.\build.bat
```

### Linux / WSL2

```bash
cd day68_ddp
bash build.sh
```

### 真实多卡 DDP

```bash
torchrun --nproc_per_node=4 train_ddp.py   # 参考脚本末尾骨架
```

> 有 torch 时用真实 autograd，无 torch 时用 numpy 手推梯度，均 `exit code 0`、稳定退出。

---

## 知识点总结

1. **模型复制、数据切分**：DDP 的两大前提。
2. **梯度 all_reduce 平均**：`SUM` 后除以 world_size，是 DDP 唯一的通信点。
3. **loss 取均值 + 梯度取平均**：两者配合保证与单卡大 batch 严格等价。
4. **DistributedSampler**：保证各 rank 拿到互不重叠的数据分片。
5. **overlap 通信与计算**：真实 DDP 在 backward 过程中分桶（bucket）逐步 all_reduce，与反向计算重叠以提速。
6. **参数天然同步**：初值 broadcast 一致 + 每步同一份平均梯度 → 参数永远一致，无需额外同步。

---

## 验收标准

- 验证 1：`dW / db 最大误差 < 1e-9`，显示「平均梯度 == 单卡梯度 通过 ✓」。
- 验证 2：5 步训练中单卡与 DDP 的 loss 完全重合，参数差异恒为 ~1e-16，显示「完全一致 ✓」。
- 你能口头推导「平均局部梯度 == 单卡梯度」。
- 程序 `exit code 0`，无卡死。

---

## 动手挑战

1. **改 world_size**：把 4 改成 2 或 8，误差应仍 < 1e-9。
2. **打破等价性**：故意让某个 rank 的 loss 用 `sum` 而非 `mean`，观察误差如何变大，理解为什么必须一致。
3. **加权平均**：若各 rank batch 大小不同，梯度平均应按样本数加权，改写聚合逻辑并验证。
4. **真跑 DDP**：在 Linux/WSL2 用 `torchrun --nproc_per_node=2` 跑一个 gloo 后端的最小 DDP 训练。

---

## 常见问题

- **Q：DDP 和 DataParallel(DP) 有什么区别？**
  A：`DataParallel` 单进程多线程、有 GIL 瓶颈、主卡负载不均；`DistributedDataParallel` 每卡一进程、通信高效，是官方推荐。

- **Q：梯度是 sum 还是 average？**
  A：DDP 内部 `all_reduce(SUM)` 后除以 world_size（等价 average）。只要 loss 对本地 batch 取均值，就与单卡大 batch 一致。

- **Q：为什么单进程模拟也算数？**
  A：DDP 的正确性来自「梯度平均」这一数学操作，与进程数无关。单进程精确复现该操作即可验证正确性；性能层面的通信重叠才需要真多进程。

---

## 延伸阅读

- [PyTorch DDP 官方教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training (VLDB'20)](https://arxiv.org/abs/2006.15704)
- [DistributedSampler 文档](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
