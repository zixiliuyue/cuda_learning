# Day 67: 分布式基础 — 进程组与集合通信

> 🌐 **第十二阶段（分布式训练）开篇**。前面你在单卡上把 kernel、推理、算子玩透了；从本天起进入分布式训练的世界。地基是「集合通信」：多个进程之间如何交换、聚合张量。命中 JD「训练优化」核心。

## 学习目标

1. **解释** rank / world_size / local_rank / backend / 进程组的含义
2. **实现** 5 个核心集合通信原语：broadcast / reduce / all_reduce / all_gather / reduce_scatter
3. **理解** 关键恒等式 `all_reduce = reduce_scatter + all_gather`，它是后续 ZeRO / 张量并行的通信基础
4. **掌握** torch.distributed 真实多进程的初始化与运行方式（torchrun / gloo）

---

## 前置依赖

- **Day 25 多 GPU**：理解多设备协作的动机
- **Day 39 NCCL AllReduce**：本天从更高层面理解 all_reduce 的语义与实现关系
- **numpy 基础**：本天用 numpy 数组列表模拟「每个 rank 各持一份数据」

---

## 核心概念

### 进程组与 rank

分布式训练里，每张 GPU 通常对应一个**进程（process）**。这些进程组成一个**进程组（process group）**，通过集合通信协作。

| 术语 | 含义 |
|------|------|
| `world_size` | 参与训练的进程总数（≈ 用到的 GPU 数） |
| `rank` | 当前进程的全局编号，`0 .. world_size-1`，全局唯一 |
| `local_rank` | 进程在本机内的编号（多机时用来 `set_device`） |
| `backend` | 通信后端：GPU 用 `nccl`，CPU/调试用 `gloo` |
| 进程组 | 一组参与集合通信的进程；默认全体进程构成一个组 |

### 5 个核心原语（ASCII 图）

```
broadcast (src=0)            reduce (sum→dst=0)
  r0:[1] ─┬─> r0:[1]           r0:[1]─┐
          ├─> r1:[1]           r1:[2]─┼─sum─> r0:[10]
          ├─> r2:[1]           r2:[3]─┤       r1:[2] (不变)
          └─> r3:[1]           r3:[4]─┘       ...

all_reduce (sum→all)         all_gather
  r0:[1]─┐                     r0:[1] ─> r0:[1,2,3,4]
  r1:[2]─┼─sum─> 全部[10]      r1:[2] ─> r1:[1,2,3,4]
  r2:[3]─┤                     r2:[3] ─> r2:[1,2,3,4]
  r3:[4]─┘                     r3:[4] ─> r3:[1,2,3,4]

reduce_scatter (先 sum 再切分)      恒等式：
  [1]                                all_reduce
  [2] ─sum─> [10,10,10,10]             = reduce_scatter + all_gather
  [3]        切分 ─> r0:[10] r1:[10]
  [4]                r2:[10] r3:[10]
```

### 为什么单卡也要学

- 真实多进程（`mp.spawn` + `nccl/gloo`）在部分平台（尤其 Windows）**容易卡死或初始化失败**，教学风险高。
- 本天用**单进程 + numpy 数组列表**精确模拟「每个 rank 各持一份数据」，忠实复现每个原语的语义与前后数据变化，**稳定可退出**。
- 语义与 `torch.distributed` 完全一致，脚本末尾给出真实多进程运行骨架（`torchrun`）供有多卡环境时使用。

---

## 文件结构

```
day67_distributed_basics/
├── python/distributed_basics.py   # 5 个集合通信原语的单进程模拟 + 真实骨架
├── build.sh                       # Linux/WSL2 运行脚本
├── build.bat                      # Windows 运行脚本
└── README.md                      # 本文件
```

---

## 编译与运行

### Windows

```powershell
cd day67_distributed_basics
.\build.bat
```

### Linux / WSL2

```bash
cd day67_distributed_basics
bash build.sh
```

### 真实多进程（有多卡或想体验 gloo）

```bash
# 把 README/脚本里的骨架保存为 run_dist.py
torchrun --nproc_per_node=4 run_dist.py
```

> 无 GPU 时脚本用 numpy 单进程模拟，语义一致、稳定退出，`exit code 0`。

---

## 知识点总结

1. **rank/world_size**：分布式的坐标系，每进程靠 rank 知道「我是谁、总共几个」。
2. **broadcast**：一对多复制，常用于同步初始参数。
3. **reduce**：多对一规约，只有 dst 拿到结果。
4. **all_reduce**：规约 + 广播，DDP 梯度同步的核心（sum 后除以 N 即平均）。
5. **all_gather**：每个 rank 拿到全量拼接，张量并行前向常用。
6. **reduce_scatter**：规约后按 rank 切分，ZeRO 显存优化的基石。
7. **核心恒等式**：`all_reduce = reduce_scatter + all_gather`，这是 Ring-AllReduce 高效实现的理论基础。

---

## 验收标准

- 程序打印 5 个原语的「通信前 / 通信后」各 rank 数据，且每个 `[校验]` 均显示「通过 ✓」。
- 你能对着 ASCII 图讲清 broadcast 与 all_reduce 的区别。
- 你能解释 `all_reduce = reduce_scatter + all_gather` 为什么成立。
- 程序 `exit code 0`，无卡死。

---

## 动手挑战

1. **改 world_size**：把 `make_ranks(4)` 改为 8，观察 all_reduce 结果（应为 1+2+..+8=36）。
2. **加 max 规约**：给 `all_reduce` 传 `op="max"`，验证结果变为各 rank 逐元素最大值。
3. **实现 all_reduce**：只用 `reduce_scatter` 和 `all_gather` 两个函数拼出 all_reduce，验证与直接 all_reduce 结果一致。
4. **真跑一次**：在 Linux/WSL2 上用 `torchrun --nproc_per_node=2 run_dist.py` 跑起来（gloo 后端，无需 GPU）。

---

## 常见问题

- **Q：为什么不直接用 `mp.spawn` 起真进程？**
  A：多进程 + 分布式初始化在 Windows / 无 GPU 环境极易卡死或报 rendezvous 错误。教学以「稳定看懂语义」为先，用单进程精确模拟；真实骨架已给出，有多卡时直接 `torchrun`。

- **Q：all_reduce 和 all_gather 有什么区别？**
  A：all_reduce 做「规约」（数据被聚合，形状不变）；all_gather 做「拼接」（数据被收集，形状变大 N 倍）。

---

## 延伸阅读

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Collective Operations (PyTorch docs)](https://pytorch.org/docs/stable/distributed.html)
- [Bringing HPC Techniques to Deep Learning — Ring AllReduce](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)
