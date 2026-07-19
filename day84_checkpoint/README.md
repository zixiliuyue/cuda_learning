# Day 84: Checkpoint 机制与性能优化

> 💾 训练动辄几天几周，机器随时可能挂或被抢占。Checkpoint（检查点）就是「定期存档」，让训练能从最近一次存档恢复。但存 checkpoint 会**阻塞训练**——本天讲同步/异步/分片/增量四种做法与优化。

## 学习目标

1. **实现** 同步 checkpoint 的保存/加载并验证恢复正确性
2. **实现** 异步 checkpoint（内存快照 + 后台线程写盘），量化阻塞时间减少
3. **实现** 分片 checkpoint（大模型/优化器状态按 rank 切分并行写）
4. **实现** 增量 checkpoint（只存相对上次变化的张量）
5. **理解** 原子替换（temp + rename）避免损坏 checkpoint

---

## 前置依赖

- **Day 26 AMP 训练 / Day 25 多 GPU**：checkpoint 是训练流程的一环
- **Day 81 NCCL 算法 / 分布式**：分片 checkpoint 与 ZeRO/FSDP 相关

---

## 核心概念

### 恢复要存什么

模型权重 + **优化器状态**（Adam 的动量/方差，往往和权重一样大）+ 学习率/step + RNG 状态。少存一样，恢复后训练轨迹就对不上。

### 四种做法

| 做法 | 主线程阻塞 | 单文件大小 | 适用 |
|------|-----------|-----------|------|
| 同步 | 长（序列化+写盘全前台） | 全量 | 简单场景 |
| 异步 | 短（仅内存快照） | 全量 | 通用，减少 GPU 空转 |
| 分片 | 中（各 rank 并行） | 1/N | 大模型 ZeRO/FSDP |
| 增量 | 短 | 仅变化部分 | 微调/LoRA，每步只动一小部分 |

### 异步的关键

主线程**先把张量快照到内存 buffer（快）**，再把慢速落盘丢给后台线程。主线程只被 snapshot 阻塞，随后可继续训练。脚本实测阻塞时间缩短约 3x+。

### 工程要点

写 `ckpt.tmp` 再 `rename` 成 `ckpt.pt`（原子替换）。避免写到一半崩溃留下损坏文件。

---

## 文件结构

```
day84_checkpoint/
├── python/checkpoint.py   # 同步/异步/分片/增量四种 checkpoint 演示
├── build.sh               # Linux/WSL2 运行脚本
├── build.bat              # Windows 运行脚本
└── README.md              # 本文件
```

> 脚本无需 GPU；有 torch 用 `torch.save/load`，否则用 `pickle+numpy`。文件写系统临时目录，运行结束自动清理。

---

## 编译与运行

### 单机（无需 GPU）

```bash
cd day84_checkpoint
python python/checkpoint.py
```

### 真实分布式 checkpoint（PyTorch）

```python
import torch.distributed.checkpoint as dcp   # 分片/异步的官方方案
# FSDP + dcp.save/load 可自动做分片 checkpoint
```

### Windows

```powershell
cd day84_checkpoint
.\build.bat
```

---

## 知识点总结

1. **存全**：权重 + 优化器状态 + step + RNG，缺一恢复轨迹就偏。
2. **异步**：内存快照放前台、落盘放后台，减少训练阻塞（GPU 少空转）。
3. **分片**：按 rank 切分并行写，单文件小，对应 ZeRO/FSDP。
4. **增量**：只存变化张量，微调/LoRA 场景省 IO 明显。
5. **原子替换**：temp + rename，防写坏。

---

## 验收标准

- 四种 checkpoint 的「恢复正确性」均显示 ✓ 一致。
- 异步版主线程阻塞时间明显小于同步版（脚本打印 speedup）。
- 分片合并后与原模型一致；增量文件明显小于全量。

---

## 动手挑战

1. **加优化器状态**：给 state 里加入模拟的 Adam `exp_avg/exp_avg_sq`，观察 ckpt 大小翻倍。
2. **原子替换**：把 `save_state` 改成写 `.tmp` 再 `os.replace`，模拟中途崩溃验证不损坏。
3. **限流异步**：用 `queue.Queue` + 单后台线程做「多次 checkpoint 排队写盘」。
4. **压缩**：给增量 ckpt 加 `gzip`，对比压缩前后大小与耗时。

---

## 常见问题

- **Q：异步 checkpoint 会不会存到不一致的状态？**
  A：不会——关键是主线程**先做一致性快照**（此刻参数被冻结拷贝），后台写的是快照，训练可继续更新参数。

- **Q：分片 checkpoint 换了卡数还能恢复吗？**
  A：需要「重分片（resharding）」。`torch.distributed.checkpoint` 支持存成与并行度解耦的格式，换 world_size 也能加载。

---

## 延伸阅读

- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)
- [DeepSpeed ZeRO Checkpoint](https://www.deepspeed.ai/tutorials/zero/)
