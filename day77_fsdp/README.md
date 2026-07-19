# Day 77: 认识 FSDP（Fully Sharded Data Parallel）

> 🌐 **第十二阶段（分布式训练）**。DDP 每张卡都存一份完整的参数/梯度/优化器状态，模型一大单卡显存就爆。**FSDP** 把这三类开销全部沿卡数切片，每卡只存 1/N，用时才 all-gather，用完即释放——用通信换显存，能训更大模型。

## 学习目标

1. **解释** 训练时每个参数的三类显存开销（参数、梯度、优化器状态）
2. **对比** DDP 与 FSDP 的每卡显存占用，理解 FSDP ≈ 降为 1/N
3. **理解** FSDP 的参数生命周期：all-gather → 计算 → reshard（释放）
4. **权衡** FSDP 相比 DDP 多出的 all-gather 通信开销与显存收益

---

## 前置依赖

- **Day 68 DDP**：FSDP 是 DDP 的显存优化升级版
- **Day 69 ZeRO-1/2/3**：FSDP 本质等价于 ZeRO-3（全分片）
- **Day 39 NCCL AllReduce**：FSDP 用 all-gather / reduce-scatter 作为核心原语

---

## 核心概念

### 每个参数的三类显存开销（fp32 + Adam）

| 开销 | 大小/参数 |
|------|-----------|
| ① 参数 param | 4 字节 |
| ② 梯度 grad | 4 字节 |
| ③ 优化器状态（Adam 一阶+二阶动量） | 8 字节 |
| **合计** | **16 字节** |

### DDP vs FSDP

- **DDP**：每卡都存完整的 ①②③，与卡数无关，单卡压力最大。
- **FSDP**：三类全部沿 `world_size` 切片，每卡只存 **1/N**。

### FSDP 参数生命周期（逐层进行）

```
┌─ 进入某层 ─┐
│  all-gather 分片 → 拼出完整参数（临时占显存）
│  用完整参数做前向/反向计算
│  reshard：立即丢弃完整参数，只留回自己那 1/N
└────────────┘
反向额外：梯度做 reduce-scatter（求和+切片），每卡只留自己那份。
```

**代价**：比 DDP 多 all-gather 通信；**收益**：显存近似降为 1/N。

---

## 文件结构

```
day77_fsdp/
├── python/fsdp.py   # DDP vs FSDP 显存账本 + 参数生命周期模拟
├── build.sh         # Linux/WSL2 运行脚本
├── build.bat        # Windows 运行脚本
└── README.md        # 本文件
```

> 脚本会探测 torch 是否支持 `FullyShardedDataParallel`；无论有无 GPU，都用 **numpy 单进程模拟** `world_size` 张卡的显存账本与参数生命周期。单机单卡不真正起 `mp.spawn`（易卡死）。

---

## 编译与运行

### Linux / WSL2

```bash
cd day77_fsdp
python python/fsdp.py
```

### Windows

```powershell
cd day77_fsdp
.\build.bat
```

> 任何环境下都稳定运行并以 exit code 0 退出。

---

## 知识点总结

1. **三类开销**：param + grad + opt state，Adam 下合计 16 字节/参数。
2. **DDP 冗余**：每卡存完整副本，显存与卡数无关。
3. **FSDP 分片**：三类全切 1/N，每卡显存 ≈ DDP 的 1/N。
4. **生命周期**：用时 all-gather，用完 reshard 立即释放。
5. **reduce-scatter**：反向梯度求和+切片，每卡只留自己那份。
6. **FSDP ≈ ZeRO-3**：两者思想一致，都是全分片。

---

## 验收标准

- 显存账本表中 FSDP 每卡合计 ≈ DDP 的 1/world_size。
- 参数生命周期演示中 all-gather 后计算结果与真值误差为 0。
- 你能说出 FSDP 相比 DDP「多了什么、省了什么」。
- 你能解释为什么 all-gather 的完整参数是「瞬时峰值」而非常驻开销。

---

## 动手挑战

1. **改精度**：把 `bytes_param` 改成 2（fp16/bf16），重算账本，看混合精度如何进一步省显存。
2. **改 world_size**：观察 world_size=8/16 时每卡显存的变化趋势。
3. **加峰值估算**：在账本里补上「all-gather 时最大一层完整参数」的瞬时峰值项，思考它是否会成为新瓶颈。

---

## 常见问题

- **Q：FSDP 和 ZeRO 什么关系？**
  A：FSDP 是 PyTorch 原生实现，思想等价于 DeepSpeed 的 ZeRO-3（全分片）。

- **Q：FSDP 一定比 DDP 快吗？**
  A：不一定。FSDP 省显存但增加了 all-gather 通信，在小模型/通信慢的环境可能更慢；大模型下它常是唯一能跑起来的选择。

---

## 延伸阅读

- [PyTorch FSDP 官方教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [ZeRO 论文：Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
