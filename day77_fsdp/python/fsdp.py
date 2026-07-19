#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 77: 认识 FSDP（Fully Sharded Data Parallel）— 把参数/梯度/优化器状态分片

DDP（数据并行）的痛点：每张卡都存**一份完整**的参数、梯度、优化器状态。
模型一大（比如优化器 Adam 每个参数要额外存 2 份动量），单卡显存就爆了。

FSDP 的核心思想：**把参数、梯度、优化器状态都沿卡数切成 N 份**，
每张卡平时只持有 1/N。真正需要完整参数计算时（前向、反向的某一层），
临时 all-gather 出完整参数，**算完立刻释放**，用时间换显存。

本天以【原理 + 单机可跑的显存账本模拟】呈现：
  - 若装了 torch 且版本支持 FSDP，会探测并打印 FSDP 可用信息；
  - 无论有无 GPU，都用单进程模拟 world_size 张卡，
    逐项计算 DDP vs FSDP 的每卡显存占用，打印对比表；
  - 用一个「玩具参数」演示 all-gather → 计算 → reshard（释放）的生命周期。

单机单卡不真正起 mp.spawn（易卡死），用单进程账本演示等价的显存结论。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量：torch / CUDA / FSDP 是否可用）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

CUDA_AVAILABLE = bool(TORCH_AVAILABLE and torch.cuda.is_available())

FSDP_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel  # noqa: F401
        FSDP_AVAILABLE = True
    except Exception:
        FSDP_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  1. 原理讲解
# ============================================================
def explain_fsdp():
    print(SEP)
    print("FSDP：Fully Sharded Data Parallel（全分片数据并行）")
    print(SEP)
    print("""
  训练时每个参数身上其实挂着 3 类显存开销（以 fp32 + Adam 为例）：
    ① 参数 param            (P 个数, 4 字节)
    ② 梯度 grad             (P 个数, 4 字节)
    ③ 优化器状态 opt state  (Adam: 一阶动量 + 二阶动量 = 2 份, 8 字节)

  DDP（数据并行）：每张卡都存【完整】的 ①②③ —— 与卡数无关，单卡压力最大。
  FSDP：把 ①②③ 全部沿 world_size 切片，每张卡只存【1/N】。

  FSDP 前向/反向的参数生命周期（逐层进行）：
    ┌─ 进入某层 ─┐
    │  all-gather 该层参数分片 → 拼出完整参数（临时占显存）
    │  用完整参数做该层前向/反向计算
    │  reshard：立即丢弃拼出的完整参数，只留回自己那 1/N
    └────────────┘
    反向额外：对梯度做 reduce-scatter（求和 + 切片），每卡只留自己那份梯度。

  代价：比 DDP 多了 all-gather 通信；收益：显存近似降为 1/N，能训更大模型。
""")


# ============================================================
#  2. 显存账本：DDP vs FSDP 每卡占用对比
# ============================================================
def bytes_human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def demo_memory_ledger(num_params, world_size, bytes_param=4):
    print(SEP)
    print(f"演示 1：DDP vs FSDP 每卡显存账本（参数量={num_params:,}, world_size={world_size}）")
    print(SEP)

    # 每参数的三类开销（字节）：param + grad + Adam(2份)
    per_param_full = bytes_param + bytes_param + 2 * bytes_param  # = 16 (fp32)

    # DDP：每卡存完整一份
    ddp_param = num_params * bytes_param
    ddp_grad = num_params * bytes_param
    ddp_opt = num_params * 2 * bytes_param
    ddp_total = ddp_param + ddp_grad + ddp_opt

    # FSDP：三类都切成 1/world_size（忽略临时 all-gather 峰值）
    fsdp_param = ddp_param / world_size
    fsdp_grad = ddp_grad / world_size
    fsdp_opt = ddp_opt / world_size
    fsdp_total = ddp_total / world_size

    print(f"  单参数三类开销合计 = {per_param_full} 字节 (param4 + grad4 + Adam 8)")
    print()
    print(f"  {'项目':<14}{'DDP/卡':>16}{'FSDP/卡':>16}")
    print(f"  {'-'*46}")
    print(f"  {'参数 param':<12}{bytes_human(ddp_param):>16}{bytes_human(fsdp_param):>16}")
    print(f"  {'梯度 grad':<13}{bytes_human(ddp_grad):>16}{bytes_human(fsdp_grad):>16}")
    print(f"  {'优化器 Adam':<11}{bytes_human(ddp_opt):>16}{bytes_human(fsdp_opt):>16}")
    print(f"  {'-'*46}")
    print(f"  {'合计':<13}{bytes_human(ddp_total):>16}{bytes_human(fsdp_total):>16}")
    print()
    print(f"  FSDP 每卡显存 ≈ DDP 的 1/{world_size}"
          f"（临时 all-gather 峰值另计，属瞬时开销）")


# ============================================================
#  3. 玩具演示：all-gather → 计算 → reshard 的参数生命周期
#     单进程模拟 world_size 张卡各持有参数的一个分片
# ============================================================
def demo_param_lifecycle(world_size=4, dim=12):
    print()
    print(SEP)
    print(f"演示 2：FSDP 参数生命周期（all-gather→计算→reshard），world_size={world_size}")
    print(SEP)

    rng = np.random.default_rng(7)
    full_weight = rng.standard_normal(dim).astype(np.float64)  # 完整参数（真值）

    assert dim % world_size == 0, "参数维度需能被 world_size 整除"
    shard = dim // world_size

    # 平时：每张卡只持有一个分片（模拟为 list，每项是该 rank 的 1/N）
    shards = [full_weight[r * shard:(r + 1) * shard].copy() for r in range(world_size)]
    print(f"  平时每卡只存 {shard}/{dim} 个参数（分片持有）")

    # 需要计算某层 → all-gather：把各 rank 的分片拼成完整参数
    gathered = np.concatenate(shards, axis=0)
    print(f"  all-gather 后临时拼出完整参数（{dim} 个），用于前向/反向计算")

    x = rng.standard_normal(dim).astype(np.float64)
    y = float(gathered @ x)                          # 用完整参数做一次计算
    y_ref = float(full_weight @ x)
    print(f"  用完整参数计算结果 = {y:.6f}，与真值误差 = {abs(y - y_ref):.2e}")

    # reshard：立即释放拼出的完整参数，只保留自己的分片
    del gathered
    print(f"  reshard：释放完整参数，每卡回到只存 {shard}/{dim} 的状态")
    print(f"  -> 这就是 FSDP 用【时间(通信)】换【显存】的核心机制")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_fsdp()

    # 用一个「假想 1 亿参数模型」做账本对比，直观感受量级
    demo_memory_ledger(num_params=100_000_000, world_size=4)
    demo_param_lifecycle(world_size=4, dim=12)

    print()
    print(SEP)
    if FSDP_AVAILABLE and CUDA_AVAILABLE:
        ver = torch.__version__ if TORCH_AVAILABLE else "?"
        print(f"环境：torch {ver} + CUDA + FSDP 可用。")
        print("      真正多卡请用 torchrun 启动，并用 FullyShardedDataParallel 包裹模型；")
        print("      本天为原理与显存账本演示，单进程模拟避免 mp.spawn 卡死。")
    elif FSDP_AVAILABLE:
        print(f"环境：torch {torch.__version__} 支持 FSDP，但当前无 CUDA GPU。已用单进程模拟原理。")
    elif TORCH_AVAILABLE:
        print(f"环境：torch {torch.__version__} 未提供 FSDP（版本较旧）。已用 numpy 模拟原理。")
    else:
        print("环境：未安装 PyTorch。已用 numpy 模拟 FSDP 显存账本与参数生命周期。")
    print("要点：FSDP 把 参数/梯度/优化器状态 全切成 1/N，用时才 all-gather，用完即释放。")
    print(SEP)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
