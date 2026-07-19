#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 39: NCCL 多 GPU AllReduce 概念演示（Python 版）

优先使用 torch.distributed（NCCL 后端）演示真实的多 GPU AllReduce；
若没有可用的 PyTorch / 多 GPU 环境，则退化为 numpy + multiprocessing
在 CPU 上模拟 AllReduce 语义，帮助理解概念。

AllReduce 语义：
    K 个进程/GPU，每个持有一份向量 x_i，
    AllReduce(sum) 后，每个进程都得到 sum(x_0 + ... + x_{K-1})。
"""

import os
import sys


LINE = "-" * 60


# ============================================================
# 路径一：torch.distributed + NCCL（需要多 GPU + CUDA）
# ============================================================
def try_torch_nccl():
    """尝试用 torch.distributed 的 NCCL 后端跑真实 AllReduce。

    该函数需要以分布式方式启动，例如：
        torchrun --nproc_per_node=4 nccl_demo.py
    单卡直接运行时会退化为世界大小为 1 的场景。
    返回 True 表示已完成 torch 路径，False 表示需要 fallback。
    """
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return False

    if not torch.cuda.is_available():
        print("[torch] 未检测到 CUDA GPU，跳过 torch.distributed 路径。")
        return False

    # torchrun 会设置这些环境变量；若不存在则说明是普通单进程运行。
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size == 1:
        # 单进程也演示 API：AllReduce 结果即自身。
        print("[torch] 单进程模式（world_size=1），演示 API 调用流程。")
        print("[torch] 多 GPU 真实通信请用: "
              "torchrun --nproc_per_node=N nccl_demo.py")

    # 初始化进程组，使用 NCCL 后端（GPU 间通信首选）。
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl",
                                rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 每个 rank 生成不同的值：全部填充为 (rank + 1)。
    n = 8
    x = torch.full((n,), float(rank + 1), device=device)
    if rank == 0:
        print(f"\n[torch] 每个 rank 持有长度 {n} 的向量，值为 (rank+1)")

    # 执行 AllReduce（求和），结果原地写回 x，所有 rank 得到相同值。
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    expected = sum(r + 1 for r in range(world_size))
    print(f"[torch][rank {rank}] AllReduce 后 x[0] = {x[0].item():.0f} "
          f"（期望 {expected}）")

    dist.barrier()
    dist.destroy_process_group()
    return True


# ============================================================
# 路径二：numpy + multiprocessing 模拟（无 torch / 无 GPU）
# ============================================================
def _worker(rank, world_size, n, send_conns, recv_conns):
    """每个进程模拟一个 GPU：生成本地数据，通过管道汇总实现 AllReduce。"""
    import numpy as np

    # 本地数据：全部为 (rank + 1)
    local = np.full(n, float(rank + 1), dtype=np.float32)

    # 简化的 AllReduce：所有 rank 把数据发给 rank 0，rank 0 求和后广播回去。
    # （真实 NCCL 用 Ring/Tree 算法并行完成，这里只演示语义。）
    send_conns[rank].send(local)
    if rank == 0:
        acc = np.zeros(n, dtype=np.float32)
        for r in range(world_size):
            acc += recv_conns[r].recv()
        for r in range(world_size):
            send_conns[r].send(acc)  # 广播结果
    result = recv_conns[rank].recv()

    print(f"[numpy][rank {rank}] AllReduce 后 = {result.tolist()}")


def numpy_simulation():
    import multiprocessing as mp
    import numpy as np  # noqa: F401

    world_size = 4
    n = 8
    print(f"[numpy] 用 {world_size} 个进程模拟 {world_size} 个 GPU，"
          f"向量长度 {n}")
    print(f"[numpy] 每个 rank 数据值为 (rank+1)")

    # 为每个 rank 创建双向管道
    pipes = [mp.Pipe() for _ in range(world_size)]
    send_conns = [p[0] for p in pipes]
    recv_conns = [p[1] for p in pipes]

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=_worker,
                       args=(rank, world_size, n, send_conns, recv_conns))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

    expected = sum(r + 1 for r in range(world_size))
    print(f"\n[numpy] 每个元素期望 = 1+2+...+{world_size} = {expected}")


def print_header():
    print()
    print("=" * 60)
    print("     Day 39: NCCL 多 GPU AllReduce 概念演示（Python）")
    print("=" * 60)
    print()
    print(LINE)
    print(" AllReduce: K 个 GPU 各持一份数据，规约(求和)后")
    print("            每个 GPU 都得到全体之和，结果完全一致。")
    print(LINE)
    print()


def print_summary():
    print()
    print("=" * 60)
    print("     AllReduce 算法对比")
    print("=" * 60)
    print(f"  {'算法':<20} | {'特点':<28}")
    print(LINE)
    print(f"  {'Ring AllReduce':<20} | {'带宽最优, 大张量, 单机多卡':<28}")
    print(f"  {'Tree AllReduce':<20} | {'延迟最优, 小张量, 多节点':<28}")
    print(f"  {'朴素 (拷回CPU求和)':<20} | {'PCIe瓶颈, 串行, 仅教学':<28}")
    print(LINE)
    print("=" * 60)
    print()


def main():
    print_header()

    done = False
    try:
        done = try_torch_nccl()
    except Exception as exc:  # noqa: BLE001
        print(f"[torch] 运行失败，退化到 numpy 模拟：{exc}")
        done = False

    if not done:
        print("[信息] 未使用 torch NCCL 路径，改用 numpy 多进程模拟。\n")
        numpy_simulation()

    print_summary()
    print("[Python] 演示结束。\n")


if __name__ == "__main__":
    main()
