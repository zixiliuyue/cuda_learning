#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 67: 分布式基础 — 进程组与集合通信原语

分布式训练的地基是「集合通信」（Collective Communication）：多个进程（rank）
之间如何高效地交换/聚合张量。本天把最核心的 5 个原语讲清楚并给出可运行实现：
  1. broadcast      — 一个 rank 的数据广播给所有 rank
  2. reduce         — 所有 rank 的数据规约（sum/max...）到一个 rank
  3. all_reduce     — 所有 rank 规约后，结果发回所有 rank（DDP 梯度同步的核心）
  4. all_gather     — 每个 rank 收集到所有 rank 的数据拼接
  5. reduce_scatter — 先规约再按 rank 切分（ZeRO / 张量并行的核心）

运行策略（能不回退就不回退，但绝不卡死）：
  - torch.distributed 真实多进程（mp.spawn + gloo）在部分平台（尤其 Windows）
    容易卡死或报错，教学场景风险高；
  - 因此本脚本用【单进程 + numpy 数组列表】精确模拟「每个 rank 各持一份数据」，
    忠实复现每个原语的通信语义与前后数据变化，稳定可退出。
  - 若安装了 torch.distributed，末尾会打印真实多进程运行的等价代码骨架（不执行）。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量，风格对齐 Day41）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DIST_AVAILABLE = False
CUDA_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        import torch.distributed as dist  # noqa: F401
        DIST_AVAILABLE = True
    except ImportError:
        DIST_AVAILABLE = False
    CUDA_AVAILABLE = torch.cuda.is_available()

SEP = "=" * 64


# ============================================================
#  单进程「多 rank」模拟器
#  用一个 list，第 i 个元素代表 rank i 本地持有的数据（numpy 数组）
#  每个原语都返回「通信后」各 rank 的数据，语义与 torch.distributed 一致
# ============================================================
def make_ranks(world_size, dim=4, seed=0):
    """构造 world_size 个 rank 的本地数据。

    为了直观，rank i 的初始数据全部填充为 (i+1)，形如 [i+1, i+1, ...]。
    """
    rng = np.random.default_rng(seed)  # 预留：需要随机数据时可用
    _ = rng
    return [np.full(dim, float(i + 1)) for i in range(world_size)]


def broadcast(ranks, src):
    """broadcast：把 src rank 的数据复制给所有 rank。"""
    src_data = ranks[src].copy()
    return [src_data.copy() for _ in ranks]


def reduce(ranks, dst, op="sum"):
    """reduce：所有 rank 规约到 dst，其余 rank 数据保持不变。"""
    stacked = np.stack(ranks)
    agg = stacked.sum(axis=0) if op == "sum" else stacked.max(axis=0)
    out = [r.copy() for r in ranks]
    out[dst] = agg
    return out


def all_reduce(ranks, op="sum"):
    """all_reduce：规约后结果发回所有 rank（DDP 梯度平均就靠它）。"""
    stacked = np.stack(ranks)
    agg = stacked.sum(axis=0) if op == "sum" else stacked.max(axis=0)
    return [agg.copy() for _ in ranks]


def all_gather(ranks):
    """all_gather：每个 rank 都得到「所有 rank 数据拼接」的完整副本。"""
    full = np.concatenate([r.copy() for r in ranks])
    return [full.copy() for _ in ranks]


def reduce_scatter(ranks):
    """reduce_scatter：先按元素规约（sum），再把结果均分给各 rank。

    这是 all_reduce 的「上半场」：all_reduce == reduce_scatter + all_gather。
    每个 rank 只保留属于自己的那一分片，是 ZeRO 显存优化的通信基石。
    """
    world_size = len(ranks)
    stacked = np.stack(ranks)
    summed = stacked.sum(axis=0)            # 先全量规约
    chunks = np.array_split(summed, world_size)  # 再按 rank 切片
    return [chunks[i].copy() for i in range(world_size)]


# ============================================================
#  打印辅助
# ============================================================
def show(title, ranks):
    print(f"  {title}")
    for i, r in enumerate(ranks):
        print(f"    rank {i}: {np.array2string(r, precision=1)}")


def explain_concepts():
    print(SEP)
    print("核心概念：进程组（Process Group）与 rank")
    print(SEP)
    print("""
  术语            含义
  ------------------------------------------------------------
  world_size      参与训练的进程总数（≈ 用到的 GPU 数）
  rank            当前进程的全局编号，0 .. world_size-1，全局唯一
  local_rank      进程在本机内的编号（多机时用来选本地 GPU）
  backend         通信后端：GPU 用 nccl，CPU/调试用 gloo
  进程组          一组参与集合通信的进程集合，默认全体构成一个组

  一次典型初始化（torch.distributed）：
    dist.init_process_group(backend="nccl", rank=r, world_size=N)
    torch.cuda.set_device(local_rank)          # 每进程绑定一张卡
    ... 训练 ...
    dist.destroy_process_group()

  集合通信原语关系图（记住这一张就够）：
    all_reduce  =  reduce_scatter  +  all_gather
    all_gather  =  每个 rank 广播自己的分片给所有人
    reduce      =  all_reduce 但只有 dst 拿结果
""")


# ============================================================
#  各原语演示
# ============================================================
def demo_broadcast():
    print(SEP)
    print("原语 1：broadcast（src=0 → 所有 rank）")
    print(SEP)
    ranks = make_ranks(4)
    show("通信前：", ranks)
    out = broadcast(ranks, src=0)
    show("通信后（全部变成 rank0 的值 1）：", out)
    ok = all(np.allclose(o, ranks[0]) for o in out)
    print(f"  [校验] 所有 rank == rank0 ? {'通过 ✓' if ok else '失败 ✗'}\n")


def demo_reduce():
    print(SEP)
    print("原语 2：reduce（sum → dst=0，其余不变）")
    print(SEP)
    ranks = make_ranks(4)
    show("通信前：", ranks)
    out = reduce(ranks, dst=0, op="sum")
    show("通信后（rank0 = 1+2+3+4 = 10，其余不变）：", out)
    ok = np.allclose(out[0], 10.0)
    print(f"  [校验] rank0 == 10 ? {'通过 ✓' if ok else '失败 ✗'}\n")


def demo_all_reduce():
    print(SEP)
    print("原语 3：all_reduce（sum → 所有 rank）★ DDP 梯度同步核心")
    print(SEP)
    ranks = make_ranks(4)
    show("通信前：", ranks)
    out = all_reduce(ranks, op="sum")
    show("通信后（所有 rank 都 = 10）：", out)
    ok = all(np.allclose(o, 10.0) for o in out)
    print(f"  [校验] 所有 rank == 10 ? {'通过 ✓' if ok else '失败 ✗'}")
    print("  [提示] 若把 sum 除以 world_size 即得梯度平均，正是 DDP 做的事\n")


def demo_all_gather():
    print(SEP)
    print("原语 4：all_gather（拼接所有 rank 的数据）")
    print(SEP)
    ranks = make_ranks(4, dim=2)
    show("通信前（每 rank 2 个元素）：", ranks)
    out = all_gather(ranks)
    show("通信后（每 rank 都拿到 8 个元素的完整拼接）：", out)
    ok = out[0].size == 8 and all(np.allclose(o, out[0]) for o in out)
    print(f"  [校验] 每 rank 拿到完整 8 元素且一致 ? {'通过 ✓' if ok else '失败 ✗'}\n")


def demo_reduce_scatter():
    print(SEP)
    print("原语 5：reduce_scatter（先 sum 再按 rank 切分）★ ZeRO 基石")
    print(SEP)
    ranks = make_ranks(4)
    show("通信前（每 rank 4 元素）：", ranks)
    print("  规约(sum)后完整向量 = [10,10,10,10]，再均分给 4 个 rank：")
    out = reduce_scatter(ranks)
    show("通信后（每 rank 只留 1 个分片）：", out)
    # 验证 all_reduce == reduce_scatter + all_gather
    combined = all_gather(out)[0]
    ar = all_reduce(make_ranks(4))[0]
    ok = np.allclose(combined, ar)
    print(f"  [校验] reduce_scatter + all_gather == all_reduce ? "
          f"{'通过 ✓' if ok else '失败 ✗'}\n")


def print_real_dist_skeleton():
    print(SEP)
    print("真实多进程运行骨架（torch.distributed，本脚本不执行以避免卡死）")
    print(SEP)
    print("""
  # save as run_dist.py, 然后:
  #   torchrun --nproc_per_node=4 run_dist.py       # 推荐
  #   或 python -m torch.distributed.run --nproc_per_node=4 run_dist.py
  import os, torch, torch.distributed as dist

  def main():
      dist.init_process_group(backend="gloo")       # 无 GPU 用 gloo
      rank = dist.get_rank()
      world = dist.get_world_size()
      x = torch.full((4,), float(rank + 1))
      dist.all_reduce(x, op=dist.ReduceOp.SUM)       # 与本脚本 all_reduce 等价
      print(f"rank {rank}/{world} after all_reduce: {x.tolist()}")
      dist.destroy_process_group()

  if __name__ == "__main__":
      main()
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_concepts()
    demo_broadcast()
    demo_reduce()
    demo_all_reduce()
    demo_all_gather()
    demo_reduce_scatter()
    print_real_dist_skeleton()

    print(SEP)
    if DIST_AVAILABLE and CUDA_AVAILABLE:
        print("环境：torch.distributed + CUDA 就绪。可用 torchrun 跑上面的真实多进程骨架。")
    elif DIST_AVAILABLE:
        print("环境：有 torch.distributed 但无 CUDA GPU。可用 gloo 后端跑真实多进程骨架。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 distributed 模块。已用 numpy 单进程精确模拟各原语。")
    else:
        print("环境：未安装 PyTorch。已用 numpy 单进程精确模拟各原语。")
    print("说明：本脚本用单进程模拟多 rank，语义与真实集合通信一致，稳定可退出。")
    print(SEP)


if __name__ == "__main__":
    main()
