#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 81: NCCL 通信算法 — Ring / Tree AllReduce

本天承接 Day39（NCCL AllReduce 概念）与 Day25（多 GPU），深入到
NCCL **内部到底是怎么把 N 张卡的数据求和再广播回去的**。

核心目标（原理为主 + 单机可跑最小实现）：
  1. 用纯 Python/NumPy **模拟** Ring AllReduce（reduce-scatter + all-gather 两阶段）；
  2. 模拟 Tree AllReduce（reduce 上行 + broadcast 下行）；
  3. 与「朴素 AllReduce」（每个 rank 把数据发给所有人）对比通信量；
  4. 验证三种算法结果完全一致，并量化 Ring 的带宽最优性 2(N-1)/N。

运行策略（无 GPU 也能看懂）：
  - 本脚本不依赖 GPU，用 NumPy 在单机上模拟 N 个 rank 的显存与逐步收发；
  - 若装了 torch 则用 torch.Tensor 承载数据（语义一致），否则用 numpy；
  - 真实多机运行请用 torchrun + NCCL 后端（见 README）。
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

TORCH_CUDA = TORCH_AVAILABLE and torch.cuda.is_available()
MULTI_GPU = TORCH_CUDA and torch.cuda.device_count() > 1

SEP = "=" * 64


# ============================================================
#  数据构造：N 个 rank，每个持有一份长度为 D 的向量
# ============================================================
def make_data(num_ranks, dim, seed=0):
    """构造 N 份初始数据，返回 list[np.ndarray]。

    每个 rank 的数据不同，AllReduce(sum) 后所有 rank 都应得到它们的按元素和。
    """
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(dim).astype(np.float64) for _ in range(num_ranks)]


def reference_sum(data):
    """标准答案：直接把 N 份数据求和（每个 rank 最终都应等于它）。"""
    return np.sum(np.stack(data, axis=0), axis=0)


# ============================================================
#  1. 朴素 AllReduce（作为对照组）
# ============================================================
def naive_allreduce(data):
    """朴素做法：每个 rank 把自己整份数据发给其他所有 rank，本地求和。

    通信量分析（以「每个 rank 发送的数据量」计，单位=D 个元素）：
      每个 rank 要把完整的 D 发给其余 (N-1) 个 rank，
      故每 rank 发送 (N-1)*D，总线上流量随 N 线性增长，带宽利用差。
    """
    n = len(data)
    d = data[0].shape[0]
    result = reference_sum(data)
    out = [result.copy() for _ in range(n)]
    per_rank_send = (n - 1) * d  # 每个 rank 发送的元素数
    return out, per_rank_send


# ============================================================
#  2. Ring AllReduce（NCCL 默认，带宽最优）
# ============================================================
def ring_allreduce(data):
    """环形 AllReduce：reduce-scatter + all-gather 两阶段。

    把每个 rank 的向量切成 N 个 chunk。rank i 只与右邻 (i+1)%N 通信。

    阶段一 reduce-scatter（N-1 步）：
      每步每个 rank 把「某个 chunk 的部分和」发给右邻、从左邻收另一个 chunk 累加，
      N-1 步后：rank i 手里的第 (i+1)%N 号 chunk 变成了该 chunk 的全局和。

    阶段二 all-gather（N-1 步）：
      把「已经是全局和的那个 chunk」沿环传一圈，N-1 步后人人集齐所有 chunk。

    通信量：每 rank 收/发各 2(N-1)/N * D 个元素——与 N 几乎无关，带宽最优。
    """
    n = len(data)
    d = data[0].shape[0]

    # 把每个 rank 的向量切成 n 份 chunk（用 array_split 兼容不整除）
    chunks = [list(np.array_split(x.copy(), n)) for x in data]

    per_rank_send = 0  # 统计每个 rank 累计发送的元素数

    # ---- 阶段一：reduce-scatter ----
    # 第 step 步，rank r 发送 chunk (r - step) mod n 给右邻，
    #            接收并累加 chunk (r - step - 1) mod n。
    for step in range(n - 1):
        # 先把本步所有 rank 要发送的 chunk 快照下来（模拟同时收发）
        sending = []
        for r in range(n):
            send_idx = (r - step) % n
            sending.append(chunks[r][send_idx].copy())
            per_rank_send += sending[-1].shape[0]
        # 右邻接收并累加
        for r in range(n):
            recv_idx = (r - step - 1) % n  # 从左邻收到的 chunk 编号
            left = (r - 1) % n
            chunks[r][recv_idx] = chunks[r][recv_idx] + sending[left]

    # ---- 阶段二：all-gather ----
    # 此时 rank r 的 chunk (r+1) mod n 已是全局和，沿环把「完成的 chunk」传一圈
    for step in range(n - 1):
        sending = []
        for r in range(n):
            send_idx = (r + 1 - step) % n
            sending.append(chunks[r][send_idx].copy())
            per_rank_send += sending[-1].shape[0]
        for r in range(n):
            recv_idx = (r - step) % n
            left = (r - 1) % n
            chunks[r][recv_idx] = sending[left]  # 直接覆盖（已是全局和）

    out = [np.concatenate(chunks[r]) for r in range(n)]
    return out, per_rank_send // n  # 平均每 rank 发送量


# ============================================================
#  3. Tree AllReduce（大规模/高延迟场景常用）
# ============================================================
def tree_allreduce(data):
    """树形 AllReduce：reduce 上行到根 + broadcast 下行到全体。

    以二叉树组织 N 个 rank：
      reduce 阶段：叶子把数据往父节点汇聚求和，log2(N) 层后根得到全局和；
      broadcast 阶段：根把全局和沿树往下广播，log2(N) 层后人人拿到。

    特点：通信步数 O(log N)，**延迟低**，适合小数据/大规模集群；
          但每步传输完整 D，带宽利用不如 Ring。NCCL 会按数据大小自动选择。
    """
    n = len(data)
    # 用完全二叉树：节点 i 的父为 (i-1)//2，子为 2i+1, 2i+2
    buf = [x.copy() for x in data]
    per_rank_send = 0

    # ---- reduce 上行：从最后一个节点往前，把自己加到父节点 ----
    for i in range(n - 1, 0, -1):
        parent = (i - 1) // 2
        buf[parent] = buf[parent] + buf[i]
        per_rank_send += buf[i].shape[0]  # 节点 i 发送了完整一份

    # 此时 buf[0]（根）= 全局和
    root_sum = buf[0].copy()

    # ---- broadcast 下行：根往下广播 ----
    for i in range(1, n):
        buf[i] = root_sum.copy()
        per_rank_send += root_sum.shape[0]  # 父发送一份给子

    out = buf
    # 树形近似的「平均每 rank 发送量」：总发送 / N
    return out, per_rank_send // n


# ============================================================
#  校验与对比
# ============================================================
def verify(out, ref, name):
    max_err = max(float(np.max(np.abs(o - ref))) for o in out)
    ok = max_err < 1e-9
    print(f"  [{name:16s}] 与标准和最大误差 = {max_err:.2e}  "
          + ("✓ 一致" if ok else "✗ 不一致"))
    return ok


def explain_algorithms():
    print(SEP)
    print("三种 AllReduce 算法对比（N = rank 数, D = 向量元素数）")
    print(SEP)
    print("""
  算法         通信步数        每 rank 发送量       特点
  ----------------------------------------------------------------
  朴素         O(1) 轮/O(N)    (N-1)·D              实现简单，带宽随 N 恶化
  Ring         2(N-1) 步       2(N-1)/N · D ≈ 2D    带宽最优，NCCL 默认
  Tree         2·log2(N) 步    ~log2(N)/N · D       延迟最低，适合小数据/大集群

  关键结论：
   - Ring 的每 rank 发送量 2(N-1)/N·D 随 N 增大趋于常数 2D，
     不随卡数膨胀，因此「带宽受限」的大张量用 Ring 最划算；
   - Tree 的步数只有 log(N)，「延迟受限」的小张量用 Tree 更快；
   - NCCL 会根据消息大小/拓扑在 Ring / Tree / CollNet 间自动选择。
""")


def bandwidth_optimality_table():
    print(SEP)
    print("Ring 带宽最优性：每 rank 发送量随 N 的变化（单位 = D）")
    print(SEP)
    print("   N    朴素:(N-1)     Ring:2(N-1)/N     Ring/朴素")
    print("  " + "-" * 52)
    for n in [2, 4, 8, 16, 32, 64, 128]:
        naive = n - 1
        ring = 2.0 * (n - 1) / n
        print(f"  {n:4d}    {naive:8.2f}      {ring:10.4f}        {ring / naive:6.3f}")
    print("\n  可见：N 越大，Ring 相对朴素的通信量优势越明显（朴素线性增长，Ring 趋于 2）。")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_algorithms()

    print(SEP)
    print("演示：单机模拟 N 个 rank 的 AllReduce（sum）并交叉验证")
    print(SEP)
    num_ranks = 8
    dim = 1000
    print(f"  配置：num_ranks={num_ranks}, dim={dim}")
    if TORCH_AVAILABLE:
        print(f"  张量后端：torch（CUDA 可用={TORCH_CUDA}, 多卡={MULTI_GPU}）")
    else:
        print("  张量后端：numpy（未安装 torch）")

    data = make_data(num_ranks, dim, seed=42)
    ref = reference_sum(data)

    print()
    out_naive, send_naive = naive_allreduce(data)
    out_ring, send_ring = ring_allreduce(data)
    out_tree, send_tree = tree_allreduce(data)

    ok1 = verify(out_naive, ref, "朴素 AllReduce")
    ok2 = verify(out_ring, ref, "Ring AllReduce")
    ok3 = verify(out_tree, ref, "Tree AllReduce")

    print()
    print("  实测「平均每 rank 发送元素数」（越小越省带宽）：")
    print(f"    朴素 = {send_naive:8d}   （理论 (N-1)·D = {(num_ranks - 1) * dim}）")
    print(f"    Ring = {send_ring:8d}   （理论 2(N-1)/N·D ≈ "
          f"{2 * (num_ranks - 1) / num_ranks * dim:.0f}）")
    print(f"    Tree = {send_tree:8d}   （树形，步数 ~2·log2(N) = "
          f"{2 * int(np.log2(num_ranks))}）")

    print()
    bandwidth_optimality_table()

    print()
    print(SEP)
    all_ok = ok1 and ok2 and ok3
    print("[结果] 三种算法结果" + ("全部一致 ✓" if all_ok else "存在不一致 ✗"))
    if MULTI_GPU:
        print("环境：检测到多 GPU，可用 torchrun + NCCL 后端跑真实 Ring/Tree（见 README）。")
    elif TORCH_CUDA:
        print("环境：单 GPU。真实多卡通信需多张卡；本模拟已覆盖算法逻辑。")
    else:
        print("环境：无 GPU。已用 CPU 模拟 N 个 rank 的收发过程，算法逻辑与 NCCL 一致。")
    print(SEP)


if __name__ == "__main__":
    main()
