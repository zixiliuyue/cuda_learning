#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 78: 通信与计算重叠优化 — 让梯度 all-reduce 藏进反向计算里

数据并行训练里，反向传播每算完一层就产生该层的梯度，然后需要跨卡
all-reduce 同步。朴素做法是「先把整个反向算完，再统一 all-reduce」，
此时通信和计算完全串行，通信时间白白叠加在总时间上。

优化思路（DDP 的核心工程）：
  1. **梯度分桶（gradient bucketing）**：把若干层的梯度攒成一个 bucket，
     bucket 一满就立刻发起 all-reduce（减少小通信的启动开销）。
  2. **异步通信 + 计算重叠**：反向计算靠近输入的层时，靠近输出的层的
     梯度 all-reduce 已经在后台网卡上跑了。通信被「藏」进后续计算，
     总时间 ≈ max(计算, 通信) 而非 计算 + 通信。

本天用【时间线 ASCII 图 + 数值模拟】展示串行 vs 重叠的耗时差异，
并计算加速比。纯 CPU/numpy 即可运行，任何环境稳定退出。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量：torch / CUDA / 分布式）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

CUDA_AVAILABLE = bool(TORCH_AVAILABLE and torch.cuda.is_available())

DIST_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        import torch.distributed as dist  # noqa: F401
        DIST_AVAILABLE = True
    except ImportError:
        DIST_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  1. 原理讲解
# ============================================================
def explain_overlap():
    print(SEP)
    print("通信与计算重叠：把梯度 all-reduce 藏进反向计算")
    print(SEP)
    print("""
  反向传播是【从输出层往输入层】逐层算梯度的。每算完一层就得到该层梯度。

  朴素（串行）：等所有层反向算完，再一次性 all-reduce 全部梯度。
    时间 = 反向计算时间 + 通信时间   （两段首尾相接，纯叠加）

  重叠（DDP 实际做法）：
    ① 分桶：几层梯度攒成一个 bucket，桶满立即触发 all-reduce；
    ② 异步：all-reduce 交给后台（NCCL 流/网卡），主流继续算前面的层；
    ③ 结果：后层的通信与前层的计算在时间上并行。
    时间 ≈ max(反向计算时间, 通信时间) + 少量收尾
""")


# ============================================================
#  2. 时间线 ASCII 图：串行 vs 重叠
# ============================================================
def draw_timeline(layer_compute, layer_comm):
    """用字符时间线直观展示串行与重叠的差异。
    layer_compute[i] / layer_comm[i]：第 i 层反向的计算 / 通信耗时（相对单位）。
    """
    n = len(layer_compute)
    scale = 1  # 每单位耗时画几个字符

    def bar(ch, length):
        return ch * max(1, int(round(length * scale)))

    print(SEP)
    print("时间线对比（C=计算 反向, R=通信 allreduce；层顺序为反向 L{n}->L1）")
    print(SEP)

    # --- 串行：先全部计算，再全部通信 ---
    print("\n[串行 serial]  先算完所有层，再统一 all-reduce：")
    line_c = ""
    for i in range(n):
        line_c += bar("C", layer_compute[i])
    line_r = " " * len(line_c)
    for i in range(n):
        line_r += bar("R", layer_comm[i])
    print("  计算: " + line_c)
    print("  通信: " + line_r)
    serial_total = sum(layer_compute) + sum(layer_comm)
    print(f"  总耗时 = 计算({sum(layer_compute)}) + 通信({sum(layer_comm)}) = {serial_total}")

    # --- 重叠：某层算完就异步发它的 allreduce，同时继续算下一层 ---
    print("\n[重叠 overlap] 每层算完立即异步 all-reduce，主流继续算下一层：")
    # 计算流：逐层顺序占用
    comp_line = ""
    for i in range(n):
        comp_line += bar("C", layer_compute[i])
    # 通信流：第 i 层通信最早在其计算结束后开始，且不能与上一层通信重叠（单一网卡）
    comm_start = [0.0] * n
    comm_end = [0.0] * n
    comp_end = np.cumsum(layer_compute)
    prev_comm_end = 0.0
    for i in range(n):
        start = max(comp_end[i], prev_comm_end)  # 计算完成 且 上个通信完成
        comm_start[i] = start
        comm_end[i] = start + layer_comm[i]
        prev_comm_end = comm_end[i]
    comm_line = ""
    cursor = 0.0
    for i in range(n):
        gap = comm_start[i] - cursor
        comm_line += " " * max(0, int(round(gap * scale)))
        comm_line += bar("R", layer_comm[i])
        cursor = comm_end[i]
    print("  计算: " + comp_line)
    print("  通信: " + comm_line)
    overlap_total = max(comp_end[-1], comm_end[-1])
    print(f"  总耗时 ≈ max(计算完成={comp_end[-1]:.0f}, 通信完成={comm_end[-1]:.0f}) = {overlap_total:.0f}")

    return serial_total, overlap_total


# ============================================================
#  3. 分桶（bucketing）效果模拟
# ============================================================
def demo_bucketing(n_layers=16, launch_overhead=0.5, per_layer_grad=1.0):
    print()
    print(SEP)
    print("演示：梯度分桶（bucketing）如何降低通信启动开销")
    print(SEP)
    print("  每次 all-reduce 都有固定启动开销（延迟）。层越多，小通信越多，")
    print("  启动开销累加越严重。分桶把多层梯度合并成一次大通信，摊薄开销。\n")
    print(f"  {'bucket大小(层)':<16}{'通信次数':>10}{'启动开销总计':>16}{'数据传输':>12}")
    print(f"  {'-'*54}")
    for bucket in [1, 2, 4, 8, 16]:
        n_comm = int(np.ceil(n_layers / bucket))
        overhead = n_comm * launch_overhead
        transfer = n_layers * per_layer_grad  # 数据量不变
        total = overhead + transfer
        print(f"  {bucket:<16}{n_comm:>10}{overhead:>14.1f}  {transfer:>10.1f}  (合计 {total:.1f})")
    print("\n  结论：bucket 越大，通信次数越少、启动开销越低；但过大会推迟首个")
    print("        all-reduce 的发起时机，削弱重叠。实践中需折中（DDP 默认 25MB）。")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_overlap()

    # 构造一个 6 层反向的耗时场景（相对单位）
    rng = np.random.default_rng(3)
    n = 6
    layer_compute = [4, 3, 5, 3, 4, 3]           # 各层反向计算耗时
    layer_comm = [3, 2, 3, 2, 3, 2]              # 各层梯度 all-reduce 耗时

    serial_total, overlap_total = draw_timeline(layer_compute, layer_comm)

    print()
    print(SEP)
    print("加速比")
    print(SEP)
    speedup = serial_total / overlap_total
    saved = serial_total - overlap_total
    print(f"  串行总耗时 = {serial_total}")
    print(f"  重叠总耗时 ≈ {overlap_total:.0f}")
    print(f"  节省 = {saved:.0f}（{saved/serial_total*100:.1f}%），加速比 = {speedup:.2f}x")
    print("  说明：通信被藏进后续计算，总时间趋近 max(计算, 通信) 而非二者之和。")

    demo_bucketing()

    print()
    print(SEP)
    if CUDA_AVAILABLE and DIST_AVAILABLE:
        print("环境：torch + CUDA + 分布式就绪。真实 DDP 已内建梯度分桶与异步重叠")
        print("      （DistributedDataParallel 的 bucket_cap_mb / 反向 hook 自动完成）。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch，无 CUDA/未用分布式。已用数值模拟重叠加速原理。")
    else:
        print("环境：未安装 PyTorch。已用 numpy 数值模拟重叠加速原理。")
    print("要点：反向逐层产生梯度→分桶→异步 all-reduce，把通信藏进计算，逼近 max(计算,通信)。")
    print(SEP)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
