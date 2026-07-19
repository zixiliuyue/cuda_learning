#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 74: 混合并行 —— 3D 并行（DP + TP + PP）

训练超大模型时，单一并行维度不够用，工业界把三种并行组合成「3D 并行」：
  - DP（数据并行 Data Parallel）：不同 GPU 处理不同 batch，梯度 AllReduce；
  - TP（张量并行 Tensor Parallel）：把单层权重矩阵切到组内多卡，层内 AllReduce；
  - PP（流水线并行 Pipeline Parallel）：模型按层切成多 stage，stage 间 send/recv。

给定 GPU 总数 W，选择 (dp, tp, pp) 使得 dp*tp*pp = W，就把这 W 张卡组织成
一个三维网格。每张卡在网格中有一个 (pp_rank, dp_rank, tp_rank) 坐标，决定它
持有模型的哪一部分、和谁通信。

本天目标：
  1. 演示如何把 W 张 GPU 划分成 (dp, tp, pp) 网格；
  2. 打印每张卡的 rank 坐标与职责；
  3. 对比不同划分方案的通信量。

运行策略：纯逻辑推演（rank 坐标映射 + 通信量估算），无需真实多卡。
有 torch 仅用于探测环境，无 torch 也完全不影响。
"""

# ------------------------------------------------------------
# 环境探测（三布尔量）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

GPU_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()
MULTI_GPU = GPU_AVAILABLE and torch.cuda.device_count() > 1

SEP = "=" * 64


# ============================================================
#  0. 原理讲解
# ============================================================
def explain_3d():
    print(SEP)
    print("3D 并行核心思想：DP × TP × PP 三维网格")
    print(SEP)
    print("""
  W 张 GPU 组织成三维网格： dp * tp * pp = W

    维度   切什么           通信原语          通信频率
    ----------------------------------------------------------
    TP     单层内部权重     层内 AllReduce    最高（每层多次）
    PP     层与层之间       stage send/recv   低（stage 边界）
    DP     不同 batch       梯度 AllReduce     每步一次（可重叠）

  经验法则（通信成本从高到低安排）：
   - TP 放「机内」（同一节点 NVLink，带宽最高，因为通信最频繁）；
   - PP 跨节点也能接受（通信少，只在 stage 边界传激活）；
   - DP 放最外层（每步一次梯度 AllReduce，可与反向计算重叠）。

  一张卡的坐标 (pp_rank, dp_rank, tp_rank) 决定它：
   - 属于哪个 pipeline stage（pp_rank）；
   - 属于哪个数据并行副本（dp_rank）；
   - 在张量并行组内的位置（tp_rank）。
""")


# ============================================================
#  1. 构建 3D 网格：把 world rank 映射到 (pp, dp, tp) 坐标
# ============================================================
def build_grid(dp, tp, pp):
    """把 world_size = dp*tp*pp 个 GPU 映射到三维坐标。

    采用常见的排布顺序：world_rank = pp_rank*(dp*tp) + dp_rank*tp + tp_rank
    即 TP 变化最快（相邻 rank 属同一 TP 组），PP 变化最慢。
    返回列表，每项是 dict 描述一张卡。
    """
    world = dp * tp * pp
    grid = []
    for r in range(world):
        pp_rank = r // (dp * tp)
        rem = r % (dp * tp)
        dp_rank = rem // tp
        tp_rank = rem % tp
        grid.append({
            "world_rank": r,
            "pp_rank": pp_rank,
            "dp_rank": dp_rank,
            "tp_rank": tp_rank,
        })
    return grid


def describe_duties(card, dp, tp, pp, num_layers):
    """根据坐标描述该卡的职责：负责哪些层、属于哪些通信组。"""
    layers_per_stage = num_layers // pp
    lo = card["pp_rank"] * layers_per_stage
    hi = lo + layers_per_stage - 1
    return (f"层 L{lo}-L{hi}（第 {card['pp_rank']} 个 stage）| "
            f"TP组内第 {card['tp_rank']}/{tp} 片 | "
            f"DP副本 {card['dp_rank']}/{dp}")


# ============================================================
#  2. 通信量估算：对比不同 (dp, tp, pp) 划分
# ============================================================
def estimate_comm(dp, tp, pp, num_layers, hidden, batch, seq):
    """粗略估算每步各维度通信量（相对单位），帮助理解划分权衡。

    - TP：每层前向 + 反向各需 ~2 次 AllReduce，通信量 ~ 层数 * 激活大小 * (tp-1)/tp。
    - PP：stage 边界传激活，通信量 ~ (pp-1) * 激活大小。
    - DP：每步一次梯度 AllReduce，通信量 ~ 参数量 * (dp-1)/dp。
    这里用简化的相对量级（非精确 byte 数），仅用于横向对比。
    """
    act = batch * seq * hidden               # 单份激活规模
    params_per_stage = (num_layers // pp) * hidden * hidden * 4  # 每 stage 参数量粗估

    tp_comm = num_layers * act * (0 if tp == 1 else 2 * (tp - 1) / tp)
    pp_comm = (pp - 1) * act
    dp_comm = params_per_stage * (0 if dp == 1 else (dp - 1) / dp)
    return tp_comm, pp_comm, dp_comm


def factorizations(world):
    """列出 world 的所有 (dp, tp, pp) 分解（dp*tp*pp = world）。"""
    combos = []
    for pp in range(1, world + 1):
        if world % pp:
            continue
        rest = world // pp
        for tp in range(1, rest + 1):
            if rest % tp:
                continue
            dp = rest // tp
            combos.append((dp, tp, pp))
    return combos


# ============================================================
#  主流程
# ============================================================
def main():
    explain_3d()

    world = 8
    dp, tp, pp = 2, 2, 2
    num_layers = 8

    print(SEP)
    print(f"示例划分：{world} 张 GPU → (dp={dp}, tp={tp}, pp={pp})，"
          f"模型共 {num_layers} 层")
    print(SEP)
    assert dp * tp * pp == world, "dp*tp*pp 必须等于 GPU 总数"

    grid = build_grid(dp, tp, pp)
    print(f"  {'world':>5} | {'pp':>3} {'dp':>3} {'tp':>3} | 职责")
    print("  " + "-" * 58)
    for card in grid:
        duty = describe_duties(card, dp, tp, pp, num_layers)
        print(f"  {card['world_rank']:>5} | "
              f"{card['pp_rank']:>3} {card['dp_rank']:>3} {card['tp_rank']:>3} | {duty}")

    print()
    print(SEP)
    print("同一通信组的成员（以 world_rank 0 为例）")
    print(SEP)
    c0 = grid[0]
    tp_group = [c["world_rank"] for c in grid
                if c["pp_rank"] == c0["pp_rank"] and c["dp_rank"] == c0["dp_rank"]]
    dp_group = [c["world_rank"] for c in grid
                if c["pp_rank"] == c0["pp_rank"] and c["tp_rank"] == c0["tp_rank"]]
    pp_group = [c["world_rank"] for c in grid
                if c["dp_rank"] == c0["dp_rank"] and c["tp_rank"] == c0["tp_rank"]]
    print(f"  rank0 的 TP 组（层内 AllReduce 伙伴）: {tp_group}")
    print(f"  rank0 的 DP 组（梯度 AllReduce 伙伴）: {dp_group}")
    print(f"  rank0 的 PP 组（stage send/recv 链）: {pp_group}")

    print()
    print(SEP)
    print(f"通信量对比：{world} 卡的所有 (dp, tp, pp) 划分方案（相对量级）")
    print(SEP)
    hidden, batch, seq = 1024, 8, 512
    print(f"  {'(dp,tp,pp)':>12} | {'TP通信':>10} | {'PP通信':>10} | "
          f"{'DP通信':>10} | {'合计':>10}")
    print("  " + "-" * 62)
    for (d, t, p) in factorizations(world):
        if num_layers % p != 0:
            continue
        tpc, ppc, dpc = estimate_comm(d, t, p, num_layers, hidden, batch, seq)
        total = tpc + ppc + dpc
        print(f"  {f'({d},{t},{p})':>12} | {tpc:>10.2e} | {ppc:>10.2e} | "
              f"{dpc:>10.2e} | {total:>10.2e}")
    print("\n  观察：TP 度数越大，层内 AllReduce 通信越重 → 所以 TP 通常只在机内；")
    print("        DP 度数越大，梯度 AllReduce 越重，但可与反向计算重叠部分隐藏。")

    print()
    print(SEP)
    if MULTI_GPU:
        print(f"环境：检测到 {torch.cuda.device_count()} 张 GPU。真实 3D 并行由 "
              "Megatron-LM / DeepSpeed 编排；本脚本做 rank 映射与通信量推演。")
    elif GPU_AVAILABLE:
        print("环境：单 GPU。3D 并行需多卡，本脚本纯逻辑推演 rank 坐标与通信量。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 无 GPU。本天为纯逻辑推演，不依赖 GPU。")
    else:
        print("环境：无 PyTorch。本天为纯逻辑推演，不依赖任何库。")
    print(SEP)


if __name__ == "__main__":
    main()
