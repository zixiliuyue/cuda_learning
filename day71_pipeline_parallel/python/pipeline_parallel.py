#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 71: 流水线并行（Pipeline Parallel）

把一个深度模型「按层」切分成若干 stage，分别放到不同 GPU 上；再把一个
mini-batch 拆成多个 micro-batch，让它们像工厂流水线一样依次流过各 stage，
从而让多张卡尽量同时忙碌。

本天要讲清三件事：
  1. 模型按层切分（layer partition）到多个 stage；
  2. micro-batch 流水线调度：GPipe 的「F-then-B」与 Megatron 的「1F1B」；
  3. 流水线气泡（bubble）从何而来、如何用 micro-batch 数量摊薄它。

运行策略（单机单卡也能学）：
  分布式的多 stage 通信在单卡上无法真实跑起来，因此本脚本用「单进程模拟
  多 stage」——每个 stage 只是一段线性层，激活在 stage 之间用普通张量/数组
  传递。调度时间线则用纯逻辑推演 + ASCII 图展示，不依赖真实多卡。
  有 torch 就用 torch 做数值前向/反向，无 torch 退化到 numpy。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量，风格与全项目一致）
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
#  1. 模型按层切分到多个 stage
# ============================================================
def explain_partition():
    print(SEP)
    print("流水线并行核心思想：模型「按层」切分到多张卡")
    print(SEP)
    print("""
  一个 8 层的网络，切成 4 个 stage（每卡 2 层）：

    GPU0          GPU1          GPU2          GPU3
   [L0 L1]  -->  [L2 L3]  -->  [L4 L5]  -->  [L6 L7]
      激活          激活          激活          输出
      发送          发送          发送

  - 前向：激活 activation 从前一个 stage 传到后一个 stage；
  - 反向：梯度 gradient 从后一个 stage 传回前一个 stage；
  - 与「张量并行」不同：TP 切的是「单层内部」的矩阵，PP 切的是「层与层之间」。

  优点：每张卡只放模型的一部分，能训练放不下单卡的超大模型。
  代价：天生存在「气泡」——首尾 stage 在填充/排空阶段会空闲。
""")


class PipelineStage:
    """模拟一个 stage：内部是若干层的线性变换 y = x @ W。

    为了让「前向/反向在 stage 间传递」可被验证，这里用最简单的线性层，
    这样把所有 stage 串起来 == 一个大线性层，便于和参考实现对拍。
    """

    def __init__(self, stage_id, in_dim, out_dim, seed):
        self.stage_id = stage_id
        if TORCH_AVAILABLE:
            g = torch.Generator().manual_seed(seed)
            self.W = torch.randn(in_dim, out_dim, generator=g) * 0.1
        else:
            rng = np.random.default_rng(seed)
            self.W = (rng.standard_normal((in_dim, out_dim)) * 0.1).astype(np.float32)

    def forward(self, x):
        """前向：接收上一个 stage 的激活 x，输出本 stage 的激活。"""
        if TORCH_AVAILABLE:
            return x @ self.W
        return x @ self.W


# ============================================================
#  2. 调度时间线模拟（F-then-B 与 1F1B）
# ============================================================
def simulate_gpipe(num_stages, num_micro):
    """GPipe「F-then-B」调度：先把所有 micro-batch 的前向做完，再统一做反向。

    返回一个 timeline：timeline[s] 是长度为 total_slots 的字符串列表，
    每个时隙里是该 stage 正在处理的任务（"F3"=第3个micro的前向，"·"=空闲）。
    """
    # 前向阶段：micro m 在 stage s 的开始时刻 = m + s（每步 1 个时间单位）
    # 反向阶段：紧接在所有前向之后，逆序流过各 stage。
    fwd_len = num_micro + num_stages - 1          # 前向总时隙
    bwd_len = num_micro + num_stages - 1          # 反向总时隙
    total = fwd_len + bwd_len
    timeline = [["·"] * total for _ in range(num_stages)]

    # 前向：micro m 到达 stage s 的时刻为 m + s
    for m in range(num_micro):
        for s in range(num_stages):
            t = m + s
            timeline[s][t] = f"F{m}"
    # 反向：从最后一个 stage 开始往回，micro m 在 stage s 的反向时刻
    # 放在前向全部结束之后。反向顺序 stage 由大到小。
    for m in range(num_micro):
        for s in range(num_stages):
            t = fwd_len + m + (num_stages - 1 - s)
            timeline[s][t] = f"B{m}"
    return timeline, total


def simulate_1f1b(num_stages, num_micro):
    """1F1B 调度：稳定期里每个 stage「做一次前向、紧跟做一次反向」交替进行。

    1F1B 相比 GPipe 的关键优势：激活显存占用大幅下降（不用同时存下所有
    micro-batch 的前向激活），气泡大小与 GPipe 相同量级但显存友好。
    这里用一个简化但自洽的时隙分配来展示交替特征。
    """
    warmup = num_stages - 1                        # 预热：只前向
    total = 2 * (num_micro + num_stages - 1)
    timeline = [["·"] * total for _ in range(num_stages)]

    for s in range(num_stages):
        # 预热阶段：stage s 先做 (warmup - s) 个前向填满流水线
        t = s                                      # stage 越靠后，起步越晚
        fwd_done = 0
        bwd_done = 0
        n_warm = warmup - s
        # 预热前向
        for _ in range(n_warm):
            if t < total and fwd_done < num_micro:
                timeline[s][t] = f"F{fwd_done}"
                fwd_done += 1
                t += 1
        # 稳定期：1F1B 交替
        while fwd_done < num_micro and t + 1 < total:
            timeline[s][t] = f"F{fwd_done}"
            fwd_done += 1
            t += 1
            timeline[s][t] = f"B{bwd_done}"
            bwd_done += 1
            t += 1
        # 排空：剩余反向
        while bwd_done < num_micro and t < total:
            timeline[s][t] = f"B{bwd_done}"
            bwd_done += 1
            t += 1
    return timeline, total


def print_timeline(name, timeline, total):
    print(f"\n[{name}] 时间线（行=stage，列=时隙，F=前向 B=反向 ·=气泡）")
    header = "        " + "".join(f"{t:>4}" for t in range(total))
    print(header)
    for s, row in enumerate(timeline):
        cells = "".join(f"{c:>4}" for c in row)
        print(f"  GPU{s}: {cells}")


def pipeline_efficiency(num_stages, num_micro):
    """计算流水线效率与气泡占比。

    理想（无气泡）耗时 = 每个 micro 的一次前向+一次反向串起来的有效工作量。
    气泡开销来自流水线的「填充 + 排空」，其大小 ≈ (stage 数 - 1)。
    效率 = 有效工作 / (有效工作 + 气泡)。
    """
    p = num_stages
    m = num_micro
    # 经典 GPipe 气泡比公式：bubble_fraction = (p - 1) / (m + p - 1)
    bubble = (p - 1) / (m + p - 1)
    efficiency = 1.0 - bubble
    return bubble, efficiency


# ============================================================
#  3. 数值前向/反向：验证 stage 串联 == 单个大层
# ============================================================
def run_numeric_forward(num_stages, batch, dim):
    """把激活依次流过各 stage，验证「流水线前向」的数值正确性。

    构造：每个 stage 都是 dim->dim 的线性层。串联所有 stage 后，
    等价于一个权重为 W0 @ W1 @ ... 的单一线性层，可作为参考对拍。
    """
    stages = [PipelineStage(s, dim, dim, seed=100 + s) for s in range(num_stages)]

    if TORCH_AVAILABLE:
        x0 = torch.randn(batch, dim)
    else:
        x0 = np.random.default_rng(0).standard_normal((batch, dim)).astype(np.float32)

    # 流水线前向：逐 stage 传递激活
    x = x0
    for st in stages:
        x = st.forward(x)          # 前一 stage 输出 -> 后一 stage 输入
    pipe_out = x

    # 参考：把所有权重先乘起来，一次算出
    W = stages[0].W
    for st in stages[1:]:
        W = W @ st.W
    ref_out = x0 @ W

    if TORCH_AVAILABLE:
        max_err = (pipe_out - ref_out).abs().max().item()
    else:
        max_err = float(np.max(np.abs(pipe_out - ref_out)))
    return max_err


# ============================================================
#  主流程
# ============================================================
def main():
    explain_partition()

    num_stages = 4
    num_micro = 8

    print(SEP)
    print(f"调度演示：{num_stages} 个 stage，{num_micro} 个 micro-batch")
    print(SEP)

    tl_g, tot_g = simulate_gpipe(num_stages, num_micro)
    print_timeline("GPipe / F-then-B", tl_g, tot_g)
    print("  特征：所有前向先做完，再统一做反向 → 需同时保存全部 micro 的激活（显存大）。")

    tl_b, tot_b = simulate_1f1b(num_stages, num_micro)
    print_timeline("1F1B（Megatron）", tl_b, tot_b)
    print("  特征：预热后前向/反向交替 → 激活显存只需 ~stage 数份（显存友好）。")

    print()
    print(SEP)
    print("流水线气泡（bubble）与效率分析")
    print(SEP)
    print("  气泡比公式： bubble = (p - 1) / (m + p - 1)   （p=stage数, m=micro数）")
    print(f"  {'micro数 m':>10} | {'气泡占比':>10} | {'流水线效率':>10}")
    print("  " + "-" * 40)
    for m in [1, 2, 4, 8, 16, 32]:
        bubble, eff = pipeline_efficiency(num_stages, m)
        print(f"  {m:>10} | {bubble*100:>9.1f}% | {eff*100:>9.1f}%")
    print("\n  结论：micro-batch 越多，气泡被摊得越薄，流水线效率越高；")
    print("        但 micro 太多会增加调度开销，需要折中。")

    print()
    print(SEP)
    print("数值验证：stage 串联前向 == 单个等价大层")
    print(SEP)
    max_err = run_numeric_forward(num_stages, batch=16, dim=32)
    backend = "torch" if TORCH_AVAILABLE else "numpy"
    print(f"  [{backend}] 流水线前向 vs 参考实现 最大误差 = {max_err:.2e}")
    print("  [结果] " + ("通过 ✓" if max_err < 1e-3 else "偏差过大 ✗"))

    print()
    print(SEP)
    if MULTI_GPU:
        print("环境：检测到多 GPU。真实多 stage 流水线可用 torch 的 Pipe / "
              "PiPPy / Megatron-LM 实现；本脚本用单进程模拟以保证可复现。")
    elif GPU_AVAILABLE:
        print("环境：单 GPU。真实流水线并行需多卡，本脚本用单进程模拟调度与前向。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 无 GPU。已用 torch(CPU) 做数值前向 + 逻辑推演调度。")
    else:
        print("环境：无 PyTorch。已用 NumPy 做数值前向 + 逻辑推演调度。")
    print(SEP)


if __name__ == "__main__":
    main()
