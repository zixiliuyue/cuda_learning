#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 79: 训练性能监控与 profiling 体系 — 找出训练的真正瓶颈

训练慢，到底慢在哪？盲目优化等于瞎猜。工程上要先建立 profiling 体系，
把一个训练 step 拆成几个阶段分别计时，看谁占大头，再对症下药：
  - 前向 forward
  - 反向 backward
  - 通信 communication（多卡梯度同步 all-reduce）
  - 优化器 optimizer step

再根据各阶段占比判断瓶颈类型：
  - 计算受限（compute-bound）：前向/反向占大头 → 上 AMP、算子融合、更好 kernel
  - 通信受限（comm-bound）：通信占大头 → 重叠、分桶、更快互联、梯度压缩
  - 显存受限（memory-bound）：显存不够被迫小 batch → 梯度检查点、ZeRO/FSDP

本天搭建一个可运行的训练 profiling：
  - 有 torch：跑一个真实小模型的训练 step，用 torch.profiler（若可用）
    或手动计时统计各阶段耗时；
  - 无 torch：用 numpy 模拟一个 MLP 训练 step，手动计时各阶段。
两种路径都打印各阶段耗时占比 + 瓶颈判定报告，任何环境稳定退出。
"""

import time
import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量：torch / CUDA / profiler 是否可用）
# ------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

CUDA_AVAILABLE = bool(TORCH_AVAILABLE and torch.cuda.is_available())

PROFILER_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        from torch.profiler import profile, ProfilerActivity  # noqa: F401
        PROFILER_AVAILABLE = True
    except Exception:
        PROFILER_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  1. 原理讲解
# ============================================================
def explain_profiling():
    print(SEP)
    print("训练 profiling：把一个 step 拆成 4 个阶段分别计时")
    print(SEP)
    print("""
  一个训练 step 的典型流水：
    forward → backward → [通信 all-reduce 梯度] → optimizer.step

  分阶段计时后看占比，判定瓶颈类型：
    ┌───────────────┬───────────────────────┬─────────────────────────┐
    │ 瓶颈类型       │ 表现                   │ 对策                     │
    ├───────────────┼───────────────────────┼─────────────────────────┤
    │ 计算受限       │ fwd/bwd 占大头         │ AMP / 融合 / 更好 kernel  │
    │ 通信受限       │ 通信占大头             │ 重叠 / 分桶 / 更快互联    │
    │ 显存受限       │ 被迫小 batch、OOM      │ 梯度检查点 / FSDP / ZeRO  │
    └───────────────┴───────────────────────┴─────────────────────────┘

  工具：GPU 上首选 torch.profiler / Nsight Systems；无 GPU 时用手动计时
        （time.perf_counter）也能拿到阶段占比，定位思路完全一致。
""")


# ============================================================
#  2. 计时上下文
# ============================================================
class Stage:
    """一个阶段的累计计时器。"""
    def __init__(self):
        self.total = 0.0

    def add(self, dt):
        self.total += dt


def timed(fn, *args, sync=False):
    """执行 fn 并返回 (结果, 耗时秒)。sync=True 时对 CUDA 做同步以准确计时。"""
    if sync and CUDA_AVAILABLE:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args)
    if sync and CUDA_AVAILABLE:
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


# ============================================================
#  3-A. torch 路径：真实小模型训练 step 计时
# ============================================================
def profile_with_torch(steps=20, batch=128, dim=512, hidden=1024):
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    print(f"  [torch] 设备={device}, 小模型 MLP({dim}->{hidden}->{dim}), "
          f"batch={batch}, steps={steps}")

    model = nn.Sequential(
        nn.Linear(dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, dim),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x = torch.randn(batch, dim, device=device)
    y = torch.randn(batch, dim, device=device)

    st_fwd, st_bwd, st_comm, st_opt = Stage(), Stage(), Stage(), Stage()

    # 预热（触发 lazy init / cudnn autotune），不计入统计
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out, dt = timed(lambda: loss_fn(model(x), y), sync=True)
        st_fwd.add(dt)

        _, dt = timed(lambda: out.backward(), sync=True)
        st_bwd.add(dt)

        # 单卡无真实通信：这里模拟「梯度同步」这一阶段（对梯度做一次归约拷贝）
        def fake_comm():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0)  # 占位：真实多卡此处是 all-reduce
        _, dt = timed(fake_comm, sync=True)
        st_comm.add(dt)

        _, dt = timed(lambda: opt.step(), sync=True)
        st_opt.add(dt)

    return {
        "forward": st_fwd.total,
        "backward": st_bwd.total,
        "communication": st_comm.total,
        "optimizer": st_opt.total,
    }, steps


# ============================================================
#  3-B. numpy 路径：模拟 MLP 训练 step 计时
# ============================================================
def profile_with_numpy(steps=30, batch=128, dim=256, hidden=512):
    print(f"  [numpy] 模拟 MLP({dim}->{hidden}->{dim}), batch={batch}, steps={steps}")
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((dim, hidden)) * 0.05
    W2 = rng.standard_normal((hidden, dim)) * 0.05
    m1 = np.zeros_like(W1); v1 = np.zeros_like(W1)
    m2 = np.zeros_like(W2); v2 = np.zeros_like(W2)
    X = rng.standard_normal((batch, dim))
    Y = rng.standard_normal((batch, dim))

    st_fwd, st_bwd, st_comm, st_opt = Stage(), Stage(), Stage(), Stage()

    def relu(z):
        return np.maximum(z, 0.0)

    for t in range(1, steps + 1):
        # forward
        t0 = time.perf_counter()
        H = relu(X @ W1)
        P = H @ W2
        diff = P - Y
        _ = np.mean(diff ** 2)
        st_fwd.add(time.perf_counter() - t0)

        # backward
        t0 = time.perf_counter()
        dP = 2.0 * diff / diff.size
        dW2 = H.T @ dP
        dH = dP @ W2.T
        dH[H <= 0] = 0.0
        dW1 = X.T @ dH
        st_bwd.add(time.perf_counter() - t0)

        # communication（单进程模拟梯度同步：对梯度做一次求和归约拷贝）
        t0 = time.perf_counter()
        _ = dW1.sum(); _ = dW2.sum()
        st_comm.add(time.perf_counter() - t0)

        # optimizer（Adam）
        t0 = time.perf_counter()
        for (W, dW, m, v) in [(W1, dW1, m1, v1), (W2, dW2, m2, v2)]:
            m[...] = 0.9 * m + 0.1 * dW
            v[...] = 0.999 * v + 0.001 * (dW ** 2)
            mhat = m / (1 - 0.9 ** t)
            vhat = v / (1 - 0.999 ** t)
            W -= 1e-3 * mhat / (np.sqrt(vhat) + 1e-8)
        st_opt.add(time.perf_counter() - t0)

    return {
        "forward": st_fwd.total,
        "backward": st_bwd.total,
        "communication": st_comm.total,
        "optimizer": st_opt.total,
    }, steps


# ============================================================
#  4. 分析报告：占比 + 瓶颈判定
# ============================================================
def report(stages, steps):
    print()
    print(SEP)
    print("Profiling 报告（各阶段耗时占比）")
    print(SEP)
    total = sum(stages.values()) or 1e-12
    order = ["forward", "backward", "communication", "optimizer"]
    name_cn = {"forward": "前向", "backward": "反向",
               "communication": "通信", "optimizer": "优化器"}
    print(f"  {'阶段':<12}{'总耗时(ms)':>14}{'每step(ms)':>14}{'占比':>10}   条形图")
    print(f"  {'-'*64}")
    for k in order:
        sec = stages[k]
        pct = sec / total * 100
        bar = "#" * max(1, int(round(pct / 2)))
        print(f"  {name_cn[k]:<11}{sec*1000:>13.2f}{sec/steps*1000:>14.3f}{pct:>9.1f}%   {bar}")
    print(f"  {'-'*64}")
    print(f"  {'合计':<11}{total*1000:>13.2f}{total/steps*1000:>14.3f}{100.0:>9.1f}%")

    # 瓶颈判定
    print()
    print(SEP)
    print("瓶颈判定")
    print(SEP)
    compute = stages["forward"] + stages["backward"]
    comm = stages["communication"]
    opt = stages["optimizer"]
    compute_pct = compute / total * 100
    comm_pct = comm / total * 100

    if comm_pct > 40:
        verdict = "通信受限（comm-bound）"
        advice = "→ 对策：通信/计算重叠(Day78)、梯度分桶、更快互联(NVLink)、梯度压缩"
    elif compute_pct > 60:
        verdict = "计算受限（compute-bound）"
        advice = "→ 对策：混合精度 AMP(Day26)、算子融合(Day43)、更优 kernel/Triton"
    else:
        verdict = "较均衡 / 视显存情况可能显存受限"
        advice = "→ 若曾 OOM 或被迫小 batch：梯度检查点(Day75)、FSDP/ZeRO(Day77/69)"
    print(f"  计算(前向+反向)占比 = {compute_pct:.1f}%，通信占比 = {comm_pct:.1f}%")
    print(f"  判定：{verdict}")
    print(f"  {advice}")
    print("  提示：单卡演示里通信阶段是占位模拟，真实多卡训练该项会显著上升。")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_profiling()

    print(SEP)
    print("搭建训练 profiling（分阶段计时）")
    print(SEP)
    if TORCH_AVAILABLE:
        if PROFILER_AVAILABLE:
            print("  torch.profiler 可用；本天用手动分阶段计时给出清晰的阶段占比。")
            print("  （torch.profiler 更适合看算子级火焰图，用法见 README。）")
        stages, steps = profile_with_torch()
    else:
        stages, steps = profile_with_numpy()

    report(stages, steps)

    print()
    print(SEP)
    if CUDA_AVAILABLE:
        print(f"环境：torch + CUDA 就绪，已在 GPU 上实测各阶段耗时。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU，已在 CPU 上实测各阶段耗时。")
    else:
        print("环境：未安装 PyTorch，已用 numpy 模拟训练 step 并计时。")
    print("要点：先分阶段计时找瓶颈，再对症优化（计算/通信/显存受限各有不同解法）。")
    print(SEP)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
