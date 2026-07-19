#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 80: 阶段项目 — 多卡训练一个小模型（第十二阶段收官）

整合第十二阶段所学，端到端训练一个小模型，用【单进程模拟多 rank 的 DDP】
把数据并行的核心闭环真正跑通：
  1. 数据并行：把一个 batch 切成 world_size 份，每个 rank 各拿一份；
  2. 各 rank 在本地前向 + 反向，得到各自的梯度；
  3. 梯度同步：对所有 rank 的梯度做 all-reduce（求平均）——这是 DDP 的灵魂；
  4. 每个 rank 用「同步后的平均梯度」更新参数，保证各 rank 参数始终一致。

并做扩展性分析：在 1 / 2 / 4 rank 下测吞吐（samples/s）与并行效率，
打印扩展性报告，理解「为什么 N 卡达不到 N 倍加速」。

为什么单进程模拟：单机单卡起 torch.distributed + mp.spawn 容易卡死；
DDP 的数学本质（梯度 all-reduce 求平均 = 大 batch 单卡训练）用单进程即可
完整、稳定地复现，学习闭环一致。有 torch 用 torch，无则 numpy。
"""

import time
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
def explain_ddp():
    print(SEP)
    print("数据并行 DDP：梯度 all-reduce 求平均 = 等价于大 batch 单卡训练")
    print(SEP)
    print("""
  单步 DDP 训练闭环（world_size 张卡）：
    ① 切数据：全局 batch 切成 N 份，rank r 拿第 r 份（各卡数据不同）
    ② 本地算：每个 rank 用【相同的参数】各自前向+反向 → 得到各自梯度 g_r
    ③ all-reduce：g = (g_0 + g_1 + ... + g_{N-1}) / N  （对所有 rank 求平均）
    ④ 同步更新：每个 rank 用同一个 g 更新参数 → 各卡参数始终保持一致

  数学等价：平均各 rank 的梯度，恰好等于「把 N 份数据拼成一个大 batch」
  在单卡上算的梯度。所以 DDP ≡ 扩大有效 batch size 的单卡训练。

  扩展性（为什么 N 卡 < N 倍加速）：
    - 通信开销：梯度 all-reduce 的时间不随算力线性下降；
    - 不可并行部分（Amdahl）：数据加载、优化器 step 等；
    => 并行效率 = 实际加速比 / N < 1，卡越多通常效率越低。
""")


# ============================================================
#  2-A. numpy 版：单进程模拟多 rank DDP 训练一个 MLP 回归
# ============================================================
def make_dataset(n=2048, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    true_w = rng.standard_normal((dim, 1))
    Y = X @ true_w + 0.05 * rng.standard_normal((n, 1))  # 线性可学 + 噪声
    return X, Y


def train_ddp_numpy(world_size, X, Y, hidden=64, epochs=8, global_batch=256, lr=5e-3, seed=1):
    """单进程模拟 world_size 个 rank 的 DDP 训练，返回 (最终loss, 总耗时, 处理样本数)。"""
    n, dim = X.shape
    rng = np.random.default_rng(seed)
    # 所有 rank 共享同一份初始参数（DDP 要求初始一致）
    W1 = rng.standard_normal((dim, hidden)) * 0.1
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, 1)) * 0.1
    b2 = np.zeros(1)

    def relu(z):
        return np.maximum(z, 0.0)

    assert global_batch % world_size == 0
    local_batch = global_batch // world_size

    t0 = time.perf_counter()
    samples = 0
    n_batches = n // global_batch
    for ep in range(epochs):
        perm = rng.permutation(n)
        for bi in range(n_batches):
            idx = perm[bi * global_batch:(bi + 1) * global_batch]
            Xb, Yb = X[idx], Y[idx]

            # 累加各 rank 的梯度（模拟 all-reduce 求和），最后求平均
            gW1 = np.zeros_like(W1); gb1 = np.zeros_like(b1)
            gW2 = np.zeros_like(W2); gb2 = np.zeros_like(b2)
            for r in range(world_size):
                xs = Xb[r * local_batch:(r + 1) * local_batch]
                ys = Yb[r * local_batch:(r + 1) * local_batch]
                # 本地前向
                H = relu(xs @ W1 + b1)
                P = H @ W2 + b2
                diff = P - ys
                # 本地反向
                m = xs.shape[0]
                dP = 2.0 * diff / m
                gW2 += H.T @ dP
                gb2 += dP.sum(axis=0)
                dH = dP @ W2.T
                dH[H <= 0] = 0.0
                gW1 += xs.T @ dH
                gb1 += dH.sum(axis=0)
                samples += m
            # all-reduce 求平均
            gW1 /= world_size; gb1 /= world_size
            gW2 /= world_size; gb2 /= world_size
            # 同步更新（各 rank 用同一平均梯度，参数保持一致）
            W1 -= lr * gW1; b1 -= lr * gb1
            W2 -= lr * gW2; b2 -= lr * gb2

    # 最终 loss（全量）
    H = relu(X @ W1 + b1)
    P = H @ W2 + b2
    final_loss = float(np.mean((P - Y) ** 2))
    return final_loss, time.perf_counter() - t0, samples


# ============================================================
#  2-B. torch 版：单进程模拟多 rank DDP（同一逻辑，用 autograd）
# ============================================================
def train_ddp_torch(world_size, X, Y, hidden=64, epochs=8, global_batch=256, lr=5e-3, seed=1):
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    torch.manual_seed(seed)
    dim = X.shape[1]
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)

    model = torch.nn.Sequential(
        torch.nn.Linear(dim, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    ).to(device)
    loss_fn = torch.nn.MSELoss()

    assert global_batch % world_size == 0
    local_batch = global_batch // world_size
    n = X.shape[0]
    n_batches = n // global_batch

    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    samples = 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for bi in range(n_batches):
            idx = perm[bi * global_batch:(bi + 1) * global_batch]
            Xb, Yb = Xt[idx], Yt[idx]
            # 清零梯度
            for p in model.parameters():
                p.grad = None
            # 各 rank 本地反向，梯度累加（.backward 默认累加）——模拟 all-reduce 求和
            for r in range(world_size):
                xs = Xb[r * local_batch:(r + 1) * local_batch]
                ys = Yb[r * local_batch:(r + 1) * local_batch]
                loss = loss_fn(model(xs), ys)
                loss.backward()
                samples += xs.shape[0]
            # all-reduce 求平均 + 手动 SGD 更新
            with torch.no_grad():
                for p in model.parameters():
                    p.grad /= world_size
                    p -= lr * p.grad
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    with torch.no_grad():
        final_loss = float(loss_fn(model(Xt), Yt))
    return final_loss, dt, samples


# ============================================================
#  3. 扩展性报告
# ============================================================
def scalability_report(results):
    """results: list of (world_size, loss, time, samples)。"""
    print()
    print(SEP)
    print("扩展性分析报告（1 / 2 / 4 rank）")
    print(SEP)
    base_tp = None
    print(f"  {'rank数':<8}{'最终loss':>12}{'耗时(s)':>12}{'吞吐(samp/s)':>16}"
          f"{'加速比':>10}{'并行效率':>10}")
    print(f"  {'-'*68}")
    for (ws, loss, t, samples) in results:
        tp = samples / t if t > 0 else 0.0
        if base_tp is None:
            base_tp = tp
        speedup = tp / base_tp if base_tp > 0 else 1.0
        eff = speedup / ws * 100
        print(f"  {ws:<8}{loss:>12.5f}{t:>12.4f}{tp:>16.0f}{speedup:>9.2f}x{eff:>9.1f}%")
    print(f"  {'-'*68}")
    print("  说明：")
    print("   - 各 rank 数下最终 loss 接近，说明 DDP 梯度平均在数学上等价，收敛一致；")
    print("   - 单进程模拟里增加 rank 只是把计算摊在同一 CPU/GPU 上串行做，")
    print("     所以吞吐不会真正随 rank 线性增长——真实多卡才有并行加速；")
    print("   - 真实多卡里，并行效率<100% 源于通信开销与不可并行部分（Amdahl 定律）。")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_ddp()

    X, Y = make_dataset()
    print(SEP)
    backend = "torch" if TORCH_AVAILABLE else "numpy"
    dev = "cuda" if CUDA_AVAILABLE else "cpu"
    print(f"端到端训练：MLP 回归（后端={backend}, 设备={dev}）")
    print(SEP)
    print(f"  数据集: {X.shape[0]} 样本, 维度={X.shape[1]}；全局 batch=256")
    print("  单进程模拟多 rank DDP：切数据 → 本地反向 → 梯度 all-reduce 求平均 → 同步更新\n")

    results = []
    for ws in [1, 2, 4]:
        if TORCH_AVAILABLE:
            loss, t, samples = train_ddp_torch(ws, X, Y)
        else:
            loss, t, samples = train_ddp_numpy(ws, X, Y)
        print(f"  [world_size={ws}] 训练完成：最终 loss={loss:.5f}, 耗时={t:.4f}s")
        results.append((ws, loss, t, samples))

    scalability_report(results)

    print()
    print(SEP)
    if CUDA_AVAILABLE and DIST_AVAILABLE:
        print("环境：torch + CUDA + 分布式就绪。")
        print("      本收官项目用单进程模拟 DDP 保证稳定；真实多卡请用：")
        print("      torchrun --nproc_per_node=N 启动，并用 DistributedDataParallel 包裹模型。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch（CPU）。已用单进程模拟多 rank DDP 完成端到端训练。")
    else:
        print("环境：未安装 PyTorch。已用 numpy 单进程模拟多 rank DDP 完成端到端训练。")
    print("收官要点：DDP = 切数据 + 本地梯度 + all-reduce 求平均 + 同步更新 ≡ 大 batch 单卡。")
    print(SEP)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
