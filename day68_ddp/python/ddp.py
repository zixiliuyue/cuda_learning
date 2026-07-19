#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 68: 数据并行 DDP（Distributed Data Parallel）

数据并行是最常用的分布式训练范式：
  - 模型在每个 rank 上各存一份完整副本（参数相同）；
  - 把一个大 batch 切成 N 份，每个 rank 各拿一份数据分片；
  - 各自独立前向 + 反向，得到各自的「局部梯度」；
  - 用 all_reduce 对梯度求和再除以 N（即梯度平均）；
  - 所有 rank 用同一份平均梯度更新参数 → 参数始终保持一致。

本天要验证的核心结论（DDP 正确性的数学保证）：
  N 个 rank 各算局部梯度、再 all_reduce 平均  ==  单卡跑同样大 batch 的梯度

运行策略（能不回退就不回退，但绝不卡死）：
  - 有 torch：用真实 autograd 做前向/反向，单进程模拟 N 个 rank（各持数据分片）；
  - 无 torch：用 numpy 手推线性层梯度，同样验证上述等式。
  - 全程单进程，稳定可退出（不使用 mp.spawn，避免卡死）。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量）
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
#  数据并行的关键：梯度平均 == 单卡大 batch 梯度（数学解释）
# ============================================================
def explain_ddp():
    print(SEP)
    print("数据并行 DDP 工作流程")
    print(SEP)
    print("""
    大 batch (B 个样本)
        │  按 rank 切分
        ├──> rank0: 数据分片 0  ─前向─反向─> 局部梯度 g0
        ├──> rank1: 数据分片 1  ─前向─反向─> 局部梯度 g1
        ├──> ...
        └──> rankN: 数据分片 N  ─前向─反向─> 局部梯度 gN
                                          │
                          all_reduce(SUM) / N  （梯度平均）
                                          │
                    所有 rank 得到同一份平均梯度 ḡ
                                          │
                    各自 optimizer.step()（参数保持一致）

  为什么「平均局部梯度 == 单卡大 batch 梯度」？
    单卡 loss = (1/B) Σ_i loss_i   → grad = (1/B) Σ_i g_i
    rank r 只有 B/N 个样本，其 loss = (N/B) Σ_{i∈r} loss_i
        → 局部梯度 g_r = (N/B) Σ_{i∈r} g_i
    平均：(1/N) Σ_r g_r = (1/N) Σ_r (N/B) Σ_{i∈r} g_i = (1/B) Σ_i g_i  ✓
  结论：DDP 在数学上完全等价于单卡大 batch，只是把计算分摊到多卡。
""")


# ============================================================
#  实现 A：PyTorch autograd 真实前向/反向
# ============================================================
def run_torch_demo(world_size=4, batch_per_rank=8, in_dim=6, out_dim=3, seed=0):
    torch.manual_seed(seed)

    # 一份初始参数（所有 rank 共享同样的初值，DDP 靠 broadcast 保证这一点）
    W0 = torch.randn(out_dim, in_dim, dtype=torch.float64)
    b0 = torch.randn(out_dim, dtype=torch.float64)

    total_batch = world_size * batch_per_rank
    X = torch.randn(total_batch, in_dim, dtype=torch.float64)
    Y = torch.randn(total_batch, out_dim, dtype=torch.float64)

    def mse_grad(Xs, Ys, W, b):
        """给定数据分片，返回 (dW, db)。loss 对分片内样本取均值。"""
        W = W.clone().requires_grad_(True)
        b = b.clone().requires_grad_(True)
        pred = Xs @ W.t() + b
        loss = ((pred - Ys) ** 2).mean()
        loss.backward()
        return W.grad.clone(), b.grad.clone()

    # --- 单卡大 batch 基准 ---
    dW_ref, db_ref = mse_grad(X, Y, W0, b0)

    # --- DDP：每个 rank 各算局部梯度 ---
    local_dW, local_db = [], []
    for r in range(world_size):
        xs = X[r * batch_per_rank:(r + 1) * batch_per_rank]
        ys = Y[r * batch_per_rank:(r + 1) * batch_per_rank]
        dW, db = mse_grad(xs, ys, W0, b0)
        local_dW.append(dW)
        local_db.append(db)

    # --- all_reduce(SUM) / N 做梯度平均（此处单进程用 stack+mean 模拟）---
    dW_avg = torch.stack(local_dW).mean(dim=0)
    db_avg = torch.stack(local_db).mean(dim=0)

    err_W = (dW_avg - dW_ref).abs().max().item()
    err_b = (db_avg - db_ref).abs().max().item()
    return err_W, err_b, "PyTorch autograd (float64)"


# ============================================================
#  实现 B：NumPy 手推线性层梯度（无 torch 时）
# ============================================================
def run_numpy_demo(world_size=4, batch_per_rank=8, in_dim=6, out_dim=3, seed=0):
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((out_dim, in_dim))
    b0 = rng.standard_normal(out_dim)

    total_batch = world_size * batch_per_rank
    X = rng.standard_normal((total_batch, in_dim))
    Y = rng.standard_normal((total_batch, out_dim))

    def mse_grad(Xs, Ys, W, b):
        """pred = Xs @ W.T + b; loss = mean((pred-Y)^2)。
        对 W/b 的解析梯度（batch 内取均值）。"""
        n = Xs.shape[0]
        pred = Xs @ W.T + b
        diff = pred - Ys                     # (n, out)
        # d loss/d pred = 2/(n*out) * diff  —— mean 同时对 batch 和 out 维
        dpred = (2.0 / (n * Ys.shape[1])) * diff
        dW = dpred.T @ Xs                    # (out, in)
        db = dpred.sum(axis=0)               # (out,)
        return dW, db

    dW_ref, db_ref = mse_grad(X, Y, W0, b0)

    local_dW, local_db = [], []
    for r in range(world_size):
        xs = X[r * batch_per_rank:(r + 1) * batch_per_rank]
        ys = Y[r * batch_per_rank:(r + 1) * batch_per_rank]
        dW, db = mse_grad(xs, ys, W0, b0)
        local_dW.append(dW)
        local_db.append(db)

    dW_avg = np.mean(np.stack(local_dW), axis=0)
    db_avg = np.mean(np.stack(local_db), axis=0)

    err_W = float(np.abs(dW_avg - dW_ref).max())
    err_b = float(np.abs(db_avg - db_ref).max())
    return err_W, err_b, "NumPy 手推解析梯度"


# ============================================================
#  一个「多步训练」的小演示：DDP 与单卡 loss 曲线应重合
# ============================================================
def run_training_loop_numpy(world_size=4, batch_per_rank=8, in_dim=6,
                            out_dim=3, steps=5, lr=0.1, seed=1):
    rng = np.random.default_rng(seed)
    total_batch = world_size * batch_per_rank
    X = rng.standard_normal((total_batch, in_dim))
    Y = rng.standard_normal((total_batch, out_dim))

    def mse_grad_loss(Xs, Ys, W, b):
        n = Xs.shape[0]
        pred = Xs @ W.T + b
        diff = pred - Ys
        loss = float((diff ** 2).mean())
        dpred = (2.0 / (n * Ys.shape[1])) * diff
        return dpred.T @ Xs, dpred.sum(axis=0), loss

    # 两套完全相同初值的参数：一套单卡、一套 DDP
    W_single = rng.standard_normal((out_dim, in_dim)); b_single = rng.standard_normal(out_dim)
    W_ddp = W_single.copy(); b_ddp = b_single.copy()

    print("  step | 单卡大batch loss | DDP(平均梯度) loss | 参数最大差异")
    print("  -----+------------------+--------------------+-------------")
    for s in range(steps):
        # 单卡
        dW, db, loss_s = mse_grad_loss(X, Y, W_single, b_single)
        W_single -= lr * dW; b_single -= lr * db
        # DDP：各 rank 局部梯度再平均
        gWs, gbs, losses = [], [], []
        for r in range(world_size):
            xs = X[r * batch_per_rank:(r + 1) * batch_per_rank]
            ys = Y[r * batch_per_rank:(r + 1) * batch_per_rank]
            gW, gb, l = mse_grad_loss(xs, ys, W_ddp, b_ddp)
            gWs.append(gW); gbs.append(gb); losses.append(l)
        dW_avg = np.mean(np.stack(gWs), axis=0)
        db_avg = np.mean(np.stack(gbs), axis=0)
        loss_d = float(np.mean(losses))
        W_ddp -= lr * dW_avg; b_ddp -= lr * db_avg
        diff = max(np.abs(W_single - W_ddp).max(), np.abs(b_single - b_ddp).max())
        print(f"   {s:2d}  |    {loss_s:10.6f}    |     {loss_d:10.6f}     |  {diff:.2e}")
    final_diff = max(np.abs(W_single - W_ddp).max(), np.abs(b_single - b_ddp).max())
    return final_diff


# ============================================================
#  主流程
# ============================================================
def main():
    explain_ddp()

    print(SEP)
    print("验证 1：平均局部梯度 == 单卡大 batch 梯度")
    print(SEP)
    world_size, bpr = 4, 8
    if TORCH_AVAILABLE:
        err_W, err_b, tag = run_torch_demo(world_size, bpr)
    else:
        err_W, err_b, tag = run_numpy_demo(world_size, bpr)
    print(f"  引擎：{tag}")
    print(f"  world_size={world_size}, 每 rank batch={bpr}, 总 batch={world_size*bpr}")
    print(f"  dW 最大误差 = {err_W:.2e}")
    print(f"  db 最大误差 = {err_b:.2e}")
    ok = err_W < 1e-9 and err_b < 1e-9
    print(f"  [校验] 平均梯度 == 单卡梯度 ? {'通过 ✓' if ok else '失败 ✗'}\n")

    print(SEP)
    print("验证 2：多步训练下，单卡与 DDP 的参数完全同步（numpy 精确演示）")
    print(SEP)
    final_diff = run_training_loop_numpy(world_size, bpr, steps=5, lr=0.1)
    ok2 = final_diff < 1e-9
    print(f"  [校验] 训练 5 步后参数差异 = {final_diff:.2e} → "
          f"{'完全一致 ✓' if ok2 else '偏差 ✗'}\n")

    print(SEP)
    print("真实 DDP 运行骨架（torch.nn.parallel.DistributedDataParallel，不执行）")
    print(SEP)
    print("""
  # torchrun --nproc_per_node=4 train_ddp.py
  import torch, torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel as DDP
  from torch.utils.data.distributed import DistributedSampler

  dist.init_process_group("nccl")                     # 无 GPU 用 "gloo"
  local_rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local_rank)
  model = MyModel().cuda()
  model = DDP(model, device_ids=[local_rank])         # 自动在 backward 里 all_reduce 梯度
  sampler = DistributedSampler(dataset)               # 保证各 rank 拿不同分片
  loader = DataLoader(dataset, sampler=sampler, batch_size=bs)
  for x, y in loader:
      loss = criterion(model(x.cuda()), y.cuda())
      loss.backward()                                 # DDP 在此自动同步梯度
      optimizer.step(); optimizer.zero_grad()
  dist.destroy_process_group()
""")

    print(SEP)
    if DIST_AVAILABLE and CUDA_AVAILABLE:
        print("环境：torch + CUDA 就绪。可用 torchrun 跑上面的真实 DDP 骨架。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA。已用真实 autograd 单进程模拟多 rank。")
    else:
        print("环境：未安装 PyTorch。已用 numpy 手推梯度单进程模拟多 rank。")
    print("说明：单进程模拟多 rank，梯度平均逻辑与真实 DDP 一致，稳定可退出。")
    print(SEP)


if __name__ == "__main__":
    main()
