#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 70: 张量并行（Tensor Parallel, Megatron-LM 风格）

当单层的权重矩阵大到单卡放不下时，用张量并行把「一个算子的计算」切到多卡：
每卡只算一部分，再通过集合通信拼出完整结果。核心是两种 Linear：

  Column Parallel Linear（按【输出维/列】切权重）
    Y = X @ W^T,  把 W 按行(即输出维)切成 [W0; W1]
      rank0: Y0 = X @ W0^T   （得到 Y 的前半列）
      rank1: Y1 = X @ W1^T   （得到 Y 的后半列）
    前向：all_gather([Y0, Y1]) 拼成完整 Y（按列拼接）

  Row Parallel Linear（按【输入维/行】切权重）
    Y = X @ W^T,  把 X 按列切、W 按列(输入维)切
      rank0: Y0 = X0 @ W0^T   （部分和）
      rank1: Y1 = X1 @ W1^T   （部分和）
    前向：all_reduce(Y0 + Y1) 得到完整 Y（求和）

  Megatron 经典组合：MLP = Column Parallel → 激活 → Row Parallel，
  这样两层之间无需通信，整个 MLP 只需 1 次 all_reduce。

本天用【单进程模拟 2 个 rank】实现上述逻辑，验证：
  张量并行结果 == 单卡完整 Linear 结果。
有 torch 用 torch，无 torch 自动回退 numpy，稳定可退出（不用 mp.spawn）。
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
#  用 numpy 统一实现（torch 张量也可 .numpy() 后走同一逻辑，保证稳定）
#  约定：Linear 计算 Y = X @ W.T  (+ b)，W 形状 (out_dim, in_dim)
# ============================================================
def linear_full(X, W, b=None):
    """单卡完整 Linear，作为参考基准。"""
    Y = X @ W.T
    if b is not None:
        Y = Y + b
    return Y


def column_parallel(X, W, b, world_size=2):
    """Column Parallel：W 按【输出维 out_dim】切成 world_size 份。

    每个 rank 算出 Y 的一段「列」，前向末尾 all_gather 按列拼接。
    返回 (完整Y, 各rank本地Y分片列表)。
    """
    out_dim = W.shape[0]
    W_shards = np.array_split(W, world_size, axis=0)         # 按行(=输出维)切
    b_shards = np.array_split(b, world_size) if b is not None else [None] * world_size

    local_Ys = []
    for r in range(world_size):
        yr = X @ W_shards[r].T
        if b_shards[r] is not None:
            yr = yr + b_shards[r]
        local_Ys.append(yr)                                 # 每 rank 得到 Y 的一段列

    # 前向通信：all_gather 沿列(特征)维拼接
    Y = np.concatenate(local_Ys, axis=1)
    assert Y.shape[1] == out_dim
    return Y, local_Ys


def row_parallel(X, W, b, world_size=2):
    """Row Parallel：W 按【输入维 in_dim】切，X 也按列切。

    每个 rank 算出一个「部分和」，前向末尾 all_reduce(SUM) 相加。
    偏置只在聚合后加一次（避免加 world_size 次）。
    返回 (完整Y, 各rank部分和列表)。
    """
    X_shards = np.array_split(X, world_size, axis=1)         # X 按输入维切
    W_shards = np.array_split(W, world_size, axis=1)         # W 按输入维(列)切

    partials = []
    for r in range(world_size):
        partials.append(X_shards[r] @ W_shards[r].T)         # 部分和，形状 (n, out_dim)

    # 前向通信：all_reduce(SUM)
    Y = np.zeros_like(partials[0])
    for p in partials:
        Y = Y + p
    if b is not None:
        Y = Y + b                                            # 偏置聚合后加一次
    return Y, partials


def megatron_mlp(X, W1, b1, W2, b2, world_size=2):
    """Megatron 经典 MLP：Column Parallel -> GELU -> Row Parallel。

    第一层 Column Parallel 后，每 rank 手里正好是隐藏维的一段，
    直接在本地做激活，再喂给 Row Parallel 的对应分片，
    两层之间【无需通信】，整个 MLP 仅末尾 1 次 all_reduce。
    """
    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    # --- 第一层：Column Parallel，得到各 rank 的隐藏分片（不 all_gather）---
    W1_shards = np.array_split(W1, world_size, axis=0)       # 按隐藏维切
    b1_shards = np.array_split(b1, world_size)
    # --- 第二层：Row Parallel，W2 按输入维(=隐藏维)切，与第一层分片对齐 ---
    W2_shards = np.array_split(W2, world_size, axis=1)

    partials = []
    for r in range(world_size):
        h_r = gelu(X @ W1_shards[r].T + b1_shards[r])        # 本地隐藏分片 + 激活
        y_r = h_r @ W2_shards[r].T                           # Row Parallel 部分和
        partials.append(y_r)

    Y = np.zeros_like(partials[0])
    for p in partials:
        Y = Y + p
    Y = Y + b2                                               # 偏置聚合后加一次
    return Y


def mlp_full(X, W1, b1, W2, b2):
    """单卡完整 MLP 参考。"""
    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
    h = gelu(X @ W1.T + b1)
    return h @ W2.T + b2


def explain_tp():
    print(SEP)
    print("张量并行核心：把「一个算子」切到多卡")
    print(SEP)
    print("""
  Column Parallel Linear（按输出维/列切 W）
    W = [W0]   rank0: Y0 = X @ W0^T  (Y 的前半列)
        [W1]   rank1: Y1 = X @ W1^T  (Y 的后半列)
    前向通信：all_gather([Y0,Y1]) 按列拼 → 完整 Y
        X ─┬─> [rank0 W0] ─> Y0 ┐
           └─> [rank1 W1] ─> Y1 ┴─ all_gather ─> Y

  Row Parallel Linear（按输入维/行切 W，X 也切）
    X=[X0|X1], W=[W0|W1]
      rank0: Y0 = X0 @ W0^T (部分和)
      rank1: Y1 = X1 @ W1^T (部分和)
    前向通信：all_reduce(Y0+Y1) → 完整 Y
        X0 ─> [rank0 W0] ─> Y0 ┐
        X1 ─> [rank1 W1] ─> Y1 ┴─ all_reduce(SUM) ─> Y

  Megatron MLP = Column Parallel ─GELU─> Row Parallel
    Column 输出的隐藏分片，直接喂给 Row 的对应分片，
    两层间【无需通信】，整个 MLP 仅 1 次 all_reduce。这是 Megatron 的精髓。
""")


def to_np(x):
    """torch 张量或 numpy 统一转 numpy（保证走同一稳定逻辑）。"""
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def main():
    explain_tp()

    # 构造数据（有 torch 用 torch 生成再转 numpy，体现"优先真实框架"）
    n, in_dim, out_dim, hidden = 5, 8, 6, 12
    if TORCH_AVAILABLE:
        torch.manual_seed(0)
        X = to_np(torch.randn(n, in_dim, dtype=torch.float64))
        W = to_np(torch.randn(out_dim, in_dim, dtype=torch.float64))
        b = to_np(torch.randn(out_dim, dtype=torch.float64))
        W1 = to_np(torch.randn(hidden, in_dim, dtype=torch.float64))
        b1 = to_np(torch.randn(hidden, dtype=torch.float64))
        W2 = to_np(torch.randn(out_dim, hidden, dtype=torch.float64))
        b2 = to_np(torch.randn(out_dim, dtype=torch.float64))
        engine = "PyTorch 生成数据 + numpy 稳定计算 (float64)"
    else:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, in_dim))
        W = rng.standard_normal((out_dim, in_dim))
        b = rng.standard_normal(out_dim)
        W1 = rng.standard_normal((hidden, in_dim))
        b1 = rng.standard_normal(hidden)
        W2 = rng.standard_normal((out_dim, hidden))
        b2 = rng.standard_normal(out_dim)
        engine = "NumPy (float64)"

    print(SEP)
    print(f"验证 1：Column Parallel Linear == 单卡完整 Linear（引擎：{engine}）")
    print(SEP)
    Y_ref = linear_full(X, W, b)
    Y_cp, shards = column_parallel(X, W, b, world_size=2)
    err = float(np.abs(Y_cp - Y_ref).max())
    print(f"  W 形状 {W.shape} 按输出维切成 2 份: "
          f"{[s.shape for s in np.array_split(W, 2, axis=0)]}")
    print(f"  rank0 本地 Y 分片形状 {shards[0].shape}, rank1 {shards[1].shape}")
    print(f"  all_gather 拼接后 Y 形状 {Y_cp.shape}, 参考 {Y_ref.shape}")
    print(f"  最大误差 = {err:.2e}  → {'通过 ✓' if err < 1e-9 else '失败 ✗'}\n")

    print(SEP)
    print("验证 2：Row Parallel Linear == 单卡完整 Linear")
    print(SEP)
    Y_rp, partials = row_parallel(X, W, b, world_size=2)
    err2 = float(np.abs(Y_rp - Y_ref).max())
    print(f"  W 形状 {W.shape} 按输入维切成 2 份: "
          f"{[s.shape for s in np.array_split(W, 2, axis=1)]}")
    print(f"  X 形状 {X.shape} 按输入维切成 2 份")
    print(f"  各 rank 部分和形状 {partials[0].shape}，all_reduce(SUM) 后得完整 Y")
    print(f"  最大误差 = {err2:.2e}  → {'通过 ✓' if err2 < 1e-9 else '失败 ✗'}\n")

    print(SEP)
    print("验证 3：Megatron MLP（Column→GELU→Row）== 单卡完整 MLP")
    print(SEP)
    Y_mlp_ref = mlp_full(X, W1, b1, W2, b2)
    Y_mlp_tp = megatron_mlp(X, W1, b1, W2, b2, world_size=2)
    err3 = float(np.abs(Y_mlp_tp - Y_mlp_ref).max())
    print(f"  第一层 W1 {W1.shape} 按隐藏维切, 第二层 W2 {W2.shape} 按隐藏维切（对齐）")
    print(f"  两层之间无通信，整个 MLP 仅末尾 1 次 all_reduce")
    print(f"  最大误差 = {err3:.2e}  → {'通过 ✓' if err3 < 1e-9 else '失败 ✗'}\n")

    all_ok = err < 1e-9 and err2 < 1e-9 and err3 < 1e-9
    print(SEP)
    print(f"[总校验] 张量并行结果 == 单卡完整结果 ? {'全部通过 ✓' if all_ok else '存在失败 ✗'}")
    print(SEP)

    print(SEP)
    print("真实张量并行运行骨架（Megatron/torch.distributed，不执行）")
    print(SEP)
    print("""
  # torchrun --nproc_per_node=2 tp.py
  import torch, torch.distributed as dist
  dist.init_process_group("nccl")                    # 无 GPU 用 "gloo"
  rank, world = dist.get_rank(), dist.get_world_size()

  # Column Parallel：本地只存 W 的 1/world 行（输出维分片）
  local_out = out_dim // world
  W_local = torch.randn(local_out, in_dim, device="cuda")
  Y_local = X @ W_local.t()                           # 各算一段列
  Y_list = [torch.empty_like(Y_local) for _ in range(world)]
  dist.all_gather(Y_list, Y_local)                    # 拼成完整 Y
  Y = torch.cat(Y_list, dim=1)

  # Row Parallel：本地只存 W 的 1/world 列（输入维分片）
  Y_partial = X_local @ W_local.t()
  dist.all_reduce(Y_partial, op=dist.ReduceOp.SUM)    # 部分和求和
  dist.destroy_process_group()
""")

    print(SEP)
    if DIST_AVAILABLE and CUDA_AVAILABLE:
        print("环境：torch + CUDA 就绪。可用 torchrun 跑上面的真实张量并行骨架。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA。已用单进程模拟 2 个 rank 验证正确性。")
    else:
        print("环境：未安装 PyTorch。已用 numpy 单进程模拟 2 个 rank 验证正确性。")
    print("说明：单进程模拟多 rank，切分/通信逻辑与真实张量并行一致，稳定可退出。")
    print(SEP)


if __name__ == "__main__":
    main()
