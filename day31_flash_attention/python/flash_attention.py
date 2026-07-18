#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 31: Flash Attention 原理与实现 — Python 版本

演示注意力机制的三种计算方式，帮助理解 Flash Attention 的核心思想：
  1. 标准注意力      : 一次性构造完整 N x N 分数矩阵后做 softmax（显存 O(N^2)）
  2. Tiled + Online  : 分块遍历 K/V，用 online softmax 增量更新（显存 O(N)）
  3. （可选）与官方   : 若 torch 可用，用 F.scaled_dot_product_attention 交叉验证

如果安装了 PyTorch，则用 torch 张量演示；否则自动 fallback 到 NumPy。
"""

import time
import numpy as np

# 尝试导入 PyTorch
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[提示] 未安装 PyTorch，将使用 NumPy 概念演示。")

# ------------------------------------------------------------
# 配置参数（与 CUDA 版本保持一致）
# ------------------------------------------------------------
SEQ_LEN = 1024     # 序列长度 N
HEAD_DIM = 64      # 头维度 d
TILE_SIZE = 32     # 分块大小
TOLERANCE = 1e-4   # 结果比较容差
SEP = "=" * 60     # 60 字符分隔线


# ============================================================
#  NumPy 实现
# ============================================================
def softmax_np(x, axis=-1):
    """数值稳定的 softmax（减去最大值防止 exp 溢出）"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def standard_attention_np(q, k, v):
    """标准注意力：构造完整 N x N 分数矩阵。

    显存开销 O(N^2)，这是长序列时的主要瓶颈。
    """
    d = q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    scores = (q @ k.T) * scale          # [N, N]，完整分数矩阵
    weights = softmax_np(scores, axis=-1)
    return weights @ v                   # [N, d]


def flash_attention_np(q, k, v, tile_size=TILE_SIZE):
    """Tiled + Online Softmax 版本（Flash Attention 概念演示）。

    从不构造完整的 N x N 矩阵，而是分块遍历 K/V，
    用 running max(m) / running sum(l) / 累加器(acc) 增量更新。
    """
    n, d = q.shape
    scale = 1.0 / np.sqrt(d)

    out = np.zeros((n, d), dtype=q.dtype)
    # 每个 Q 行维护 online softmax 的运行状态
    m = np.full((n, 1), -np.inf, dtype=q.dtype)   # running max
    l = np.zeros((n, 1), dtype=q.dtype)           # running sum of exp
    acc = np.zeros((n, d), dtype=q.dtype)         # 加权 V 累加器

    num_tiles = (n + tile_size - 1) // tile_size
    for t in range(num_tiles):
        j0 = t * tile_size
        j1 = min(j0 + tile_size, n)
        k_tile = k[j0:j1]                  # [Bc, d]
        v_tile = v[j0:j1]                  # [Bc, d]

        # 当前 tile 的分数块 S = Q K_tile^T / sqrt(d)
        s = (q @ k_tile.T) * scale         # [N, Bc]

        # online softmax 增量更新
        m_new = np.maximum(m, np.max(s, axis=1, keepdims=True))   # [N, 1]
        p = np.exp(s - m_new)                                     # [N, Bc]
        alpha = np.exp(m - m_new)                                 # 回溯缩放
        l = l * alpha + np.sum(p, axis=1, keepdims=True)
        acc = acc * alpha + p @ v_tile
        m = m_new

    out = acc / l
    return out


# ============================================================
#  PyTorch 实现
# ============================================================
def standard_attention_torch(q, k, v):
    """PyTorch 标准注意力实现"""
    d = q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def flash_attention_torch(q, k, v, tile_size=TILE_SIZE):
    """PyTorch 版 tiled + online softmax（概念演示）"""
    n, d = q.shape
    scale = 1.0 / (d ** 0.5)
    device = q.device

    out = torch.zeros((n, d), dtype=q.dtype, device=device)
    m = torch.full((n, 1), float("-inf"), dtype=q.dtype, device=device)
    l = torch.zeros((n, 1), dtype=q.dtype, device=device)
    acc = torch.zeros((n, d), dtype=q.dtype, device=device)

    num_tiles = (n + tile_size - 1) // tile_size
    for t in range(num_tiles):
        j0 = t * tile_size
        j1 = min(j0 + tile_size, n)
        k_tile = k[j0:j1]
        v_tile = v[j0:j1]

        s = (q @ k_tile.transpose(-2, -1)) * scale
        m_new = torch.maximum(m, s.max(dim=1, keepdim=True).values)
        p = torch.exp(s - m_new)
        alpha = torch.exp(m - m_new)
        l = l * alpha + p.sum(dim=1, keepdim=True)
        acc = acc * alpha + p @ v_tile
        m = m_new

    return acc / l


# ============================================================
#  运行入口
# ============================================================
def run_numpy():
    print(SEP)
    print("  使用 NumPy 演示 Flash Attention 概念")
    print(SEP)
    rng = np.random.default_rng(31)
    q = rng.standard_normal((SEQ_LEN, HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((SEQ_LEN, HEAD_DIM)).astype(np.float32)
    v = rng.standard_normal((SEQ_LEN, HEAD_DIM)).astype(np.float32)

    t0 = time.time()
    o_std = standard_attention_np(q, k, v)
    t1 = time.time()
    o_flash = flash_attention_np(q, k, v)
    t2 = time.time()

    max_diff = float(np.max(np.abs(o_std - o_flash)))
    ok = max_diff < TOLERANCE

    print(f"  标准注意力耗时  : {(t1 - t0) * 1000:8.3f} ms")
    print(f"  Flash 注意力耗时: {(t2 - t1) * 1000:8.3f} ms")
    print(f"  最大绝对误差    : {max_diff:.3e}")
    print(f"  结果一致性      : {'通过' if ok else '失败'}")
    return ok


def run_torch():
    print(SEP)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  使用 PyTorch 演示 Flash Attention 概念 (device={dev})")
    print(SEP)
    torch.manual_seed(31)
    q = torch.randn(SEQ_LEN, HEAD_DIM, device=dev)
    k = torch.randn(SEQ_LEN, HEAD_DIM, device=dev)
    v = torch.randn(SEQ_LEN, HEAD_DIM, device=dev)

    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    o_std = standard_attention_torch(q, k, v)
    if dev == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    o_flash = flash_attention_torch(q, k, v)
    if dev == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()

    diff_flash = float((o_std - o_flash).abs().max())
    ok = diff_flash < 1e-3
    print(f"  标准注意力耗时      : {(t1 - t0) * 1000:8.3f} ms")
    print(f"  Flash(tiled) 耗时   : {(t2 - t1) * 1000:8.3f} ms")
    print(f"  标准 vs Flash 误差  : {diff_flash:.3e}")

    # 与官方 scaled_dot_product_attention（内部使用 Flash/内存高效实现）交叉验证
    try:
        o_sdpa = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
        ).squeeze(0).squeeze(0)
        diff_sdpa = float((o_std - o_sdpa).abs().max())
        print(f"  标准 vs 官方 SDPA   : {diff_sdpa:.3e}")
    except Exception as e:  # noqa: BLE001
        print(f"  [提示] 官方 SDPA 不可用: {e}")

    print(f"  结果一致性          : {'通过' if ok else '失败'}")
    return ok


def main():
    print()
    print(SEP)
    print("     Day 31: Flash Attention 原理与实现 (Python)")
    print(SEP)
    print(f"  序列长度 N : {SEQ_LEN}")
    print(f"  头维度   d : {HEAD_DIM}")
    print(f"  分块 TILE  : {TILE_SIZE}")

    if TORCH_AVAILABLE:
        ok = run_torch()
    else:
        ok = run_numpy()

    print()
    print(SEP)
    print("                        知识点总结")
    print(SEP)
    print("  1. 注意力 = softmax(QK^T / sqrt(d)) V")
    print("  2. 标准实现需完整 N x N 分数矩阵，显存 O(N^2)")
    print("  3. Flash: 分块遍历 K/V，显存降到 O(N)")
    print("  4. Online Softmax: running max/sum 增量更新")
    print("  5. alpha = exp(m_old - m_new) 回溯修正旧累加")
    print(SEP)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
