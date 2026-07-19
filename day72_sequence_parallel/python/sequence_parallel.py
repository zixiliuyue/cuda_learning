#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 72: 序列并行 / 上下文并行（Sequence / Context Parallel）

当序列长度 S 极长（如 128K token）时，单卡放不下这么长序列的激活与
attention 中间量。序列并行的思路：把序列「按 seq 维度」切成若干块，
每个 rank 只持有 S/N 个 token 的 Q/K/V，从而分摊显存。

难点在 attention：每个 query 需要看到「全部」key/value，而 K/V 被切散在
各 rank 上。两种主流做法：
  1. AllGather KV：先把所有 rank 的 K/V 收齐，再本地算完整 attention；
  2. Ring Attention：K/V 块在 rank 之间「环形」传递，边传边累加，显存更省。

本天目标：验证「序列并行注意力」的结果 == 「完整序列注意力」。

运行策略（单机单卡也能学）：
  用「单进程模拟多 rank」——把完整序列切成 N 段，用列表存各 rank 的分片，
  显式模拟 AllGather 与 Ring 两种通信，最后对拍完整 attention。
  有 torch 用 torch，无则 numpy。
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

GPU_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()
MULTI_GPU = GPU_AVAILABLE and torch.cuda.device_count() > 1

SEP = "=" * 64


# ------------------------------------------------------------
# 后端无关的小工具（torch / numpy 统一接口）
# ------------------------------------------------------------
def softmax_lastdim(x):
    if TORCH_AVAILABLE:
        return torch.softmax(x, dim=-1)
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def matmul(a, b):
    return a @ b


def transpose_last2(x):
    if TORCH_AVAILABLE:
        return x.transpose(-1, -2)
    return np.swapaxes(x, -1, -2)


def concat_seq(chunks):
    """沿 seq 维（axis=0，形状 [S, D]）拼接各 rank 的分片。"""
    if TORCH_AVAILABLE:
        return torch.cat(chunks, dim=0)
    return np.concatenate(chunks, axis=0)


def make_qkv(seq_len, dim, seed):
    if TORCH_AVAILABLE:
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(seq_len, dim, generator=g)
        k = torch.randn(seq_len, dim, generator=g)
        v = torch.randn(seq_len, dim, generator=g)
        return q, k, v
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((seq_len, dim)).astype(np.float32)
    k = rng.standard_normal((seq_len, dim)).astype(np.float32)
    v = rng.standard_normal((seq_len, dim)).astype(np.float32)
    return q, k, v


# ============================================================
#  0. 原理讲解
# ============================================================
def explain_sequence_parallel():
    print(SEP)
    print("序列并行核心思想：沿「序列长度 S」维度切分")
    print(SEP)
    print("""
  完整序列 S=8，切到 4 个 rank，每 rank 持有 2 个 token 的 Q/K/V：

    rank0        rank1        rank2        rank3
   tok0,1       tok2,3       tok4,5       tok6,7
   Q/K/V        Q/K/V        Q/K/V        Q/K/V

  问题：attention 里每个 query 要和「全部 token」的 key/value 交互，
        但 K/V 被切散了。解决方案：

    (A) AllGather KV：各 rank 把自己的 K/V 广播收齐 → 本地拥有完整 K/V
        → 直接算本地 query 对全序列的 attention。实现简单，通信量大。

    (B) Ring Attention：K/V 分块在 rank 间「环形」流动，每收到一块就用
        online-softmax 增量累加。峰值显存最省，是超长上下文的主流方案。

  与「数据并行」不同：DP 切 batch，序列并行切的是「单条样本的 seq 维」。
""")


# ============================================================
#  1. 参考实现：完整序列 attention（单卡直接算）
# ============================================================
def full_attention(q, k, v):
    """标准 scaled dot-product attention（作为对拍参考）。"""
    dim = q.shape[-1]
    scale = 1.0 / (dim ** 0.5)
    scores = matmul(q, transpose_last2(k)) * scale     # [S, S]
    attn = softmax_lastdim(scores)                     # 每行归一化
    return matmul(attn, v)                             # [S, D]


# ============================================================
#  2. 序列并行 —— 方案 A：AllGather KV
# ============================================================
def sequence_parallel_allgather(q, k, v, num_ranks):
    """模拟：先把序列切成 num_ranks 份，再 AllGather 收齐 K/V，各 rank 本地算。

    单进程模拟：用列表存每个 rank 的 Q 分片；K/V 用「拼接」模拟 AllGather。
    最后把各 rank 的输出按 seq 维拼回，应等于完整 attention。
    """
    S = q.shape[0]
    assert S % num_ranks == 0, "序列长度需能被 rank 数整除"
    chunk = S // num_ranks

    # 切分：每个 rank 只持有自己的 Q/K/V 分片
    q_shards = [q[r * chunk:(r + 1) * chunk] for r in range(num_ranks)]
    k_shards = [k[r * chunk:(r + 1) * chunk] for r in range(num_ranks)]
    v_shards = [v[r * chunk:(r + 1) * chunk] for r in range(num_ranks)]

    # AllGather：所有 rank 收齐完整 K/V（这里用拼接模拟通信结果）
    k_full = concat_seq(k_shards)
    v_full = concat_seq(v_shards)

    # 每个 rank 用「本地 Q 分片」对「完整 K/V」算 attention
    out_shards = []
    for r in range(num_ranks):
        out_r = full_attention(q_shards[r], k_full, v_full)
        out_shards.append(out_r)

    # 各 rank 输出按 seq 维拼回完整输出
    return concat_seq(out_shards)


# ============================================================
#  3. 序列并行 —— 方案 B：Ring Attention（online softmax 增量累加）
# ============================================================
def _exp(x):
    return torch.exp(x) if TORCH_AVAILABLE else np.exp(x)


def _max_lastdim(x):
    if TORCH_AVAILABLE:
        return torch.max(x, dim=-1, keepdim=True).values
    return np.max(x, axis=-1, keepdims=True)


def _sum_lastdim(x):
    if TORCH_AVAILABLE:
        return torch.sum(x, dim=-1, keepdim=True)
    return np.sum(x, axis=-1, keepdims=True)


def _zeros_like_out(rows, dim):
    if TORCH_AVAILABLE:
        return torch.zeros(rows, dim), torch.full((rows, 1), -1e30), torch.zeros(rows, 1)
    return (np.zeros((rows, dim), np.float32),
            np.full((rows, 1), -1e30, np.float32),
            np.zeros((rows, 1), np.float32))


def sequence_parallel_ring(q, k, v, num_ranks):
    """模拟 Ring Attention：K/V 块环形流经每个持有 Q 分片的 rank。

    每个 rank 对自己的 Q 分片，依次「见到」所有 K/V 块（模拟环形传递），
    用 online-softmax 增量融合，无需一次性存下完整 attention 矩阵。
    这正是 Flash Attention 的分块思想在序列维度上的分布式版本。
    """
    S = q.shape[0]
    chunk = S // num_ranks
    dim = q.shape[-1]
    scale = 1.0 / (dim ** 0.5)

    q_shards = [q[r * chunk:(r + 1) * chunk] for r in range(num_ranks)]
    k_shards = [k[r * chunk:(r + 1) * chunk] for r in range(num_ranks)]
    v_shards = [v[r * chunk:(r + 1) * chunk] for r in range(num_ranks)]

    out_shards = []
    for r in range(num_ranks):
        qr = q_shards[r]
        acc, m_run, l_run = _zeros_like_out(chunk, dim)   # 累加输出、运行最大值、运行归一化和
        # 环形：依次接收每个 rank 的 K/V 块（顺序不影响结果）
        for step in range(num_ranks):
            src = (r + step) % num_ranks                  # 模拟从环上第 src 个邻居收到块
            kj, vj = k_shards[src], v_shards[src]
            scores = matmul(qr, transpose_last2(kj)) * scale   # [chunk, chunk]
            m_blk = _max_lastdim(scores)
            m_new = m_run if TORCH_AVAILABLE else m_run
            m_new = (torch.maximum(m_run, m_blk) if TORCH_AVAILABLE
                     else np.maximum(m_run, m_blk))
            # 用新的全局最大值重新缩放旧累加量
            alpha = _exp(m_run - m_new)
            p = _exp(scores - m_new)
            l_run = l_run * alpha + _sum_lastdim(p)
            acc = acc * alpha + matmul(p, vj)
            m_run = m_new
        out_shards.append(acc / l_run)

    return concat_seq(out_shards)


# ============================================================
#  主流程
# ============================================================
def main():
    explain_sequence_parallel()

    seq_len, dim, num_ranks = 8, 16, 4
    q, k, v = make_qkv(seq_len, dim, seed=42)

    print(SEP)
    print(f"配置：序列长度 S={seq_len}，维度 D={dim}，切分到 {num_ranks} 个 rank")
    print(SEP)
    print(f"  每个 rank 持有 {seq_len // num_ranks} 个 token 的 Q/K/V 分片")

    ref = full_attention(q, k, v)

    print()
    print(SEP)
    print("方案 A：AllGather KV —— 收齐完整 K/V 后本地算 attention")
    print(SEP)
    out_ag = sequence_parallel_allgather(q, k, v, num_ranks)
    if TORCH_AVAILABLE:
        err_ag = (out_ag - ref).abs().max().item()
    else:
        err_ag = float(np.max(np.abs(out_ag - ref)))
    print(f"  与完整序列 attention 最大误差 = {err_ag:.2e}")
    print("  [结果] " + ("通过 ✓" if err_ag < 1e-4 else "偏差过大 ✗"))

    print()
    print(SEP)
    print("方案 B：Ring Attention —— K/V 块环形流动 + online-softmax 增量累加")
    print(SEP)
    out_ring = sequence_parallel_ring(q, k, v, num_ranks)
    if TORCH_AVAILABLE:
        err_ring = (out_ring - ref).abs().max().item()
    else:
        err_ring = float(np.max(np.abs(out_ring - ref)))
    print(f"  与完整序列 attention 最大误差 = {err_ring:.2e}")
    print("  [结果] " + ("通过 ✓" if err_ring < 1e-4 else "偏差过大 ✗"))

    print()
    print(SEP)
    print("通信量对比（切成 N 份，每份 K/V 大小记为 1 单位）")
    print(SEP)
    print("  AllGather KV：每 rank 需收齐 (N-1) 份 K/V，通信量 O(N) 且需存完整 K/V")
    print("  Ring Attention：每步只传 1 份 K/V，峰值显存 O(1)，适合超长上下文")

    print()
    print(SEP)
    if MULTI_GPU:
        print("环境：多 GPU。真实序列并行可用 Megatron / Ring-Attention 实现；"
              "本脚本单进程模拟以保证可复现。")
    elif GPU_AVAILABLE:
        print("环境：单 GPU。序列并行需多卡真实通信，本脚本用单进程模拟 AllGather / Ring。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 无 GPU。已用 torch(CPU) 模拟序列并行并对拍完整 attention。")
    else:
        print("环境：无 PyTorch。已用 NumPy 模拟序列并行并对拍完整 attention。")
    print(SEP)


if __name__ == "__main__":
    main()
