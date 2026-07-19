#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 44: Triton Flash Attention（tiled + online softmax）

用 Triton 实现 Flash Attention 的前向：
  Attention(Q,K,V) = softmax(Q·K^T / sqrt(d)) · V
且**从不构造完整 N×N 分数矩阵**——沿 K/V 分块遍历，用 online softmax
增量更新 running max(m)、running sum(l) 与输出累加器(acc)。

对照 Day 31 的 CUDA 手写版：
  Day31 里你要手写 __shared__ 缓存 K/V tile、手写 __syncthreads()、
  手写 online softmax 的 m/l 回溯修正、手写 float4 向量化访存……几百行。
  Triton 版把这些交给编译器：tl.load 分块、tl.dot 计算、mask 边界，
  核心逻辑只需几十行，且同样是 O(N) 显存。

核心目的：
  1. 用 Triton 写 tiled + online softmax 的注意力 kernel
  2. 与 PyTorch 标准注意力交叉验证
  3. 对照 Day31 CUDA 版，体会 Triton 的简洁性

运行策略（能不回退就不回退）：
  - 若安装了 Triton + 有可用 GPU：运行真实 Triton kernel，与 PyTorch 交叉验证；
  - 若无 Triton/GPU：不伪造 Triton 结果，打印原理讲解 + 用 NumPy 给出
    「标准注意力」与「online softmax 分块」两条路径的等价参考结果，
    让你在无 GPU 环境也能验证 online softmax 与直接算等价。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（优先真实 GPU，不轻易回退）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TRITON_AVAILABLE = False
if TORCH_AVAILABLE and torch.cuda.is_available():
    try:
        import triton
        import triton.language as tl
        TRITON_AVAILABLE = True
    except ImportError:
        TRITON_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  Triton Flash Attention kernel（tiled + online softmax）
# ============================================================
if TRITON_AVAILABLE:

    @triton.jit
    def flash_attn_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr,
        stride_qm, stride_qd,
        stride_kn, stride_kd,
        stride_vn, stride_vd,
        stride_om, stride_od,
        N, sm_scale,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """每个 program 处理 Q 的一个 BLOCK_M 行块，沿 K/V 分块做 online softmax。

        online softmax 三个运行状态（对照 Day31 CUDA 版的 m/l/O）：
          m_i : running max（每个 Q 行当前见过的最大分数）
          l_i : running sum（exp 归一化分母的累计）
          acc : 输出累加器（尚未除以 l_i 的加权 V 和）
        遍历每个 K/V tile 时：
          1. 算本块分数 S = Q·K^T * scale（只是一个小块，绝不存整行）
          2. m_new = max(m_i, rowmax(S))
          3. alpha = exp(m_i - m_new) 用来回溯修正旧的 l 与 acc
          4. p = exp(S - m_new)；l = l*alpha + rowsum(p)；acc = acc*alpha + p·V
        """
        start_m = tl.program_id(0)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        # 载入本 program 负责的 Q 行块（常驻片上）
        q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < N
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # 初始化 online softmax 运行状态
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        # 沿 K/V 的 N 维分块遍历
        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            n_mask = offs_n < N
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

            # ---- 本块分数 S = Q · K^T * scale（[BLOCK_M, BLOCK_N]，不存整行）----
            s = tl.dot(q, tl.trans(k)) * sm_scale
            s = tl.where(n_mask[None, :], s, -float("inf"))  # 屏蔽越界 key

            # ---- online softmax 增量更新 ----
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)                       # 旧状态回溯修正系数
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # ---- 收尾：除以归一化分母 l ----
        acc = acc / l_i[:, None]
        o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=q_mask)

    def triton_flash_attention(q, k, v):
        """单头注意力（q,k,v 形状均为 [N, d]）。"""
        N, d = q.shape
        o = torch.empty_like(q)
        sm_scale = 1.0 / (d ** 0.5)
        BLOCK_M, BLOCK_N = 64, 64
        BLOCK_D = triton.next_power_of_2(d)
        grid = (triton.cdiv(N, BLOCK_M),)
        flash_attn_kernel[grid](
            q, k, v, o,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            o.stride(0), o.stride(1),
            N, sm_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        return o


# ============================================================
#  无 GPU/无 Triton 时的原理演示（不伪造 Triton，只讲清逻辑）
# ============================================================
def numpy_standard_attention(q, k, v):
    """标准注意力：显式构造 N×N 分数矩阵（O(N²) 显存）。"""
    d = q.shape[-1]
    scores = q @ k.T / np.sqrt(d)                 # [N, N]
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=-1, keepdims=True)
    return p @ v


def numpy_flash_attention(q, k, v, block_n=64):
    """用 NumPy 模拟 tiled + online softmax（O(N) 显存，从不建整张分数矩阵）。

    与 Triton kernel 逻辑一一对应，用于在无 GPU 时验证与标准注意力等价。
    """
    N, d = q.shape
    scale = 1.0 / np.sqrt(d)
    o = np.zeros((N, d), dtype=np.float32)
    m = np.full((N,), -np.inf, dtype=np.float32)   # running max
    l = np.zeros((N,), dtype=np.float32)           # running sum
    for n0 in range(0, N, block_n):
        n1 = min(n0 + block_n, N)
        s = q @ k[n0:n1].T * scale                 # [N, block]，只是一小块
        m_new = np.maximum(m, s.max(axis=1))
        alpha = np.exp(m - m_new)                  # 回溯修正
        p = np.exp(s - m_new[:, None])
        l = l * alpha + p.sum(axis=1)
        o = o * alpha[:, None] + p @ v[n0:n1]
        m = m_new
    return o / l[:, None]


def explain_flash():
    print(SEP)
    print("Flash Attention：tiled + online softmax（对照 Day31 CUDA 手写版）")
    print(SEP)
    print("""
  标准注意力（O(N²) 显存）：
    Q[N,d] × K^T ─▶ S[N,N] ─▶ softmax ─▶ ×V ─▶ O[N,d]
                    ^^^^^^ 完整分数矩阵，N 越大平方增长，显存瓶颈

  Flash Attention（O(N) 显存）：对每个 Q 行维护 m / l / acc 三个运行状态，
  沿 K/V 分块遍历，从不构造整张 S：
    for 每个 K/V tile:
        S_blk = Q · K_tile^T * scale         # 只算一小块
        m_new = max(m, rowmax(S_blk))
        alpha = exp(m - m_new)               # 回溯修正旧状态
        p     = exp(S_blk - m_new)
        l     = l*alpha + rowsum(p)
        acc   = acc*alpha + p · V_tile
    O = acc / l

  Day31 CUDA 手写版 vs Triton 版
  ------------------------------------------------------------
  项目            CUDA(Day31)                  Triton(本天)
  K/V 分块缓存    手写 __shared__ + 加载        tl.load（编译器管片上）
  同步            手写 __syncthreads()          自动
  m/l 回溯修正    手写标量循环维护              向量化，几行搞定
  向量化访存      手写 float4                   编译器自动向量化
  边界            手写 if                       mask / tl.where
  代码量          数百行                        约几十行

  结论：online softmax 的数学不变，但 Triton 把访存/同步/向量化的“脏活”
  交给编译器，让你专注算法本身——这正是它成为业界主流的原因。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_flash()

    print(SEP)
    print("演示：单头注意力 Attention(Q,K,V)")
    print(SEP)
    N, d = 256, 64
    if TRITON_AVAILABLE:
        q = torch.randn(N, d, device="cuda", dtype=torch.float16)
        k = torch.randn(N, d, device="cuda", dtype=torch.float16)
        v = torch.randn(N, d, device="cuda", dtype=torch.float16)
        o_triton = triton_flash_attention(q, k, v)
        # PyTorch 标准注意力参考（fp32 计算更准）
        qf, kf, vf = q.float(), k.float(), v.float()
        scores = qf @ kf.T / (d ** 0.5)
        p = torch.softmax(scores, dim=-1)
        o_ref = p @ vf
        max_err = (o_triton.float() - o_ref).abs().max().item()
        print(f"[Triton/GPU] N={N}, d={d}, fp16, 与标准注意力最大误差 = {max_err:.3e}")
        print("[结果] " + ("通过 ✓" if max_err < 1e-2 else "偏差过大 ✗"))
    else:
        rng = np.random.default_rng(0)
        q = rng.standard_normal((N, d)).astype(np.float32)
        k = rng.standard_normal((N, d)).astype(np.float32)
        v = rng.standard_normal((N, d)).astype(np.float32)
        o_std = numpy_standard_attention(q, k, v)
        o_flash = numpy_flash_attention(q, k, v, block_n=64)
        max_err = np.abs(o_std - o_flash).max()
        print(f"[无 GPU/Triton] 用 NumPy 对照，N={N}, d={d}")
        print(f"       online softmax 分块结果 与 标准注意力最大误差 = {max_err:.3e}（应≈0）")
        print("       => 证明 tiled + online softmax 与直接算数学等价，且只需 O(N) 显存")
        print("       Triton 版逻辑见 flash_attn_kernel（分块遍历 K/V，片上维护 m/l/acc）")

    print()
    print(SEP)
    if TRITON_AVAILABLE:
        print("环境：Triton + GPU 就绪，已运行真实 Triton Flash Attention。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU / 未装 Triton。已用 NumPy 演示原理。")
        print("提示：在 Linux + NVIDIA GPU 上 `pip install triton` 可运行真实 kernel。")
    else:
        print("环境：未安装 PyTorch。已用 NumPy 演示原理。")
    print(SEP)


if __name__ == "__main__":
    main()
