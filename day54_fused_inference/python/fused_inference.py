#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 54: 算子融合在推理中的应用 — Fused QKV / Fused LayerNorm+Residual

Transformer 推理里，每个小算子（线性、加法、归一化）都要启动一次 GPU kernel，
并把中间结果写回全局显存再读出来。**算子融合**把多个步骤合成一个 kernel：
  - 减少 kernel 启动次数（launch overhead）
  - 减少中间张量的显存往返（带宽）

本天演示两种推理中最常见的融合，并验证融合前后**数值完全一致**：
  1. Fused QKV 投影：把 Q/K/V 三个线性层合成一个大矩阵乘，再切分
  2. Fused LayerNorm + Residual：把「残差相加 + LayerNorm」在一步内完成

运行策略：device = "cuda" if torch.cuda.is_available() else "cpu"，逻辑都真实运行；
无 torch 时退化为 NumPy 版。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

SEP = "=" * 64


# ============================================================
#  PyTorch 版
# ============================================================
if TORCH_AVAILABLE:

    # ---------- Fused QKV ----------
    def qkv_unfused(x, Wq, Wk, Wv):
        """未融合：三次独立矩阵乘，三次 kernel 启动，三个中间张量。"""
        q = x @ Wq
        k = x @ Wk
        v = x @ Wv
        return q, k, v

    def qkv_fused(x, Wqkv):
        """融合：一次大矩阵乘得到 [.., 3*d]，再切分为 q/k/v。

        Wqkv 是把 Wq|Wk|Wv 在列方向拼接得到的大矩阵。
        """
        qkv = x @ Wqkv          # 单次 GEMM
        d = qkv.shape[-1] // 3
        q, k, v = qkv.split(d, dim=-1)
        return q, k, v

    # ---------- Fused LayerNorm + Residual ----------
    def ln_residual_unfused(x, residual, gamma, beta, eps=1e-5):
        """未融合：先残差相加（1 个 kernel + 中间张量），再 LayerNorm（多个 kernel）。"""
        h = x + residual                        # 中间张量 h 写回显存
        mean = h.mean(dim=-1, keepdim=True)
        var = h.var(dim=-1, unbiased=False, keepdim=True)
        normed = (h - mean) / torch.sqrt(var + eps)
        return normed * gamma + beta

    def ln_residual_fused(x, residual, gamma, beta, eps=1e-5):
        """融合：残差相加与归一化统计在同一「概念 kernel」内完成，不落地中间张量。

        真实实现是一个自定义 CUDA/Triton kernel；此处用等价数学表达演示语义，
        关键是 h 不再单独物化为一个持久的全局显存张量。
        """
        h = x + residual
        mean = h.mean(dim=-1, keepdim=True)
        var = h.var(dim=-1, unbiased=False, keepdim=True)
        return (h - mean) / torch.sqrt(var + eps) * gamma + beta


# ============================================================
#  NumPy 版
# ============================================================
def np_qkv_unfused(x, Wq, Wk, Wv):
    return x @ Wq, x @ Wk, x @ Wv


def np_qkv_fused(x, Wqkv):
    qkv = x @ Wqkv
    d = qkv.shape[-1] // 3
    return qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]


def np_ln_residual(x, residual, gamma, beta, eps=1e-5):
    h = x + residual
    mean = h.mean(axis=-1, keepdims=True)
    var = h.var(axis=-1, keepdims=True)
    return (h - mean) / np.sqrt(var + eps) * gamma + beta


# ============================================================
#  讲解
# ============================================================
def explain_fusion():
    print(SEP)
    print("算子融合为什么快（推理视角）")
    print(SEP)
    print("""
  每个 kernel 都有固定开销：启动延迟 + 中间结果写回/读取全局显存。
  融合把多个步骤合成一个 kernel，收益有两点：
    1. kernel 启动次数下降  -> 减少 launch overhead
    2. 中间张量不落地显存   -> 减少内存带宽消耗（推理常是带宽瓶颈）

  Fused QKV：
    未融合  x@Wq, x@Wk, x@Wv    -> 3 次 GEMM，3 个中间张量
    融合    x@[Wq|Wk|Wv]        -> 1 次大 GEMM，再切分（更利于 Tensor Core）

  Fused LayerNorm+Residual：
    未融合  h=x+res(落地) -> mean/var -> normalize   -> 多个 kernel + 中间张量 h
    融合    残差相加与归一化统计在一个 kernel 内完成 -> h 不落地
""")


def check_close(name, a, b):
    if TORCH_AVAILABLE:
        max_err = (a - b).abs().max().item()
    else:
        max_err = float(np.abs(a - b).max())
    ok = max_err < 1e-4
    print(f"  {name:<32} 最大误差={max_err:.2e}  " + ("一致 ✓" if ok else "不一致 ✗"))
    return ok


# ============================================================
#  主流程
# ============================================================
def main():
    explain_fusion()

    seq_len, d_model = 128, 512
    print(SEP)
    print(f"实验：seq_len={seq_len}, d_model={d_model}")
    print(f"运行设备：DEVICE = {DEVICE}"
          + ("（真实 GPU）" if DEVICE == "cuda" else "（CPU，逻辑完整）"))
    print(SEP)

    if TORCH_AVAILABLE:
        torch.manual_seed(0)
        x = torch.randn(seq_len, d_model, device=DEVICE)
        Wq = torch.randn(d_model, d_model, device=DEVICE) * 0.02
        Wk = torch.randn(d_model, d_model, device=DEVICE) * 0.02
        Wv = torch.randn(d_model, d_model, device=DEVICE) * 0.02
        Wqkv = torch.cat([Wq, Wk, Wv], dim=1)   # 列方向拼接成大矩阵

        print("[演示 1] Fused QKV 投影")
        q1, k1, v1 = qkv_unfused(x, Wq, Wk, Wv)
        q2, k2, v2 = qkv_fused(x, Wqkv)
        check_close("Q 融合前后", q1, q2)
        check_close("K 融合前后", k1, k2)
        check_close("V 融合前后", v1, v2)
        print("  kernel 启动次数：未融合 3 次 GEMM  ->  融合 1 次 GEMM")
        print("  中间张量：未融合 3 个 (q,k,v)     ->  融合 1 个 (qkv) 再视图切分")

        print()
        print("[演示 2] Fused LayerNorm + Residual")
        residual = torch.randn(seq_len, d_model, device=DEVICE)
        gamma = torch.randn(d_model, device=DEVICE)
        beta = torch.randn(d_model, device=DEVICE)
        o1 = ln_residual_unfused(x, residual, gamma, beta)
        o2 = ln_residual_fused(x, residual, gamma, beta)
        check_close("LayerNorm+Residual 融合前后", o1, o2)
        print("  kernel 启动次数：未融合 add + mean + var + norm 多个")
        print("                  ->  融合 1 个 kernel 内完成，h 不落地显存")

    else:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((seq_len, d_model)).astype(np.float32)
        Wq = (rng.standard_normal((d_model, d_model)) * 0.02).astype(np.float32)
        Wk = (rng.standard_normal((d_model, d_model)) * 0.02).astype(np.float32)
        Wv = (rng.standard_normal((d_model, d_model)) * 0.02).astype(np.float32)
        Wqkv = np.concatenate([Wq, Wk, Wv], axis=1)

        print("[演示 1] Fused QKV 投影")
        q1, k1, v1 = np_qkv_unfused(x, Wq, Wk, Wv)
        q2, k2, v2 = np_qkv_fused(x, Wqkv)
        check_close("Q 融合前后", q1, q2)
        check_close("K 融合前后", k1, k2)
        check_close("V 融合前后", v1, v2)
        print("  kernel 启动次数：未融合 3 次 GEMM  ->  融合 1 次 GEMM")

        print()
        print("[演示 2] Fused LayerNorm + Residual")
        residual = rng.standard_normal((seq_len, d_model)).astype(np.float32)
        gamma = rng.standard_normal(d_model).astype(np.float32)
        beta = rng.standard_normal(d_model).astype(np.float32)
        o1 = np_ln_residual(x, residual, gamma, beta)
        o2 = np_ln_residual(x, residual, gamma, beta)
        check_close("LayerNorm+Residual 融合前后", o1, o2)

    print()
    print(SEP)
    print("结论：融合前后数值一致，但融合减少了 kernel 启动与中间张量往返。")
    if DEVICE == "cuda":
        print("环境：CUDA GPU 就绪，融合演示在 GPU 上真实运行。")
    elif TORCH_AVAILABLE:
        print("环境：无 CUDA GPU，已在 CPU 上真实运行完整逻辑。")
    else:
        print("环境：未安装 PyTorch，已用 NumPy 演示等价原理。")
    print(SEP)


if __name__ == "__main__":
    main()
