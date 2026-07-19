#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 45: 算子融合进阶 — Fused MLP / GELU+Linear

Transformer 的 FFN（前馈网络）里最常见的一步是：

    y = GELU(x @ W + b)

朴素写法（PyTorch 分步）会产生多个中间张量并多次访问显存：
    1) tmp1 = x @ W          # 写回一个 [M, N] 中间张量
    2) tmp2 = tmp1 + b       # 再读 tmp1、写回 tmp2
    3) y    = GELU(tmp2)     # 再读 tmp2、写回 y

本天用 Triton 把「矩阵乘 → 加 bias → GELU」融合进同一个 kernel：
矩阵乘的结果块还在片上寄存器时，直接加 bias 并做 GELU，只写回一次 y，
省掉了 tmp1 / tmp2 两次中间张量的落地（写 + 读各一次显存往返）。

运行策略（能不回退就不回退）：
  - 有 Triton + GPU：运行真实融合 kernel，与 PyTorch 分步计算交叉验证；
  - 无 Triton/GPU：不伪造，用 NumPy/PyTorch-CPU 给出等价的参考结果并讲清融合原理。
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
#  Fused GEMM + Bias + GELU：Triton kernel
# ============================================================
if TRITON_AVAILABLE:

    @triton.jit
    def gelu_tanh(x):
        """GELU 的 tanh 近似（与 PyTorch approximate='tanh' 一致）。

        0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        全部在片上完成，是融合的关键一步。
        """
        c = 0.7978845608028654  # sqrt(2/pi)
        return 0.5 * x * (1.0 + tl.math.tanh(c * (x + 0.044715 * x * x * x)))

    @triton.jit
    def fused_mlp_kernel(
        x_ptr, w_ptr, b_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """计算 y = GELU(x @ w + b)，x:[M,K] w:[K,N] b:[N] y:[M,N]。

        每个 program 负责 y 的一个 [BLOCK_M, BLOCK_N] 输出块：
          - 沿 K 维分块累加得到 x@w 的结果块（还在片上）；
          - 直接加 bias、做 GELU；
          - 只写回一次 y。这就是「一次落地」的融合。
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_mask = offs_k[None, :] < (K - k)
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
            w_mask = (offs_k[:, None] < (K - k)) & (offs_n[None, :] < N)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        # ---- 融合：结果块还在片上，直接加 bias + GELU ----
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + b[None, :]
        acc = gelu_tanh(acc)

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_ptrs, acc, mask=y_mask)

    def triton_fused_mlp(x: "torch.Tensor", w: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        M, K = x.shape
        K2, N = w.shape
        assert K == K2, "x 的列数必须等于 w 的行数"
        y = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        fused_mlp_kernel[grid](
            x, w, b, y,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return y


# ============================================================
#  无 GPU/无 Triton 时的等价参考实现（NumPy，不伪造）
# ============================================================
def numpy_gelu_tanh(x):
    c = 0.7978845608028654  # sqrt(2/pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def numpy_fused_mlp(x, w, b):
    """等价语义：GELU(x @ w + b)。分步与融合数学上一致，差别只在访存次数。"""
    return numpy_gelu_tanh(x @ w + b)


# ============================================================
#  讲解
# ============================================================
def explain_fusion():
    print(SEP)
    print("融合前 vs 融合后：中间张量落地次数对比")
    print(SEP)
    print("""
  操作 y = GELU(x @ W + b)，设 x:[M,K], W:[K,N], y:[M,N]

  【朴素分步（PyTorch）】                    显存往返
    tmp1 = x @ W        -> 写 tmp1[M,N]      1 写
    tmp2 = tmp1 + b     -> 读 tmp1 / 写 tmp2 1 读 + 1 写
    y    = GELU(tmp2)   -> 读 tmp2 / 写 y    1 读 + 1 写
    ----------------------------------------------------
    额外中间张量：tmp1、tmp2 两个 [M,N] 张量
    针对 [M,N] 的显存往返：约 2 次写 + 2 次读（tmp1/tmp2）

  【Triton 融合 kernel】
    x@W 的结果块仍在片上寄存器 -> 直接 +b -> 直接 GELU -> 只写回 y
    ----------------------------------------------------
    额外中间张量：0 个
    针对 [M,N] 的显存往返：只有 1 次写（最终 y）

  收益：
   - 省掉 tmp1 / tmp2 两个 [M,N] 中间张量的分配与显存往返；
   - bias 与 GELU 属于 memory-bound（访存受限），融合后几乎「免费」；
   - 张量越大、越是访存受限，融合收益越明显。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_fusion()

    print(SEP)
    print("演示：Fused GELU(x @ W + b)")
    print(SEP)
    M, K, N = 256, 512, 384
    if TRITON_AVAILABLE:
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        w = torch.randn(K, N, device="cuda", dtype=torch.float32) * 0.02
        b = torch.randn(N, device="cuda", dtype=torch.float32)

        y_triton = triton_fused_mlp(x, w, b)
        # PyTorch 分步参考（等价于融合语义）
        y_ref = torch.nn.functional.gelu(x @ w + b, approximate="tanh")

        max_err = (y_triton - y_ref).abs().max().item()
        print(f"[Triton/GPU] shape=({M},{N})，与 PyTorch 分步计算最大误差 = {max_err:.2e}")
        print("[结果] " + ("通过 ✓" if max_err < 1e-3 else "偏差过大 ✗"))
        print("[融合] 该 kernel 在片上完成 matmul→+bias→GELU，省掉 2 个中间张量落地。")
    else:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((M, K)).astype(np.float32)
        w = (rng.standard_normal((K, N)) * 0.02).astype(np.float32)
        b = rng.standard_normal(N).astype(np.float32)

        # 用 NumPy 分步 vs 融合语义交叉验证（验证数学等价）
        tmp1 = x @ w
        tmp2 = tmp1 + b
        y_step = numpy_gelu_tanh(tmp2)
        y_fused = numpy_fused_mlp(x, w, b)
        max_err = np.abs(y_step - y_fused).max()
        print(f"[无 GPU/Triton] 用 NumPy 演示，分步 vs 融合语义最大误差 = {max_err:.2e}")
        print("[结果] " + ("数学等价 ✓" if max_err < 1e-5 else "不一致 ✗"))
        print("       分步产生 tmp1/tmp2 两个中间张量；Triton 融合版只写回一次 y。")
        print("       Triton 版逻辑见 fused_mlp_kernel（结果块在片上直接 +bias+GELU）。")

    print()
    print(SEP)
    if TRITON_AVAILABLE:
        print("环境：Triton + GPU 就绪，已运行真实融合 kernel。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU / 未装 Triton。已用 NumPy 演示原理。")
        print("提示：在 Linux + NVIDIA GPU 上 `pip install triton` 可运行真实 kernel。")
    else:
        print("环境：未安装 PyTorch。已用 NumPy 演示原理。")
    print(SEP)


if __name__ == "__main__":
    main()
