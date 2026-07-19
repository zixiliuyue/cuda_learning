#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 42: Triton 矩阵乘与自动调优（@triton.autotune）

用 Triton 实现分块矩阵乘（GEMM），对照前 40 天里：
  - Day 6 的手写 Tiled 矩阵乘（共享内存分块）
  - Day 13 的 cuBLAS 调用

核心目的：
  1. 用 Triton 写出分块 GEMM kernel（program 处理 C 的一个 BLOCK_M×BLOCK_N 块）
  2. 掌握 @triton.autotune —— 让编译器自动在多组 BLOCK 配置里搜出最快的
  3. 与 torch.matmul 交叉验证正确性

运行策略（能不回退就不回退）：
  - 若安装了 Triton + 有可用 GPU：运行真实 Triton kernel + autotune，
    并与 torch.matmul 交叉验证；
  - 若无 Triton/GPU：不伪造 Triton 结果，而是打印分块 GEMM 原理讲解，
    并用 NumPy/PyTorch-CPU 给出等价参考结果，让你在无 GPU 环境也能看懂逻辑。
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
#  Triton 分块矩阵乘 kernel（含 @triton.autotune）
# ============================================================
if TRITON_AVAILABLE:

    # autotune：给定一批候选配置，Triton 会用真实输入各跑一次，
    # 缓存最快的那组，之后同样 shape 直接复用。key 指定“哪些参数变化时重新搜索”。
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """C = A @ B，每个 program 计算 C 的一个 BLOCK_M×BLOCK_N 子块。

        对照 Day 6 手写 Tiled CUDA：
          CUDA 里你手动开 __shared__ 缓存 A/B 的 tile、手动 __syncthreads()、
          手动累加；Triton 里只需沿 K 维循环，tl.load 一块块读入，tl.dot
          做块内乘加，编译器自动帮你管流水线（num_stages）与片上缓存。
        """
        pid = tl.program_id(axis=0)
        # ---- L2 友好的分组重排（group-major）：让相邻 program 复用同一批 A/B 行列 ----
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ---- 计算本 program 负责的行/列偏移 ----
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # ---- 沿 K 维分块累加（片上累加器，float32 精度）----
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k[None, :] < K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=k_mask, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            accumulator += tl.dot(a, b)          # 块内矩阵乘加，硬件 Tensor Core 可用
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = accumulator.to(tl.float16)
        # ---- 写回 C 子块，用 mask 处理边界 ----
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def triton_matmul(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        assert a.shape[1] == b.shape[0], "K 维不匹配"
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        # grid 只与 M/N 有关，BLOCK 由 autotune 选中的 meta 决定
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
        return c


# ============================================================
#  无 GPU/无 Triton 时的原理演示（不伪造 Triton，只讲清逻辑）
# ============================================================
def numpy_tiled_matmul(a, b, block_k=32):
    """用 NumPy 演示“沿 K 维分块累加”的分块 GEMM 思想（结果等价于 a @ b）。"""
    M, K = a.shape
    _, N = b.shape
    c = np.zeros((M, N), dtype=np.float32)
    for k0 in range(0, K, block_k):
        k1 = min(k0 + block_k, K)
        c += a[:, k0:k1] @ b[k0:k1, :]   # 一块块累加，模拟片上 accumulator
    return c


def explain_matmul():
    print(SEP)
    print("Triton 分块矩阵乘 vs CUDA Tiled（核心对照）")
    print(SEP)
    print("""
  C[M,N] = A[M,K] @ B[K,N]，按块切分：

      N 维 →                     每个 program 负责 C 的一个
    ┌───┬───┬───┐                BLOCK_M × BLOCK_N 子块，
  M │C00│C01│C02│                沿 K 维循环，一次读入 A/B 的
  维 ├───┼───┼───┤                一个 tile 做 tl.dot 累加。
  ↓ │C10│C11│C12│
    └───┴───┴───┘

  步骤            CUDA Tiled(Day6)             Triton
  ------------------------------------------------------------
  子块划分        blockIdx.{x,y}               tl.program_id + group 重排
  片上缓存        手写 __shared__ A/B tile     tl.load 到寄存器/片上，编译器管
  同步            __syncthreads()              自动（num_stages 软件流水线）
  块内乘加        手写三重循环累加             tl.dot（可走 Tensor Core）
  边界            if 判断                      mask
  配置搜索        手动试 TILE 大小             @triton.autotune 自动搜

  关键点：@triton.autotune 给一批候选 Config（不同 BLOCK_M/N/K、num_warps、
  num_stages），编译器用真实输入各跑一次，缓存最快的，无需你手动调参。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_matmul()

    print(SEP)
    print("演示：分块矩阵乘 C = A @ B")
    print(SEP)
    M, N, K = 512, 512, 512
    if TRITON_AVAILABLE:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        c_triton = triton_matmul(a, b)
        c_ref = torch.matmul(a, b)
        max_err = (c_triton.float() - c_ref.float()).abs().max().item()
        print(f"[Triton/GPU] shape=({M},{N},{K}) fp16, 与 torch.matmul 最大误差 = {max_err:.3e}")
        # fp16 GEMM 误差主要来自累加与舍入，阈值放宽
        print("[结果] " + ("通过 ✓" if max_err < 1.0 else "偏差过大 ✗"))
        best = matmul_kernel.best_config
        print(f"[autotune] 本 shape 选中的最优配置：{best}")
    else:
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        c_tiled = numpy_tiled_matmul(a, b, block_k=64)
        c_ref = a @ b
        max_err = np.abs(c_tiled - c_ref).max()
        print(f"[无 GPU/Triton] 用 NumPy 演示分块累加 GEMM，shape=({M},{N},{K})")
        print(f"       分块结果 与 直接 a@b 最大误差 = {max_err:.3e}（应≈0，仅浮点误差）")
        print("       Triton 版逻辑见 matmul_kernel（每个 program 算一个 C 子块，K 维分块 tl.dot 累加）")
        print("       autotune 逻辑见 @triton.autotune（多组 BLOCK 配置自动择优）")

    print()
    print(SEP)
    if TRITON_AVAILABLE:
        print("环境：Triton + GPU 就绪，已运行真实 Triton GEMM + autotune。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU / 未装 Triton。已用 NumPy 演示原理。")
        print("提示：在 Linux + NVIDIA GPU 上 `pip install triton` 可运行真实 kernel。")
    else:
        print("环境：未安装 PyTorch。已用 NumPy 演示原理。")
    print(SEP)


if __name__ == "__main__":
    main()
