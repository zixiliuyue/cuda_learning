#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 47: 阶段小结 — Triton vs CUDA vs cuBLAS

第九阶段（Day41-47）用 Triton 重写了向量加法、Softmax、GEMM、
Fused Softmax/LayerNorm、Flash Attention、Fused MLP，并把算子接入了
PyTorch 自动微分。本天做一次系统对比与总结：

对同一个操作（矩阵乘 与 softmax）分别用
  - torch (底层调 cuBLAS / cuDNN)
  - Triton（若可用）
测量耗时，打印对比表；并总结三种实现方式（手写 CUDA / cuBLAS / Triton）
在开发效率与性能上的取舍。

运行策略（能不回退就不回退）：
  - 有 Triton + GPU：对 torch 与 Triton 分别计时，交叉验证正确性并打印对比表；
  - 只有 torch-CPU：用 torch-CPU 对 matmul / softmax 计时，说明这不代表 GPU 性能；
  - 无 torch：用 NumPy 计时演示，重点在方法论与总结表。
"""

import time
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
#  Triton kernel：GEMM 与 行 Softmax（仅在可用时定义）
# ============================================================
if TRITON_AVAILABLE:

    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            km = offs_k[None, :] < (K - k)
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & km, other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def triton_matmul(a, b):
        M, K = a.shape
        K2, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        BM, BN, BK = 64, 64, 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        matmul_kernel[grid](a, b, c, M, N, K,
                            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                            c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
        return c

    @triton.jit
    def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        e = tl.exp(x)
        s = tl.sum(e, axis=0)
        tl.store(out_ptr + row * out_stride + cols, e / s, mask=mask)

    def triton_softmax(x):
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        bs = triton.next_power_of_2(n_cols)
        softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=bs)
        return out


# ============================================================
#  计时工具
# ============================================================
def bench_gpu(fn, warmup=10, iters=50):
    """GPU 计时：用 cuda.Event，含 warmup 与同步。返回毫秒。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_cpu(fn, warmup=2, iters=10):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


# ============================================================
#  讲解与阶段回顾
# ============================================================
def explain_three_ways():
    print(SEP)
    print("三种算子实现方式：开发效率 vs 性能 vs 灵活性")
    print(SEP)
    print("""
  方式          代表           开发效率   性能        融合能力    适用场景
  --------------------------------------------------------------------------
  手写 CUDA     .cu kernel     低         极高(可调)  强(全手动)  极致优化/特殊算子
  厂商库        cuBLAS/cuDNN   高(调 API) 极高        无(黑盒)    标准算子(GEMM/卷积)
  Triton        @triton.jit    中高       高(近手写)  强(易写)    自定义融合算子

  经验法则：
   - 标准 GEMM / 卷积 → 直接用 cuBLAS/cuDNN，几乎没人能手写超过它；
   - 需要融合的自定义算子（Fused LayerNorm / Flash Attention）→ Triton 首选，
     开发成本远低于 CUDA C++，性能可达手写的 80%~100%+；
   - 追求极致或库/Triton 都覆盖不了的特殊模式 → 手写 CUDA。
""")


def review_stage9():
    print(SEP)
    print("第九阶段（Day41-47）Triton 要点回顾")
    print(SEP)
    print("""
  Day 41  编程模型：program/BLOCK/mask ↔ CUDA thread/block/if 边界
  Day 42  分块 GEMM + @triton.autotune 自动搜索 block 配置
  Day 43  Fused Softmax / LayerNorm：一次读入、片上完成多步、只写回一次
  Day 44  Flash Attention：tiling + online softmax，比 CUDA 手写简洁得多
  Day 45  Fused MLP：GEMM→+bias→GELU 融合，省掉中间张量落地
  Day 46  接入 PyTorch：autograd.Function forward/backward + 梯度校验 + nn.Module
  Day 47  小结：Triton vs CUDA vs cuBLAS 的取舍（本天）

  一句话总结：
   Triton 让你用 Python 写「以块为单位」的 kernel，编译器负责向量化、
   共享内存与调度；它在「自定义融合算子」这个 CUDA 最痛、cuBLAS 覆盖不到
   的地带，取得了开发效率与性能的最佳平衡，是当今大模型算子的主流写法。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_three_ways()

    print(SEP)
    print("性能对比：矩阵乘 (GEMM) 与 行 Softmax")
    print(SEP)

    M = K = N = 1024
    rows, cols = 4096, 1024

    if TRITON_AVAILABLE:
        torch.manual_seed(0)
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

        # 正确性交叉验证
        mm_err = (triton_matmul(a, b) - (a @ b)).abs().max().item()
        sm_err = (triton_softmax(x) - torch.softmax(x, dim=-1)).abs().max().item()

        t_torch_mm = bench_gpu(lambda: a @ b)                 # cuBLAS
        t_triton_mm = bench_gpu(lambda: triton_matmul(a, b))  # Triton
        t_torch_sm = bench_gpu(lambda: torch.softmax(x, dim=-1))
        t_triton_sm = bench_gpu(lambda: triton_softmax(x))

        print(f"[正确性] GEMM 误差={mm_err:.2e}  Softmax 误差={sm_err:.2e}")
        print()
        print(f"  {'操作':<16}{'torch(cuBLAS/cuDNN)':<22}{'Triton':<14}{'加速比':<10}")
        print("  " + "-" * 58)
        print(f"  {'GEMM 1024³':<16}{t_torch_mm:>8.3f} ms{'':<10}{t_triton_mm:>8.3f} ms{'':<2}{t_torch_mm/t_triton_mm:>6.2f}x")
        print(f"  {'Softmax 4096x1024':<16}{t_torch_sm:>8.3f} ms{'':<8}{t_triton_sm:>8.3f} ms{'':<2}{t_torch_sm/t_triton_sm:>6.2f}x")
        print()
        print("  注：加速比>1 表示 Triton 更快；GEMM 上 cuBLAS 通常很强，")
        print("      Softmax 这类访存受限融合算子 Triton 往往更有优势。")

    elif TORCH_AVAILABLE:
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        x = torch.randn(rows, cols)
        t_mm = bench_cpu(lambda: a @ b)
        t_sm = bench_cpu(lambda: torch.softmax(x, dim=-1))
        print("[torch-CPU] 无 CUDA GPU / 未装 Triton，用 CPU 计时（不代表 GPU 性能）：")
        print(f"  GEMM 1024³        : {t_mm:.3f} ms")
        print(f"  Softmax 4096x1024 : {t_sm:.3f} ms")
        print("  在 GPU 上，torch 会调用 cuBLAS/cuDNN，Triton 可与之对比（见上方对照表）。")

    else:
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        x = np.random.randn(rows, cols).astype(np.float32)

        def np_softmax(z):
            e = np.exp(z - z.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)

        t_mm = bench_cpu(lambda: a @ b)
        t_sm = bench_cpu(lambda: np_softmax(x))
        print("[NumPy-CPU] 未安装 PyTorch，用 NumPy 计时（仅演示方法论）：")
        print(f"  GEMM 1024³        : {t_mm:.3f} ms")
        print(f"  Softmax 4096x1024 : {t_sm:.3f} ms")
        print("  真实对比需在 GPU 上进行；重点看下方三种方式的取舍与阶段回顾。")

    print()
    review_stage9()

    print(SEP)
    if TRITON_AVAILABLE:
        print("环境：Triton + GPU 就绪，已完成 torch(cuBLAS) vs Triton 真实对比。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU / 未装 Triton。已用 CPU 计时演示方法论。")
        print("提示：在 Linux + NVIDIA GPU 上 `pip install triton` 可得到真实 GPU 对比。")
    else:
        print("环境：未安装 PyTorch。已用 NumPy 计时演示方法论。")
    print(SEP)


if __name__ == "__main__":
    main()
