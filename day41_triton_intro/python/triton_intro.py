#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 41: Triton 入门 — 编程模型 vs CUDA

用 Triton 重写前 40 天里最经典的两个 kernel：
  1. 向量加法（对照 Day 2 的 CUDA 版）
  2. Softmax（对照 Day 31 / common 里的 CUDA 版）

核心目的：理解 Triton 的编程模型（program / BLOCK / mask）与 CUDA
（thread / block / if 边界检查）的对应关系。

运行策略（能不回退就不回退）：
  - 若安装了 Triton + 有可用 GPU：运行真实 Triton kernel，并与 PyTorch 结果交叉验证；
  - 若无 Triton/GPU：不强制回退到假实现，而是打印原理讲解 + 用 PyTorch/NumPy
    给出等价的“参考结果”，让你在无 GPU 环境也能看懂逻辑。
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
#  1. 向量加法：Triton kernel
# ============================================================
if TRITON_AVAILABLE:

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """Triton 向量加法 kernel。

        对照 CUDA：
          CUDA 里每个 thread 处理一个元素，用 blockIdx*blockDim+threadIdx 定位；
          Triton 里每个 "program" 处理一整个 BLOCK_SIZE 的连续块，用
          program_id + arange 生成一批 offset，再用 mask 处理越界。
        """
        pid = tl.program_id(axis=0)                     # 相当于 CUDA 的 blockIdx.x
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 一次处理一批下标
        mask = offsets < n_elements                       # 相当于 CUDA 的 if(tid < n)
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    def triton_add(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        out = torch.empty_like(x)
        n_elements = out.numel()
        # grid：需要多少个 program 才能覆盖所有元素（向上取整）
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
        return out


# ============================================================
#  2. Softmax：Triton kernel（每个 program 处理一行）
# ============================================================
if TRITON_AVAILABLE:

    @triton.jit
    def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                       n_cols, BLOCK_SIZE: tl.constexpr):
        """行 softmax：每个 program 负责矩阵的一行。

        体现“融合”思想：一次把整行读入片上，减最大值、exp、求和、相除
        全部在寄存器/片上完成，只写回一次结果。
        """
        row_idx = tl.program_id(axis=0)
        row_start = in_ptr + row_idx * in_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(row_start + col_offsets, mask=mask, other=-float("inf"))
        row = row - tl.max(row, axis=0)          # 数值稳定：减最大值
        numerator = tl.exp(row)
        denominator = tl.sum(numerator, axis=0)
        result = numerator / denominator
        out_start = out_ptr + row_idx * out_row_stride
        tl.store(out_start + col_offsets, result, mask=mask)

    def triton_softmax(x: "torch.Tensor") -> "torch.Tensor":
        n_rows, n_cols = x.shape
        block_size = triton.next_power_of_2(n_cols)
        out = torch.empty_like(x)
        softmax_kernel[(n_rows,)](
            out, x, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=block_size,
        )
        return out


# ============================================================
#  无 GPU/无 Triton 时的原理演示（不伪造 Triton，只讲清逻辑）
# ============================================================
def numpy_add(x, y):
    return x + y


def numpy_softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def explain_programming_model():
    print(SEP)
    print("Triton 编程模型 vs CUDA（核心对照）")
    print(SEP)
    print("""
  概念            CUDA                         Triton
  ------------------------------------------------------------
  并行单位        thread（单个元素）           program（一个数据块）
  块内索引        threadIdx.x                  tl.arange(0, BLOCK)
  块编号          blockIdx.x                   tl.program_id(0)
  边界处理        if (tid < n)                 mask = offsets < n
  访存            指针 + 手动索引              tl.load / tl.store(+mask)
  共享内存        显式 __shared__ + sync       编译器自动管理片上缓存
  网格配置        <<<grid, block>>>            kernel[grid](..., BLOCK=...)

  关键差异：
   - Triton 以“块”为思考单位，你不用再管单个线程，也基本不用手写
     共享内存和 __syncthreads()，编译器帮你做向量化与调度。
   - 因此融合算子（Fused Softmax/LayerNorm/Attention）在 Triton 里
     比 CUDA C++ 简洁得多，这正是它成为业界主流的原因。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_programming_model()

    print(SEP)
    print("演示 1：向量加法")
    print(SEP)
    N = 1 << 20  # 1,048,576 个元素
    if TRITON_AVAILABLE:
        x = torch.rand(N, device="cuda")
        y = torch.rand(N, device="cuda")
        out_triton = triton_add(x, y)
        out_ref = x + y
        max_err = (out_triton - out_ref).abs().max().item()
        print(f"[Triton/GPU] N={N}, 与 PyTorch 最大误差 = {max_err:.2e}")
        print("[结果] " + ("通过 ✓" if max_err < 1e-5 else "偏差过大 ✗"))
    else:
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        out = numpy_add(x, y)
        print(f"[无 GPU/Triton] 用 NumPy 演示等价逻辑，N={N}，结果[:3]={out[:3]}")
        print("       Triton 版逻辑见 add_kernel（每个 program 处理 1024 个元素）")

    print()
    print(SEP)
    print("演示 2：行 Softmax")
    print(SEP)
    rows, cols = 1024, 512
    if TRITON_AVAILABLE:
        x = torch.randn(rows, cols, device="cuda")
        out_triton = triton_softmax(x)
        out_ref = torch.softmax(x, axis=-1)
        max_err = (out_triton - out_ref).abs().max().item()
        print(f"[Triton/GPU] shape=({rows},{cols}), 与 torch.softmax 最大误差 = {max_err:.2e}")
        print("[结果] " + ("通过 ✓" if max_err < 1e-4 else "偏差过大 ✗"))
    else:
        x = np.random.randn(rows, cols).astype(np.float32)
        out = numpy_softmax(x)
        row_sums = out.sum(axis=-1)
        print(f"[无 GPU/Triton] 用 NumPy 演示，每行和应≈1，实测均值={row_sums.mean():.6f}")
        print("       Triton 版逻辑见 softmax_kernel（每个 program 处理一整行，片上融合）")

    print()
    print(SEP)
    if TRITON_AVAILABLE:
        print("环境：Triton + GPU 就绪，已运行真实 Triton kernel。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU / 未装 Triton。已用 NumPy 演示原理。")
        print("提示：在 Linux + NVIDIA GPU 上 `pip install triton` 可运行真实 kernel。")
    else:
        print("环境：未安装 PyTorch。已用 NumPy 演示原理。")
    print(SEP)


if __name__ == "__main__":
    main()
