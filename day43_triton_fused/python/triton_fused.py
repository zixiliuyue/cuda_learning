#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 43: Triton 融合算子 — Fused Softmax / LayerNorm

延续 Day41 的“融合”思想，本天把 LayerNorm 写成一个 fused kernel：
  一次把整行读入片上，完成 mean → var → normalize → scale/shift 全流程，
  只写回一次，避免 PyTorch eager 模式里多次 kernel launch 与全局访存往返。

对照：
  - Day 41 的 Fused Softmax（每个 program 处理一行）
  - common/ 里的 CUDA 版 Softmax / 归约

核心目的：
  1. 用 Triton 写 fused layernorm kernel（每个 program 负责一行）
  2. 与 torch.nn.functional.layer_norm 交叉验证
  3. 顺带给出 fused softmax，对比二者的“片上多步融合”共性

运行策略（能不回退就不回退）：
  - 若安装了 Triton + 有可用 GPU：运行真实 Triton kernel，与 PyTorch 交叉验证；
  - 若无 Triton/GPU：不伪造 Triton 结果，打印融合原理讲解 + 用 NumPy/PyTorch-CPU
    给出等价参考结果，让你在无 GPU 环境也能看懂逻辑。
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
#  Triton Fused LayerNorm kernel（每个 program 处理一行）
# ============================================================
if TRITON_AVAILABLE:

    @triton.jit
    def layernorm_kernel(
        x_ptr, y_ptr, w_ptr, b_ptr,
        x_row_stride, y_row_stride,
        n_cols, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused LayerNorm：一次读入整行，片上完成全部计算，只写回一次。

        y = (x - mean) / sqrt(var + eps) * weight + bias

        融合体现：mean、var、归一化、仿射变换全部在片上寄存器完成，
        整行只从全局内存读 1 次、写 1 次。对照 PyTorch eager：
        它可能拆成多个 kernel（求均值、求方差、归一化…），访存往返更多。
        """
        row_idx = tl.program_id(axis=0)
        x_start = x_ptr + row_idx * x_row_stride
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        # 一次读入整行（越界补 0，但不参与统计——用 mask 修正计数）
        x = tl.load(x_start + cols, mask=mask, other=0.0).to(tl.float32)

        # ---- 片上求均值 ----
        mean = tl.sum(x, axis=0) / n_cols
        # ---- 片上求方差（只对有效列）----
        x_centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(x_centered * x_centered, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)

        # ---- 归一化 + 仿射变换（weight/bias）----
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x_centered * rstd * w + b

        # ---- 只写回一次 ----
        y_start = y_ptr + row_idx * y_row_stride
        tl.store(y_start + cols, y, mask=mask)

    def triton_layernorm(x, weight, bias, eps=1e-5):
        assert x.is_contiguous()
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        block_size = triton.next_power_of_2(n_cols)
        layernorm_kernel[(n_rows,)](
            x, y, weight, bias,
            x.stride(0), y.stride(0),
            n_cols, eps,
            BLOCK_SIZE=block_size,
        )
        return y

    @triton.jit
    def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                       n_cols, BLOCK_SIZE: tl.constexpr):
        """Fused Softmax：每个 program 一行，片上减最大值→exp→求和→相除。"""
        row_idx = tl.program_id(axis=0)
        row_start = in_ptr + row_idx * in_row_stride
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        row = tl.load(row_start + cols, mask=mask, other=-float("inf"))
        row = row - tl.max(row, axis=0)
        numerator = tl.exp(row)
        denominator = tl.sum(numerator, axis=0)
        result = numerator / denominator
        out_start = out_ptr + row_idx * out_row_stride
        tl.store(out_start + cols, result, mask=mask)

    def triton_softmax(x):
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
def numpy_layernorm(x, weight, bias, eps=1e-5):
    """NumPy 版 LayerNorm，等价于 fused kernel 的逐行计算。"""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)          # 有偏方差，与 torch/kernel 一致
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * weight + bias


def numpy_softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def explain_fusion():
    print(SEP)
    print("融合算子思想：LayerNorm 为例")
    print(SEP)
    print("""
  未融合（PyTorch eager 的典型多 kernel 路径）：
    x ─▶[kernel1 求mean]─▶[kernel2 求var]─▶[kernel3 归一化]─▶[kernel4 仿射]─▶ y
        每一步都要把整行/中间结果在全局内存里来回读写，访存往返多。

  融合（Triton fused layernorm，一个 kernel 一行）：
    x ─▶┌───────────────────────────────────────┐─▶ y
        │ 读入整行到片上                          │
        │  mean → var → (x-mean)*rstd*w + b       │  全在寄存器/片上完成
        │ 只写回一次                              │
        └───────────────────────────────────────┘

  对照表
  ------------------------------------------------------------
  项目            未融合(多kernel)         Triton Fused
  全局读          多次                     整行 1 次
  全局写          多次(中间结果落地)       结果 1 次
  kernel launch   多次                     1 次
  片上复用        无(每步重新读)           mean/var/归一化共享同一份片上数据

  公式（LayerNorm）：
    mean = Σx / N
    var  = Σ(x-mean)^2 / N
    y    = (x-mean)/sqrt(var+eps) * weight + bias
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_fusion()

    # -------- 演示 1：Fused LayerNorm --------
    print(SEP)
    print("演示 1：Fused LayerNorm")
    print(SEP)
    rows, cols = 1024, 768
    eps = 1e-5
    if TRITON_AVAILABLE:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        weight = torch.randn(cols, device="cuda", dtype=torch.float32)
        bias = torch.randn(cols, device="cuda", dtype=torch.float32)
        y_triton = triton_layernorm(x, weight, bias, eps)
        y_ref = torch.nn.functional.layer_norm(x, (cols,), weight, bias, eps)
        max_err = (y_triton - y_ref).abs().max().item()
        print(f"[Triton/GPU] shape=({rows},{cols}), 与 F.layer_norm 最大误差 = {max_err:.2e}")
        print("[结果] " + ("通过 ✓" if max_err < 1e-3 else "偏差过大 ✗"))
    else:
        x = np.random.randn(rows, cols).astype(np.float32)
        weight = np.random.randn(cols).astype(np.float32)
        bias = np.random.randn(cols).astype(np.float32)
        y = numpy_layernorm(x, weight, bias, eps)
        # 归一化部分（去掉 affine）每行均值应≈0、方差应≈1
        x_norm = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)
        print(f"[无 GPU/Triton] 用 NumPy 演示 LayerNorm，shape=({rows},{cols})")
        print(f"       归一化后每行均值≈{x_norm.mean(-1).mean():.2e}，方差≈{x_norm.var(-1).mean():.4f}")
        print("       Triton 版逻辑见 layernorm_kernel（每个 program 一行，片上融合 mean/var/归一化/仿射）")

    # -------- 演示 2：Fused Softmax --------
    print()
    print(SEP)
    print("演示 2：Fused Softmax")
    print(SEP)
    rows2, cols2 = 1024, 512
    if TRITON_AVAILABLE:
        x = torch.randn(rows2, cols2, device="cuda")
        y_triton = triton_softmax(x)
        y_ref = torch.softmax(x, axis=-1)
        max_err = (y_triton - y_ref).abs().max().item()
        print(f"[Triton/GPU] shape=({rows2},{cols2}), 与 torch.softmax 最大误差 = {max_err:.2e}")
        print("[结果] " + ("通过 ✓" if max_err < 1e-4 else "偏差过大 ✗"))
    else:
        x = np.random.randn(rows2, cols2).astype(np.float32)
        y = numpy_softmax(x)
        print(f"[无 GPU/Triton] 用 NumPy 演示，每行和应≈1，实测均值={y.sum(-1).mean():.6f}")
        print("       Triton 版逻辑见 softmax_kernel（片上减最大值/exp/求和/相除融合）")

    print()
    print(SEP)
    if TRITON_AVAILABLE:
        print("环境：Triton + GPU 就绪，已运行真实 Triton fused kernel。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU / 未装 Triton。已用 NumPy 演示原理。")
        print("提示：在 Linux + NVIDIA GPU 上 `pip install triton` 可运行真实 kernel。")
    else:
        print("环境：未安装 PyTorch。已用 NumPy 演示原理。")
    print(SEP)


if __name__ == "__main__":
    main()
