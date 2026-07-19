#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 53: 推理量化 — INT8/FP16 权重量化

LLM 权重通常是 FP32/FP16，参数量巨大，显存与带宽是推理瓶颈。**权重量化**把
浮点权重压成低比特整数（如 INT8），推理时再反量化回浮点参与计算，从而：
  - 显存/存储节省 4x（FP32→INT8）或 2x（FP16→INT8）
  - 内存带宽降低，访存更快

本天手写 **INT8 对称量化**，对比两种粒度并验证精度：
  1. per-tensor  : 整个权重张量共用一个 scale（简单，但异常值会拉低精度）
  2. per-channel : 每个输出通道（每行）独立 scale（更精细，误差更小）

同时演示 FP16 转换的显存节省。

运行策略：device = "cuda" if torch.cuda.is_available() else "cpu"，逻辑都真实运行；
完全没有 torch 时退化为 NumPy 版演示原理。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（有 CUDA 用 GPU，无 CUDA 用 CPU）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

SEP = "=" * 64
QMAX = 127  # INT8 对称量化范围 [-127, 127]


# ============================================================
#  PyTorch 版：INT8 对称量化 / 反量化
# ============================================================
if TORCH_AVAILABLE:

    def quantize_per_tensor(w: "torch.Tensor"):
        """per-tensor 对称量化：整个张量共用一个 scale。

        scale = max(|w|) / 127；q = round(w / scale)，clamp 到 [-127,127]。
        """
        scale = w.abs().max() / QMAX
        scale = torch.clamp(scale, min=1e-8)
        q = torch.clamp(torch.round(w / scale), -QMAX, QMAX).to(torch.int8)
        return q, scale

    def dequantize_per_tensor(q: "torch.Tensor", scale) -> "torch.Tensor":
        """反量化：w_hat = q * scale。"""
        return q.to(torch.float32) * scale

    def quantize_per_channel(w: "torch.Tensor"):
        """per-channel 对称量化：每个输出通道（行）独立 scale。

        w 形状 [out_features, in_features]，沿 dim=1 求每行最大绝对值。
        """
        scale = w.abs().amax(dim=1, keepdim=True) / QMAX
        scale = torch.clamp(scale, min=1e-8)
        q = torch.clamp(torch.round(w / scale), -QMAX, QMAX).to(torch.int8)
        return q, scale

    def dequantize_per_channel(q: "torch.Tensor", scale) -> "torch.Tensor":
        return q.to(torch.float32) * scale


# ============================================================
#  NumPy 版：等价实现
# ============================================================
def np_quant_per_tensor(w):
    scale = np.abs(w).max() / QMAX
    scale = max(scale, 1e-8)
    q = np.clip(np.round(w / scale), -QMAX, QMAX).astype(np.int8)
    return q, scale


def np_quant_per_channel(w):
    scale = np.abs(w).max(axis=1, keepdims=True) / QMAX
    scale = np.clip(scale, 1e-8, None)
    q = np.clip(np.round(w / scale), -QMAX, QMAX).astype(np.int8)
    return q, scale


def np_dequant(q, scale):
    return q.astype(np.float32) * scale


# ============================================================
#  讲解
# ============================================================
def explain_quant():
    print(SEP)
    print("权重量化原理（对称量化）")
    print(SEP)
    print("""
  量化    : q     = clamp(round(w / scale), -127, 127)   # 存 INT8
  反量化  : w_hat = q * scale                            # 计算时还原
  scale   : 对称量化用 max(|w|)/127（零点固定为 0）

  粒度对比：
    per-tensor  —— 整张量一个 scale：省事，但一个异常大的权重会把
                   scale 拉大，导致大部分小权重量化后精度损失严重。
    per-channel —— 每个输出通道一个 scale：更贴合各通道的实际范围，
                   误差更小，是实践中权重量化的主流做法。

  显存节省：FP32(4B) -> INT8(1B) 约 4x；FP16(2B) -> INT8(1B) 约 2x。
""")


def report_error(name, w_ref, w_hat):
    """打印量化误差（均方误差 + 最大绝对误差）。"""
    if TORCH_AVAILABLE:
        mse = torch.mean((w_ref - w_hat) ** 2).item()
        max_err = (w_ref - w_hat).abs().max().item()
    else:
        mse = float(np.mean((w_ref - w_hat) ** 2))
        max_err = float(np.abs(w_ref - w_hat).max())
    print(f"  {name:<28} MSE={mse:.3e}  最大绝对误差={max_err:.3e}")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_quant()

    out_features, in_features = 256, 512
    print(SEP)
    print(f"实验：线性层权重 W [{out_features}, {in_features}]")
    print(f"运行设备：DEVICE = {DEVICE}"
          + ("（真实 GPU）" if DEVICE == "cuda" else "（CPU，逻辑完整）"))
    print(SEP)

    if TORCH_AVAILABLE:
        torch.manual_seed(0)
        # 构造带异常值的权重：大部分是小值，少数几个是大值（凸显 per-tensor 缺点）
        W = torch.randn(out_features, in_features, device=DEVICE) * 0.1
        W[0, 0] = 5.0   # 人为注入一个异常大权重
        W[10, 5] = -4.0
        x = torch.randn(in_features, device=DEVICE)

        # 原始 FP32 输出作为参考
        y_ref = W @ x

        # ---- per-tensor ----
        q_t, s_t = quantize_per_tensor(W)
        W_t = dequantize_per_tensor(q_t, s_t)
        y_t = W_t @ x

        # ---- per-channel ----
        q_c, s_c = quantize_per_channel(W)
        W_c = dequantize_per_channel(q_c, s_c)
        y_c = W_c @ x

        # ---- FP16 ----
        W_fp16 = W.half()
        y_fp16 = W_fp16.float() @ x

        print("[权重重建误差]（反量化后与原始 FP32 权重比）")
        report_error("per-tensor INT8", W, W_t)
        report_error("per-channel INT8", W, W_c)
        report_error("FP16", W, W_fp16.float())

        print()
        print("[层输出误差]（量化权重计算 y=Wx 与 FP32 输出比）")
        report_error("per-tensor INT8", y_ref, y_t)
        report_error("per-channel INT8", y_ref, y_c)
        report_error("FP16", y_ref, y_fp16)

        # ---- 显存/存储占用 ----
        bytes_fp32 = W.numel() * 4
        bytes_fp16 = W.numel() * 2
        bytes_int8 = W.numel() * 1  # scale 很小，忽略不计
        print()
        print("[存储占用] W 共 %d 个权重" % W.numel())
        print(f"  FP32 : {bytes_fp32/1024:8.1f} KB  (基准)")
        print(f"  FP16 : {bytes_fp16/1024:8.1f} KB  (节省 {bytes_fp32/bytes_fp16:.1f}x)")
        print(f"  INT8 : {bytes_int8/1024:8.1f} KB  (节省 {bytes_fp32/bytes_int8:.1f}x)")

        # ---- 精度对齐验证 ----
        rel = (y_ref - y_c).abs().max().item() / (y_ref.abs().max().item() + 1e-8)
        print()
        print(f"[精度对齐] per-channel 层输出相对最大误差 = {rel:.3%}")
        print("[结果] " + ("通过 ✓（per-channel 明显优于 per-tensor）"
                          if rel < 0.05 else "误差偏大 ✗"))

    else:
        rng = np.random.default_rng(0)
        W = (rng.standard_normal((out_features, in_features)) * 0.1).astype(np.float32)
        W[0, 0] = 5.0
        W[10, 5] = -4.0
        x = rng.standard_normal(in_features).astype(np.float32)
        y_ref = W @ x

        q_t, s_t = np_quant_per_tensor(W)
        W_t = np_dequant(q_t, s_t)
        y_t = W_t @ x

        q_c, s_c = np_quant_per_channel(W)
        W_c = np_dequant(q_c, s_c)
        y_c = W_c @ x

        W_fp16 = W.astype(np.float16)
        y_fp16 = W_fp16.astype(np.float32) @ x

        print("[权重重建误差]（反量化后与原始 FP32 权重比）")
        report_error("per-tensor INT8", W, W_t)
        report_error("per-channel INT8", W, W_c)
        report_error("FP16", W, W_fp16.astype(np.float32))

        print()
        print("[层输出误差]（量化权重计算 y=Wx 与 FP32 输出比）")
        report_error("per-tensor INT8", y_ref, y_t)
        report_error("per-channel INT8", y_ref, y_c)
        report_error("FP16", y_ref, y_fp16)

        bytes_fp32 = W.size * 4
        bytes_int8 = W.size * 1
        print()
        print("[存储占用] W 共 %d 个权重" % W.size)
        print(f"  FP32 : {bytes_fp32/1024:8.1f} KB  (基准)")
        print(f"  INT8 : {bytes_int8/1024:8.1f} KB  (节省 {bytes_fp32/bytes_int8:.1f}x)")

        rel = np.abs(y_ref - y_c).max() / (np.abs(y_ref).max() + 1e-8)
        print()
        print(f"[精度对齐] per-channel 层输出相对最大误差 = {rel:.3%}")
        print("[结果] " + ("通过 ✓（per-channel 明显优于 per-tensor）"
                          if rel < 0.05 else "误差偏大 ✗"))

    print()
    print(SEP)
    if DEVICE == "cuda":
        print("环境：CUDA GPU 就绪，量化/反量化在 GPU 上真实运行。")
    elif TORCH_AVAILABLE:
        print("环境：无 CUDA GPU，已在 CPU 上真实运行完整量化逻辑。")
    else:
        print("环境：未安装 PyTorch，已用 NumPy 演示等价原理。")
    print(SEP)


if __name__ == "__main__":
    main()
