#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 59: 量化基础 — 对称/非对称、per-tensor/per-channel

Day53 我们初步手写了 INT8 对称权重量化，本天把「量化」这件事拆到最底层，
彻底讲清四个正交的设计维度：

  维度 1：对称 vs 非对称
    对称   —— 零点 zero_point 固定为 0，只需一个 scale。适合分布关于 0 对称
              的权重（如 Linear 权重）。q = round(x/scale)，范围 [-127,127]。
    非对称 —— 允许零点偏移，scale + zero_point 两个参数。适合分布不对称、
              单边的激活（如 ReLU 之后全为正）。q = round(x/scale)+zp，范围 [0,255]。

  维度 2：per-tensor vs per-channel
    per-tensor  —— 整个张量一个量化参数，最省事，但一个离群值会拉大 scale，
                   把大量小值挤到同一个量化台阶上，误差骤增。
    per-channel —— 每个输出通道（行）一套参数，更贴合各通道实际范围，误差更小，
                   是权重量化的主流做法。

本天手写上述量化/反量化，并用「带离群值」的数据凸显 per-channel 的优势，
最后打印一张「量化前后张量对比表」，让你直观看到每个数被压到了哪个整数台阶。

运行策略：device = "cuda" if torch.cuda.is_available() else "cpu"，逻辑都真实运行；
完全没有 torch 时退化为 NumPy 版演示同样的原理。无 GPU 也能 `python quant_basics.py`。
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


# ============================================================
#  PyTorch 版：四种量化的核心实现
# ============================================================
if TORCH_AVAILABLE:

    def sym_quant_per_tensor(x, n_bits=8):
        """对称量化 · per-tensor。

        量化范围 [-qmax, qmax]，qmax = 2^(n_bits-1) - 1（如 8bit -> 127）。
        scale = max(|x|) / qmax；零点固定为 0。
        """
        qmax = 2 ** (n_bits - 1) - 1
        scale = x.abs().max() / qmax
        scale = torch.clamp(scale, min=1e-8)
        q = torch.clamp(torch.round(x / scale), -qmax, qmax)
        return q, scale

    def sym_dequant(q, scale):
        """对称反量化：x_hat = q * scale。"""
        return q * scale

    def asym_quant_per_tensor(x, n_bits=8):
        """非对称量化 · per-tensor。

        量化范围 [0, 2^n_bits - 1]（如 8bit -> [0,255]）。
        scale = (max - min) / (qmax - qmin)
        zp    = round(qmin - min/scale)        # 零点：浮点 0 对应的整数
        q     = clamp(round(x/scale) + zp, qmin, qmax)
        """
        qmin, qmax = 0, 2 ** n_bits - 1
        x_min = torch.minimum(x.min(), torch.tensor(0.0, device=x.device))
        x_max = torch.maximum(x.max(), torch.tensor(0.0, device=x.device))
        scale = (x_max - x_min) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        zp = torch.round(qmin - x_min / scale)
        q = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
        return q, scale, zp

    def asym_dequant(q, scale, zp):
        """非对称反量化：x_hat = (q - zp) * scale。"""
        return (q - zp) * scale

    def sym_quant_per_channel(w, n_bits=8):
        """对称量化 · per-channel（沿 dim=1，每行一个 scale）。"""
        qmax = 2 ** (n_bits - 1) - 1
        scale = w.abs().amax(dim=1, keepdim=True) / qmax
        scale = torch.clamp(scale, min=1e-8)
        q = torch.clamp(torch.round(w / scale), -qmax, qmax)
        return q, scale


# ============================================================
#  NumPy 版：无 torch 时的等价实现
# ============================================================
def np_sym_quant_per_tensor(x, n_bits=8):
    qmax = 2 ** (n_bits - 1) - 1
    scale = max(np.abs(x).max() / qmax, 1e-8)
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return q, scale


def np_asym_quant_per_tensor(x, n_bits=8):
    qmin, qmax = 0, 2 ** n_bits - 1
    x_min = min(x.min(), 0.0)
    x_max = max(x.max(), 0.0)
    scale = max((x_max - x_min) / (qmax - qmin), 1e-8)
    zp = np.round(qmin - x_min / scale)
    q = np.clip(np.round(x / scale) + zp, qmin, qmax)
    return q, scale, zp


def np_sym_quant_per_channel(w, n_bits=8):
    qmax = 2 ** (n_bits - 1) - 1
    scale = np.clip(np.abs(w).max(axis=1, keepdims=True) / qmax, 1e-8, None)
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return q, scale


# ============================================================
#  讲解
# ============================================================
def explain_quant():
    print(SEP)
    print("量化四维度：对称/非对称 × per-tensor/per-channel")
    print(SEP)
    print("""
  ┌─ 对称量化（权重常用，零点=0）───────────────────────────┐
  │  scale = max(|x|) / qmax           (8bit qmax=127)      │
  │  q     = clamp(round(x/scale), -127, 127)               │
  │  x_hat = q * scale                                      │
  └────────────────────────────────────────────────────────┘
  ┌─ 非对称量化（激活常用，零点可偏移）─────────────────────┐
  │  scale = (max-min) / 255                                │
  │  zp    = round(0 - min/scale)      浮点0映射到的整数    │
  │  q     = clamp(round(x/scale)+zp, 0, 255)               │
  │  x_hat = (q - zp) * scale                               │
  └────────────────────────────────────────────────────────┘

  数轴直观（非对称把 [min,max] 线性铺满 [0,255]）：
     浮点  min ────────── 0 ─────────────── max
            │            │                  │
     整数   0 ────────── zp ─────────────── 255

  per-tensor  : 全张量 1 组 (scale[,zp])，离群值会毒化整体精度。
  per-channel : 每行 1 组，隔离离群值，误差显著更小。
""")


def report_error(name, ref, hat):
    if TORCH_AVAILABLE and isinstance(ref, torch.Tensor):
        mse = torch.mean((ref - hat) ** 2).item()
        max_err = (ref - hat).abs().max().item()
    else:
        mse = float(np.mean((ref - hat) ** 2))
        max_err = float(np.abs(ref - hat).max())
    print(f"  {name:<32} MSE={mse:.3e}  最大绝对误差={max_err:.3e}")


def print_compare_table(x, q, hat, extra_title=""):
    """打印「量化前后对比表」：原始值 -> 整数台阶 -> 反量化值。"""
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        q = q.detach().cpu().numpy()
        hat = hat.detach().cpu().numpy()
    print(f"  {'原始 x':>12} {'量化 q(int)':>14} {'反量化 x_hat':>14} {'误差':>12}   {extra_title}")
    for xi, qi, hi in zip(x, q, hat):
        print(f"  {xi:12.4f} {int(round(qi)):14d} {hi:14.4f} {abs(xi - hi):12.4f}")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_quant()

    print(SEP)
    print("实验 1：一维向量的对称 vs 非对称量化（含对比表）")
    print(f"运行设备：DEVICE = {DEVICE}"
          + ("（真实 GPU）" if DEVICE == "cuda" else "（CPU，逻辑完整）"))
    print(SEP)

    # 一个「大部分为正、分布不对称」的小向量（模拟 ReLU 激活）
    sample = [0.0, 0.3, 1.2, 2.5, 4.0, 6.8, 0.1, 3.3]

    if TORCH_AVAILABLE:
        v = torch.tensor(sample, device=DEVICE)
        q_s, s_s = sym_quant_per_tensor(v)
        v_s = sym_dequant(q_s, s_s)
        q_a, s_a, zp_a = asym_quant_per_tensor(v)
        v_a = asym_dequant(q_a, s_a, zp_a)
        print(f"[对称] scale={s_s.item():.5f}")
        print_compare_table(v, q_s, v_s)
        print()
        print(f"[非对称] scale={s_a.item():.5f}, zero_point={int(zp_a.item())}")
        print_compare_table(v, q_a, v_a)
    else:
        v = np.array(sample, dtype=np.float32)
        q_s, s_s = np_sym_quant_per_tensor(v)
        v_s = q_s * s_s
        q_a, s_a, zp_a = np_asym_quant_per_tensor(v)
        v_a = (q_a - zp_a) * s_a
        print(f"[对称] scale={s_s:.5f}")
        print_compare_table(v, q_s, v_s)
        print()
        print(f"[非对称] scale={s_a:.5f}, zero_point={int(zp_a)}")
        print_compare_table(v, q_a, v_a)

    print()
    print("[误差对比] 该向量偏单边分布：")
    if TORCH_AVAILABLE:
        report_error("对称   per-tensor", v, v_s)
        report_error("非对称 per-tensor", v, v_a)
    else:
        report_error("对称   per-tensor", v, v_s)
        report_error("非对称 per-tensor", v, v_a)
    print("  说明：非对称能用满 0~255 全部台阶，单边分布下通常误差更小。")

    print()
    print(SEP)
    print("实验 2：带离群值的权重矩阵 —— per-tensor vs per-channel")
    print(SEP)

    out_features, in_features = 128, 256
    if TORCH_AVAILABLE:
        torch.manual_seed(0)
        W = torch.randn(out_features, in_features, device=DEVICE) * 0.1
        W[3, 7] = 6.0      # 注入离群值：只毒化第 3 行
        W[3, 100] = -5.0
        W[50, 20] = 4.5    # 第 50 行也放一个

        q_t, s_t = sym_quant_per_tensor(W)
        W_t = sym_dequant(q_t, s_t)
        q_c, s_c = sym_quant_per_channel(W)
        W_c = sym_dequant(q_c, s_c)

        report_error("对称 per-tensor  INT8", W, W_t)
        report_error("对称 per-channel INT8", W, W_c)

        # 看未含离群值的「干净行」误差差多少（per-tensor 被离群值拖累最明显）
        clean = 0
        report_error(f"per-tensor  第{clean}行(干净)", W[clean], W_t[clean])
        report_error(f"per-channel 第{clean}行(干净)", W[clean], W_c[clean])
        ratio = (torch.mean((W - W_t) ** 2) / torch.mean((W - W_c) ** 2)).item()
    else:
        rng = np.random.default_rng(0)
        W = (rng.standard_normal((out_features, in_features)) * 0.1).astype(np.float32)
        W[3, 7] = 6.0
        W[3, 100] = -5.0
        W[50, 20] = 4.5

        q_t, s_t = np_sym_quant_per_tensor(W)
        W_t = q_t * s_t
        q_c, s_c = np_sym_quant_per_channel(W)
        W_c = q_c * s_c

        report_error("对称 per-tensor  INT8", W, W_t)
        report_error("对称 per-channel INT8", W, W_c)
        clean = 0
        report_error(f"per-tensor  第{clean}行(干净)", W[clean], W_t[clean])
        report_error(f"per-channel 第{clean}行(干净)", W[clean], W_c[clean])
        ratio = float(np.mean((W - W_t) ** 2) / np.mean((W - W_c) ** 2))

    print()
    print(f"[结论] per-tensor MSE 是 per-channel 的 {ratio:.1f} 倍")
    print("[结果] " + ("通过 ✓（per-channel 隔离了离群值，误差显著更小）"
                      if ratio > 2.0 else "差距不明显（可增大离群值）"))

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
