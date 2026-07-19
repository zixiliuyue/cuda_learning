#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 62: 大模型量化 — GPTQ / AWQ / SmoothQuant 原理

LLM 量化和小模型（Day59-61）最大的不同：**激活里存在系统性的离群通道
(outlier channels)**。Transformer 的某些特征维度激活值特别大（可达其他通道的
几十上百倍），而权重分布相对平稳。直接对激活做 INT8 量化，这些离群通道会把
scale 撑大，导致其余通道精度崩溃。三种主流方法各有思路：

  ┌────────────┬──────────────────────────────────────────────────────┐
  │ SmoothQuant│ 把激活的「难量化」迁移一部分到权重：对激活除以平滑因子 s、│
  │            │ 对权重乘以 s，数学等价但让两边都变得好量化（激活+权重 W8A8）│
  │ AWQ        │ 发现「重要权重」由大激活通道决定：按激活幅度缩放权重，保护 │
  │            │ 重要通道，仅量化权重（W4A16），无需反缩放激活             │
  │ GPTQ       │ 逐列量化权重，用 Hessian(二阶信息) 补偿已量化列引入的误差，│
  │            │ 逐步把误差「转嫁」给尚未量化的列，精度极高（W4A16）      │
  └────────────┴──────────────────────────────────────────────────────┘

本天以原理为主，并**真实复现 SmoothQuant 的核心步骤**：
  1. 构造带离群通道的激活 X 与权重 W；
  2. 直接量化的误差（基线）；
  3. 用平滑因子 s = (max|X|)^a / (max|W|)^(1-a) 迁移难度，再量化，验证误差下降；
  4. 附带一个「AWQ 式按激活重要性缩放权重」的迷你复现做对照。
最后打印 GPTQ / AWQ / SmoothQuant 三者对比表。

运行策略：device = "cuda" if torch.cuda.is_available() else "cpu"，逻辑都真实运行；
完全没有 torch 时退化为 NumPy 版，复现同样的 SmoothQuant 数学。
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
#  量化工具（NumPy 语义，供两条分支复用）
# ============================================================
def quantize_per_tensor(x, n_bits=8):
    """对称 per-tensor 量化-反量化，返回反量化结果。"""
    qmax = 2 ** (n_bits - 1) - 1
    scale = max(float(np.abs(x).max()) / qmax, 1e-12)
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return q * scale


def quantize_per_channel_cols(w, n_bits=8):
    """对权重按「输入通道（列）」对称量化，per-channel（每列一个 scale）。

    w 形状 [out, in]，SmoothQuant 里权重的输入通道与激活通道对齐，
    因此按列量化能与激活的按通道平滑配合。
    """
    qmax = 2 ** (n_bits - 1) - 1
    scale = np.clip(np.abs(w).max(axis=0, keepdims=True) / qmax, 1e-12, None)
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return q * scale


def rel_error(ref, approx):
    """相对误差：||ref - approx||_F / ||ref||_F。"""
    return float(np.linalg.norm(ref - approx) / (np.linalg.norm(ref) + 1e-12))


# ============================================================
#  SmoothQuant 核心步骤（真实复现）
# ============================================================
def smoothquant_factor(X, W, alpha=0.5):
    """计算 SmoothQuant 的 per-channel 平滑因子 s。

    s_j = (max_i |X_ij|)^alpha / (max_k |W_kj|)^(1-alpha)

    直觉：激活离群通道 j 的 max|X_.j| 很大 -> s_j 大 -> 激活除以 s_j 后被压小；
          代价是该列权重乘以 s_j 变大，但权重原本平稳，能承受。
          alpha 控制迁移多少「难度」到权重侧（0.5 常用）。
    """
    act_max = np.abs(X).max(axis=0)                 # 每个激活通道的幅度 [in]
    wt_max = np.abs(W).max(axis=0)                  # 每个权重输入通道的幅度 [in]
    s = (act_max ** alpha) / (np.power(wt_max, 1 - alpha) + 1e-12)
    s = np.clip(s, 1e-4, 1e4)
    return s


def awq_scale(X, W, alpha=0.5):
    """AWQ 式「按激活重要性缩放权重」的迷你版本。

    AWQ 观察：激活大的输入通道对应的权重更「重要」，量化时应被保护。
    做法（简化）：对重要通道的权重放大 s 再量化、量化后除回 s，
    等价于给这些通道更细的有效量化台阶。这里 s 由激活幅度决定。
    """
    act_max = np.abs(X).max(axis=0)
    act_max = act_max / (act_max.mean() + 1e-12)    # 归一化到均值附近
    s = act_max ** alpha                            # 重要通道 s>1
    s = np.clip(s, 1e-2, 1e2)
    return s


# ============================================================
#  讲解
# ============================================================
def explain_llm_quant():
    print(SEP)
    print("大模型量化：激活离群通道问题 + 三种主流解法")
    print(SEP)
    print("""
  LLM 的痛点：激活里有「离群通道」，个别特征维度的激活值极大（几十~上百倍），
  而权重分布平稳。直接量化激活，离群通道撑大 scale，其余通道精度崩溃。

     激活 X（按通道看幅度）：
       通道:  0    1    2 *离群*  3    4    5
       |X|:  1.2  0.9  ██████89   1.1  0.8  1.0
                        ↑ 一个巨大的离群通道毒化整个 per-tensor scale

  ┌ SmoothQuant ─ 迁移难度：X'=X/s, W'=W·s（数学等价 X'W'^T = XW^T）───┐
  │   激活除以 s 变平缓（好量化），难度转嫁给平稳的权重（还扛得住）。  │
  │   目标 W8A8：权重和激活都量化成 INT8。                             │
  └───────────────────────────────────────────────────────────────────┘
  ┌ AWQ ─ 保护重要权重：按激活幅度缩放权重，重要通道给更细台阶 ────────┐
  │   只量化权重（W4A16），激活保持 FP16，无需反缩放激活。            │
  └───────────────────────────────────────────────────────────────────┘
  ┌ GPTQ ─ 误差补偿：逐列量化权重，用 Hessian 二阶信息把量化误差 ──────┐
  │   转嫁给尚未量化的列，逐步补偿，精度最高（W4A16），但最慢。        │
  └───────────────────────────────────────────────────────────────────┘

  关键等式（SmoothQuant 数学等价性）：
     Y = X · W^T = (X / s) · (W · s)^T = X' · W'^T     （s 为 per-channel 向量）
""")


def print_compare_table():
    print(SEP)
    print("GPTQ / AWQ / SmoothQuant 对比表")
    print(SEP)
    rows = [
        ("维度", "SmoothQuant", "AWQ", "GPTQ"),
        ("量化对象", "权重+激活 W8A8", "仅权重 W4A16", "仅权重 W4A16"),
        ("核心思想", "激活难度迁移到权重", "按激活重要性缩放权重", "Hessian 误差补偿"),
        ("是否需校准", "需(定平滑因子)", "需(定缩放)", "需(算Hessian)"),
        ("激活离群处理", "平滑因子s压制", "保护对应权重", "不直接处理激活"),
        ("速度", "快", "快", "慢(逐列+求逆)"),
        ("典型精度", "高", "很高", "最高"),
        ("适用场景", "W8A8推理加速", "W4权重压缩", "W4极致精度"),
    ]
    w0, w1, w2, w3 = 14, 20, 20, 18
    for i, (a, b, c, d) in enumerate(rows):
        print(f"  {a:<{w0}}{b:<{w1}}{c:<{w2}}{d:<{w3}}")
        if i == 0:
            print("  " + "-" * (w0 + w1 + w2 + w3))


# ============================================================
#  主流程
# ============================================================
def run_smoothquant_experiment(X, W, alpha=0.5):
    """在给定 X[batch,in]、W[out,in] 上对比三种方案的量化误差。

    返回 (基线相对误差, SmoothQuant, AWQ) 以及量化后输出的相对误差。
    """
    Y_ref = X @ W.T                                  # FP32 参考输出

    # ---- 基线：直接量化（激活 per-tensor + 权重 per-channel）----
    Xq = quantize_per_tensor(X)
    Wq = quantize_per_channel_cols(W)
    Y_base = Xq @ Wq.T
    err_base = rel_error(Y_ref, Y_base)

    # ---- SmoothQuant：X'=X/s, W'=W*s，再量化 ----
    s = smoothquant_factor(X, W, alpha)              # [in]
    Xs = X / s                                       # 广播到列
    Ws = W * s                                       # W 的列 j 乘 s_j
    Xsq = quantize_per_tensor(Xs)
    Wsq = quantize_per_channel_cols(Ws)
    Y_sq = Xsq @ Wsq.T                               # 数学上等价 X'W'^T = XW^T
    err_sq = rel_error(Y_ref, Y_sq)

    # ---- AWQ 式：按激活重要性缩放权重（仅权重量化，激活保持 FP32）----
    #   注意：AWQ 的缩放要在「比 per-channel 更粗」的量化粒度下才有意义
    #   （per-tensor / per-group）。若权重按列 per-channel 量化，按列缩放会被
    #   scale 精确抵消（缩放变成 no-op）。故这里权重用 per-tensor 量化，
    #   放大重要通道 -> 它们获得相对更细的有效台阶 -> 量化后除回，得到保护。
    sa = awq_scale(X, W, alpha)                       # [in]
    Wa = W * sa
    Waq = quantize_per_tensor(Wa)                     # per-tensor（粗粒度）
    Wa_hat = Waq / sa                                 # 量化后除回缩放
    Y_awq = X @ Wa_hat.T                              # 激活不量化（W4A16 类比）
    err_awq = rel_error(Y_ref, Y_awq)

    return err_base, err_sq, err_awq


def main():
    explain_llm_quant()

    print(SEP)
    print("实验：复现 SmoothQuant / AWQ 核心，验证离群通道下量化误差下降")
    print(f"运行设备：DEVICE = {DEVICE}"
          + ("（真实 GPU）" if DEVICE == "cuda" else "（CPU，逻辑完整）"))
    print(SEP)

    batch, in_dim, out_dim = 64, 256, 128
    rng = np.random.default_rng(0)

    # 构造带离群通道的激活：主体 ~N(0,1)，少数几个通道幅度放大几十倍
    X = rng.standard_normal((batch, in_dim)).astype(np.float32)
    outlier_ch = rng.choice(in_dim, size=6, replace=False)
    X[:, outlier_ch] *= 40.0                          # 离群通道
    # 权重：平稳分布（LLM 权重通常比激活好量化）
    W = (rng.standard_normal((out_dim, in_dim)) * 0.1).astype(np.float32)

    if TORCH_AVAILABLE:
        # 用 torch 在目标设备上真实计算（把 numpy 语义搬到张量）
        Xt = torch.from_numpy(X).to(DEVICE)
        Wt = torch.from_numpy(W).to(DEVICE)
        # 复用 numpy 实现前，先确认结果一致：这里直接用 numpy 逻辑，
        # 但把大矩阵乘法交给 GPU 验证一遍数值一致性。
        Y_ref_gpu = (Xt @ Wt.t()).cpu().numpy()
        Y_ref_cpu = X @ W.T
        consistent = rel_error(Y_ref_cpu, Y_ref_gpu) < 1e-4
        print(f"[数值一致性] GPU 与 CPU 计算 XW^T 相对误差 < 1e-4：{consistent}")

    print(f"[激活] 形状 {X.shape}，注入 {len(outlier_ch)} 个离群通道（×40）")
    print(f"[权重] 形状 {W.shape}，平稳分布（std≈0.1）")
    print()

    print("[不同 alpha 下 SmoothQuant 的输出相对误差]（alpha 控制难度迁移比例）:")
    print(f"  {'alpha':>8}{'基线(直接量化)':>18}{'SmoothQuant':>16}{'AWQ式':>12}")
    best = None
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        eb, es, ea = run_smoothquant_experiment(X, W, alpha)
        print(f"  {alpha:8.1f}{eb:18.4e}{es:16.4e}{ea:12.4e}")
        if best is None or es < best[1]:
            best = (alpha, es)

    # 用最优 alpha 报告收益
    eb, es, ea = run_smoothquant_experiment(X, W, best[0])
    print()
    print(f"[最佳] alpha={best[0]}：SmoothQuant 相对误差 {es:.4e}，"
          f"基线 {eb:.4e}")
    improve = eb / (es + 1e-12)
    print(f"[结论] SmoothQuant 把量化误差降到基线的 1/{improve:.1f}")
    print("[结果] " + ("通过 ✓（迁移离群难度后量化误差显著下降）"
                      if es < eb else "未见改善 ✗（可增大离群幅度/调 alpha）"))

    print()
    print_compare_table()

    print()
    print(SEP)
    if DEVICE == "cuda":
        print("环境：CUDA GPU 就绪，矩阵乘法在 GPU 上验证，量化逻辑真实运行。")
    elif TORCH_AVAILABLE:
        print("环境：无 CUDA GPU，已在 CPU 上真实运行完整 SmoothQuant/AWQ 复现。")
    else:
        print("环境：未安装 PyTorch，已用 NumPy 复现 SmoothQuant/AWQ 核心数学。")
    print(SEP)


if __name__ == "__main__":
    main()
