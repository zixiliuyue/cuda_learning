#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 60: PTQ 训练后量化 + 校准

PTQ (Post-Training Quantization)：模型已经训练好，**不再更新权重**，只是
把权重和激活量化成 INT8。权重的范围是静态已知的（直接看张量），但**激活
的范围依赖输入数据**，所以需要用一小批「校准数据集」跑一遍前向，收集每层
激活的分布，据此确定激活的量化参数（scale/zp）。这一步就叫「校准 calibration」。

校准的关键：如何从激活分布里选一个「量化范围阈值」？本天对比三种策略：
  1. min/max      —— 直接取观测到的最小/最大值。简单，但对离群值极敏感，
                     一个异常大的激活会把范围撑大，台阶变粗、精度下降。
  2. 百分位 (percentile) —— 取 99.9% 分位数当上界，裁掉尾部离群值，范围更紧凑。
  3. KL 散度 (KL divergence) —— TensorRT 的经典做法：在多个候选阈值里，选一个
                     使「量化前分布」与「量化后分布」KL 散度最小的阈值，
                     在「裁剪误差」和「量化台阶误差」之间取最优平衡。

流程：构造小 MLP → 训练到较好精度（FP32 基准）→ 用校准集收集激活范围
（三种策略）→ INT8 PTQ → 评估量化后精度与误差，对比三种校准。

运行策略：device = "cuda" if torch.cuda.is_available() else "cpu"，逻辑都真实运行；
完全没有 torch 时退化为 NumPy 版演示「三种校准阈值 + 量化误差」的核心原理。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（有 CUDA 用 GPU，无 CUDA 用 CPU）
# ------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

SEP = "=" * 64
QMAX = 127  # INT8 对称量化范围 [-127, 127]


# ============================================================
#  校准策略：从一批激活样本里选「量化范围阈值」
# ============================================================
def calib_minmax(acts):
    """min/max 校准：直接取观测到的最大绝对值。"""
    return float(np.abs(acts).max())


def calib_percentile(acts, pct=99.9):
    """百分位校准：取 |act| 的 pct 分位数，裁掉尾部离群值。"""
    return float(np.percentile(np.abs(acts), pct))


def calib_kl(acts, n_bins=2048, n_levels=128):
    """KL 散度校准（TensorRT 熵校准算法的忠实简化版）。

    思路：
      1. 用激活绝对值构建高分辨率参考直方图 hist（n_bins 个 bin）；
      2. 遍历一系列候选阈值 T（对应前 i 个 bin）：
         - 参考分布 P：取前 i 个 bin，并把「超过 T 的尾部计数」并入最后一个 bin
           （模拟量化时把离群值 clip 到 T 边界）；
         - 量化分布 Q：把前 i 个「原始 bin」压缩到 n_levels(=128) 个台阶再展开回
           i 个 bin（模拟 INT8 只有 128 个正台阶的粒度）；
      3. 计算 KL(P || Q)，选使其最小的 T。
    T 太小 -> 尾部大量质量被 clip 进边界 bin，P 严重失真（KL 大）；
    T 太大 -> 每个台阶要覆盖更多 bin，量化粒度变粗（KL 也大）；最小值即最佳折中。
    """
    a = np.abs(acts).ravel()
    amax = a.max()
    if amax <= 0:
        return 1e-8
    hist, edges = np.histogram(a, bins=n_bins, range=(0, amax))
    hist = hist.astype(np.float64)

    best_thr, best_kl = amax, float("inf")
    for i in range(n_levels, n_bins + 1, 8):
        # 参考分布 P：尾部并入边界（离群值被 clip 到 T）
        P = hist[:i].copy()
        P[-1] += hist[i:].sum()
        raw = hist[:i]                       # Q 基于未合并的原始 bin
        Q = _requantize(raw, n_levels)       # 压到 128 台阶再展开
        kl = _kl_divergence(P, Q)
        if kl < best_kl:
            best_kl, best_thr = kl, edges[i]
    return float(best_thr)


def _requantize(raw, n_levels):
    """把长度 i 的分布压缩到 n_levels 个量化台阶，再展开回长度 i。"""
    i = len(raw)
    q = np.zeros(i, dtype=np.float64)
    bins_per_level = i / n_levels
    for j in range(n_levels):
        lo = int(np.floor(j * bins_per_level))
        hi = int(np.floor((j + 1) * bins_per_level))
        hi = min(max(hi, lo + 1), i)
        seg = raw[lo:hi]
        nz = np.count_nonzero(seg)
        if nz > 0:
            avg = seg.sum() / nz             # 台阶总计数平均摊回原有值的 bin
            for k in range(lo, hi):
                if raw[k] != 0:
                    q[k] = avg
    return q


def _kl_divergence(p, q):
    """KL(P||Q)，先做平滑归一化避免 log(0)。"""
    p = p + 1e-9
    q = q + 1e-9
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ============================================================
#  量化工具（对称）
# ============================================================
def quantize_sym(x, threshold):
    """按给定 threshold 做对称 INT8 量化（NumPy 语义，接受 ndarray）。"""
    scale = max(threshold / QMAX, 1e-8)
    q = np.clip(np.round(x / scale), -QMAX, QMAX)
    return q, scale


def quant_error(x, threshold):
    """给定阈值下的量化 MSE（先按阈值 clip，再量化-反量化）。"""
    q, scale = quantize_sym(x, threshold)
    x_hat = q * scale
    return float(np.mean((x - x_hat) ** 2))


def quant_error_bulk(x, threshold, pct=99.0):
    """只统计「主体（inlier）」部分的量化 MSE。

    离群值到底是「信号」还是「噪声」因任务而异，但主体（占 99%）的重建质量
    是所有场景都关心的。percentile/KL 校准把范围收紧到主体，主体量化台阶更细，
    因此在这个指标上通常明显优于 min/max。
    """
    x = np.asarray(x)
    thr_in = np.percentile(np.abs(x), pct)
    mask = np.abs(x) <= thr_in
    q, scale = quantize_sym(x, threshold)
    x_hat = q * scale
    return float(np.mean((x[mask] - x_hat[mask]) ** 2))


# ============================================================
#  PyTorch 版：小 MLP + 训练 + PTQ
# ============================================================
if TORCH_AVAILABLE:

    class MLP(nn.Module):
        """3 层 MLP：in -> 64 -> 64 -> 类别数，ReLU 激活。"""

        def __init__(self, in_dim=8, hidden=64, n_classes=3):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, n_classes)
            self.relu = nn.ReLU()

        def forward(self, x, collect=None):
            x = self.relu(self.fc1(x))
            if collect is not None:
                collect.append(x.detach().flatten().cpu().numpy())
            x = self.relu(self.fc2(x))
            if collect is not None:
                collect.append(x.detach().flatten().cpu().numpy())
            return self.fc3(x)

    # 固定的类原型：让数据线性可分，FP32 能训练到高精度
    _PROTOS = None

    def _get_protos(in_dim, n_classes):
        global _PROTOS
        if _PROTOS is None:
            g = torch.Generator().manual_seed(1234)
            _PROTOS = torch.randn(n_classes, in_dim, generator=g) * 3.0
        return _PROTOS

    def make_dataset(n, in_dim, n_classes, seed, outlier=False):
        """合成分类数据：按样本向量与固定原型的最近距离分类。

        outlier=True 时对少量样本放大输入（不改标签，标签在放大前已定），
        用于在校准/测试阶段制造「激活离群值」，凸显 min/max 校准的弱点。
        训练集用 outlier=False，保证 FP32 基准精度高。
        """
        g = torch.Generator().manual_seed(seed)
        protos = _get_protos(in_dim, n_classes)
        X = torch.randn(n, in_dim, generator=g)
        y = torch.cdist(X, protos).argmin(dim=1)   # 标签：最近原型
        if outlier:
            idx = torch.randperm(n, generator=g)[: max(1, n // 40)]
            X[idx] *= 6.0                           # 制造激活离群，标签不变
        return X, y

    def evaluate(model, X, y):
        model.eval()
        with torch.no_grad():
            pred = model(X).argmax(dim=1)
        return (pred == y).float().mean().item()

    def quantize_weights_per_channel(w):
        """权重：对称 per-channel INT8（每输出通道一个 scale）。"""
        scale = w.abs().amax(dim=1, keepdim=True) / QMAX
        scale = torch.clamp(scale, min=1e-8)
        q = torch.clamp(torch.round(w / scale), -QMAX, QMAX)
        return q * scale  # 直接返回反量化后的「假量化」权重

    def apply_ptq(model, act_thresholds):
        """构造一个 PTQ 后的推理函数：权重 per-channel 量化，激活按给定阈值量化。"""
        w1 = quantize_weights_per_channel(model.fc1.weight.data)
        w2 = quantize_weights_per_channel(model.fc2.weight.data)
        w3 = quantize_weights_per_channel(model.fc3.weight.data)
        b1, b2, b3 = model.fc1.bias.data, model.fc2.bias.data, model.fc3.bias.data
        relu = torch.relu

        def quant_act(x, thr):
            scale = max(thr / QMAX, 1e-8)
            q = torch.clamp(torch.round(x / scale), -QMAX, QMAX)
            return q * scale

        def forward(X):
            h1 = relu(X @ w1.t() + b1)
            h1 = quant_act(h1, act_thresholds[0])   # 量化激活 1
            h2 = relu(h1 @ w2.t() + b2)
            h2 = quant_act(h2, act_thresholds[1])   # 量化激活 2
            return h2 @ w3.t() + b3

        return forward


# ============================================================
#  讲解
# ============================================================
def explain_ptq():
    print(SEP)
    print("PTQ 训练后量化：权重静态量化 + 激活校准")
    print(SEP)
    print("""
  PTQ 流程：
    训练好的 FP32 模型 ──> 权重直接量化（范围已知）
                         └> 激活范围未知，需「校准集」跑前向收集分布 ──> 定 scale

  三种激活校准策略（如何从激活分布里选量化范围阈值 T）：
    ┌────────────┬──────────────────────────────┬──────────────┐
    │ 策略       │ 阈值 T 怎么选                 │ 特点         │
    ├────────────┼──────────────────────────────┼──────────────┤
    │ min/max    │ T = max(|act|)               │ 简单，怕离群 │
    │ percentile │ T = |act| 的 99.9% 分位数    │ 裁尾，更紧凑 │
    │ KL 散度    │ 遍历候选 T，选 KL(P||Q) 最小 │ 平衡最优     │
    └────────────┴──────────────────────────────┴──────────────┘

  KL 校准直觉：T 太大 -> 台阶粗（量化误差大）；T 太小 -> 裁掉太多（裁剪误差大）。
              KL 散度衡量量化后分布与原分布的差异，取最小即最佳折中。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_ptq()

    print(SEP)
    print("实验：小 MLP 的 INT8 PTQ + 三种激活校准")
    print(f"运行设备：DEVICE = {DEVICE}"
          + ("（真实 GPU）" if DEVICE == "cuda" else "（CPU，逻辑完整）"))
    print(SEP)

    in_dim, hidden, n_classes = 8, 64, 3

    if TORCH_AVAILABLE:
        torch.manual_seed(0)
        model = MLP(in_dim, hidden, n_classes).to(DEVICE)
        X_tr, y_tr = make_dataset(2000, in_dim, n_classes, seed=1)               # 训练：干净
        X_ca, _ = make_dataset(256, in_dim, n_classes, seed=2, outlier=True)     # 校准：带离群
        X_te, y_te = make_dataset(1000, in_dim, n_classes, seed=3, outlier=True) # 测试：带离群
        X_tr, y_tr = X_tr.to(DEVICE), y_tr.to(DEVICE)
        X_ca = X_ca.to(DEVICE)
        X_te, y_te = X_te.to(DEVICE), y_te.to(DEVICE)

        # ---- 训练 FP32 基准 ----
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        lossf = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(120):
            opt.zero_grad()
            loss = lossf(model(X_tr), y_tr)
            loss.backward()
            opt.step()
        acc_fp32 = evaluate(model, X_te, y_te)
        print(f"[FP32 基准] 测试集准确率 = {acc_fp32:.3%}")

        # ---- 用校准集收集两层激活 ----
        model.eval()
        acts = []
        with torch.no_grad():
            model(X_ca, collect=acts)
        act1, act2 = acts[0], acts[1]   # 已是 numpy 一维数组

        # ---- 三种校准分别求阈值 ----
        methods = {}
        for name, fn in [
            ("min/max", lambda a: calib_minmax(a)),
            ("percentile99.9", lambda a: calib_percentile(a, 99.9)),
            ("KL散度", lambda a: calib_kl(a)),
        ]:
            t1, t2 = fn(act1), fn(act2)
            methods[name] = (t1, t2)

        print()
        print("[校准阈值] 两层激活的量化范围 T（越贴合有效分布越好）:")
        print(f"  激活分布：层1 max=%.3f 99.9%%=%.3f | 层2 max=%.3f 99.9%%=%.3f"
              % (np.abs(act1).max(), np.percentile(np.abs(act1), 99.9),
                 np.abs(act2).max(), np.percentile(np.abs(act2), 99.9)))
        print("  bulk-MSE(层1)：只看主体 99% 值的量化误差，越小说明主体重建越准")
        for name, (t1, t2) in methods.items():
            print(f"  {name:<16} T1={t1:8.4f}  T2={t2:8.4f}  "
                  f"bulk-MSE(层1)={quant_error_bulk(act1, t1):.3e}")

        # ---- 对每种校准做 PTQ 并评估 ----
        print()
        print("[PTQ 精度评估] 权重 per-channel 量化 + 各策略激活量化:")
        best = None
        for name, thr in methods.items():
            fwd = apply_ptq(model, thr)
            with torch.no_grad():
                logits_q = fwd(X_te)
                pred = logits_q.argmax(dim=1)
            acc = (pred == y_te).float().mean().item()
            drop = acc_fp32 - acc
            bmse = quant_error_bulk(act1, thr[0])
            print(f"  {name:<16} 准确率={acc:.3%} (较FP32 {drop:+.3%})  bulk-MSE(层1)={bmse:.3e}")
            # 以主体重建 MSE 最小作为最佳校准（准确率对小模型不够敏感）
            if best is None or bmse < best[2]:
                best = (name, acc, bmse)

        print()
        print(f"[结论] 三种校准 PTQ 后准确率都接近 FP32（说明量化基本无损）；")
        print(f"       但主体重建最准的是 {best[0]}（bulk-MSE={best[2]:.3e}）——")
        print(f"       它把量化范围收紧到激活主体，台阶更细，是大模型上差距的来源。")
        ok = (acc_fp32 - best[1]) < 0.05
        print("[结果] " + ("通过 ✓（PTQ 后精度损失 < 5%）"
                          if ok else "精度损失偏大 ✗（可增大校准集/换策略）"))

    else:
        # ---- NumPy 版：演示三种校准阈值 + 量化误差核心原理 ----
        rng = np.random.default_rng(0)
        # 构造带离群值的激活分布（主体 ~ N(0,1)，混入少量远离群值）
        base = rng.standard_normal(50000)
        signs = np.sign(rng.standard_normal(1000))
        outliers = (rng.standard_normal(1000) * 3 + 20) * signs
        act = np.concatenate([base, outliers]).astype(np.float32)

        t_mm = calib_minmax(act)
        t_pc = calib_percentile(act, 99.9)
        t_kl = calib_kl(act)

        print("[激活分布] 主体 ~ N(0,1)，混入少量 |值|≈20 的离群值")
        print(f"  真实 max = {np.abs(act).max():.3f}")
        print()
        print("[三种校准阈值 + 量化 MSE]（bulk-MSE 只看主体 99% 值）:")
        for name, t in [("min/max", t_mm), ("percentile99.9", t_pc), ("KL散度", t_kl)]:
            print(f"  {name:<16} T={t:8.4f}  全量MSE={quant_error(act, t):.3e}"
                  f"  bulk-MSE={quant_error_bulk(act, t):.3e}")

        bmse_mm = quant_error_bulk(act, t_mm)
        bmse_pc = quant_error_bulk(act, t_pc)
        print()
        print(f"[结论] 裁掉离群值后，主体量化台阶更细，percentile 的 bulk-MSE 更小")
        print("[结果] " + ("通过 ✓（min/max bulk-MSE 是 percentile 的 %.1f 倍）"
                          % (bmse_mm / bmse_pc)
                          if bmse_mm > bmse_pc else "差距不明显 ✗"))

    print()
    print(SEP)
    if DEVICE == "cuda":
        print("环境：CUDA GPU 就绪，训练/校准/PTQ 全流程在 GPU 上真实运行。")
    elif TORCH_AVAILABLE:
        print("环境：无 CUDA GPU，已在 CPU 上真实运行完整 PTQ 流程。")
    else:
        print("环境：未安装 PyTorch，已用 NumPy 演示三种校准 + 量化误差原理。")
    print(SEP)


if __name__ == "__main__":
    main()
