#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 61: QAT 量化感知训练入门

PTQ（Day60）在训练结束后才量化，模型「没预料到」自己会被量化，遇到对量化
敏感的模型（激活分布长尾、低比特如 INT4）精度会明显掉。**QAT (Quantization-
Aware Training)** 的思路：在训练过程中就插入「伪量化节点 (fake quantize)」，
让模型在前向时体验量化误差，从而学到对量化鲁棒的权重。

核心技术难点：round() 的梯度处处为 0（阶梯函数），反向传播会断掉。解决方案是
**STE (Straight-Through Estimator，直通估计器)**：
  - 前向：正常做「量化 -> 反量化」，引入真实的量化误差；
  - 反向：假装量化是恒等函数，梯度直接「穿过去」（gradient ≈ 1），
          从而权重能照常更新。

  fake_quant(x) = dequant(quant(x))          # 前向：有量化误差
  d fake_quant / d x ≈ 1                      # 反向：STE 直通

本天：
  1. 手写一个支持 STE 的伪量化 autograd.Function；
  2. 搭一个小模型，分别做 FP32 训练、PTQ、QAT；
  3. 用较激进的低比特（INT4）放大差距，对比 PTQ vs QAT 的最终精度。

运行策略：device = "cuda" if torch.cuda.is_available() else "cpu"，逻辑都真实运行；
完全没有 torch 时退化为 NumPy 版，演示 STE 前向有误差/反向直通的核心原理。
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


# ============================================================
#  伪量化 + STE（PyTorch autograd.Function）
# ============================================================
if TORCH_AVAILABLE:

    class FakeQuantize(torch.autograd.Function):
        """对称伪量化节点，反向用 STE 直通。

        前向：q = clamp(round(x/scale), -qmax, qmax)；x_hat = q * scale
        反向：把量化视为恒等映射，梯度直接穿过（在量化范围内），
              范围外（被 clamp 饱和）的位置梯度置 0（那里量化后已饱和，
              继续推大没有意义）。
        """

        @staticmethod
        def forward(ctx, x, scale, qmax):
            q = torch.clamp(torch.round(x / scale), -qmax, qmax)
            x_hat = q * scale
            # 记录哪些位置在有效范围内（未饱和），供反向裁剪梯度
            ctx.save_for_backward((x / scale).abs() <= qmax)
            return x_hat

        @staticmethod
        def backward(ctx, grad_output):
            (in_range,) = ctx.saved_tensors
            # STE：范围内梯度=1（直通），范围外=0
            grad_x = grad_output * in_range.to(grad_output.dtype)
            return grad_x, None, None  # scale/qmax 不需要梯度

    def fake_quant(x, n_bits=8):
        """按 x 当前范围做对称伪量化（per-tensor）。"""
        qmax = 2 ** (n_bits - 1) - 1
        scale = torch.clamp(x.detach().abs().max() / qmax, min=1e-8)
        return FakeQuantize.apply(x, scale, qmax)

    class QuantLinear(nn.Module):
        """带伪量化的线性层：权重伪量化 + 激活伪量化。

        enabled=False 时退化为普通 FP32 Linear（用于 FP32 基准训练）。
        """

        def __init__(self, in_dim, out_dim, n_bits=8):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.n_bits = n_bits
            self.enabled = True

        def forward(self, x):
            if self.enabled:
                x = fake_quant(x, self.n_bits)                    # 量化激活
                w = fake_quant(self.linear.weight, self.n_bits)   # 量化权重
                return torch.nn.functional.linear(x, w, self.linear.bias)
            return self.linear(x)

    class QuantMLP(nn.Module):
        """3 层 MLP，中间层用 QuantLinear。"""

        def __init__(self, in_dim=8, hidden=64, n_classes=3, n_bits=8):
            super().__init__()
            self.fc1 = QuantLinear(in_dim, hidden, n_bits)
            self.fc2 = QuantLinear(hidden, hidden, n_bits)
            self.fc3 = QuantLinear(hidden, n_classes, n_bits)
            self.relu = nn.ReLU()

        def set_quant(self, flag):
            for m in [self.fc1, self.fc2, self.fc3]:
                m.enabled = flag

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    _PROTOS = None

    def _get_protos(in_dim, n_classes):
        global _PROTOS
        if _PROTOS is None:
            g = torch.Generator().manual_seed(1234)
            _PROTOS = torch.randn(n_classes, in_dim, generator=g) * 3.0
        return _PROTOS

    def make_dataset(n, in_dim, n_classes, seed):
        g = torch.Generator().manual_seed(seed)
        protos = _get_protos(in_dim, n_classes)
        X = torch.randn(n, in_dim, generator=g)
        y = torch.cdist(X, protos).argmin(dim=1)
        return X, y

    def evaluate(model, X, y):
        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(dim=1) == y).float().mean().item()
        return acc

    def train(model, X, y, epochs=200, lr=0.01):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(model(X), y)
            loss.backward()
            opt.step()
        return loss.item()


# ============================================================
#  NumPy 版：演示 STE 前向有误差 / 反向直通
# ============================================================
def np_fake_quant(x, n_bits=8):
    qmax = 2 ** (n_bits - 1) - 1
    scale = max(np.abs(x).max() / qmax, 1e-8)
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return q * scale, scale, qmax


def np_ste_demo():
    """用一个 1D 最小化问题演示：无 STE 梯度断裂 vs 有 STE 能收敛。

    目标：让 fake_quant(w) 逼近 target。
      - 无 STE：round 的梯度=0，w 永远不动；
      - 有 STE：梯度直通，w 能被推动到使量化值≈target。
    """
    target = 0.37
    w_no_ste = 0.0
    w_ste = 0.0
    lr = 0.5
    for _ in range(200):
        # ---- 无 STE：round 梯度视为 0 -> 权重不更新 ----
        # grad = 0，w_no_ste 保持不变

        # ---- 有 STE：前向量化，反向把 d(quant)/dw 当作 1 ----
        qw, scale, _ = np_fake_quant(np.array([w_ste]))
        qw = qw[0]
        grad = 2 * (qw - target) * 1.0   # STE：*1 直通
        w_ste -= lr * grad
    q_no, _, _ = np_fake_quant(np.array([w_no_ste]))
    q_ste, _, _ = np_fake_quant(np.array([w_ste]))
    return w_no_ste, q_no[0], w_ste, q_ste[0], target


# ============================================================
#  讲解
# ============================================================
def explain_qat():
    print(SEP)
    print("QAT 量化感知训练：伪量化节点 + STE 直通估计器")
    print(SEP)
    print("""
  问题：round() 是阶梯函数，梯度处处为 0，反向传播会断掉，无法训练。

     round(x)
       ^
     3 |          ┌────
     2 |      ┌───┘         斜率处处为 0 -> 梯度=0 -> 权重不更新
     1 |  ┌───┘
       └──┴───┴───┴──> x     （台阶之间是平的）

  STE (Straight-Through Estimator) 解法：
     前向： x ──> [quant -> dequant] ──> x_hat   （引入真实量化误差）
     反向： grad_x  <── [ 视为恒等，梯度=1 ] <── grad_x_hat  （直通）

     直觉：前向让模型「感受」量化误差并学着适应；反向假装量化不存在，
           让梯度顺利回流，权重照常更新。范围外（饱和）位置梯度置 0。

  PTQ vs QAT：
     PTQ —— 训练完才量化，模型没料到量化，低比特下精度掉得多，但零训练成本。
     QAT —— 训练时就带量化，模型主动学习鲁棒权重，精度更高，但要额外训练。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_qat()

    print(SEP)
    print("实验 A：STE 为什么必要（round 梯度断裂 vs STE 直通）")
    print(SEP)
    w_no, q_no, w_ste, q_ste, target = np_ste_demo()
    print(f"  目标：让量化后的值逼近 target = {target}")
    print(f"  无 STE：w 停在 {w_no:.4f}，量化值={q_no:.4f}（梯度=0，纹丝不动）")
    print(f"  有 STE：w 收敛到 {w_ste:.4f}，量化值={q_ste:.4f}（成功逼近 target）")
    print("  [结论] STE 让量化节点可训练，是 QAT 的基石。")

    print()
    print(SEP)
    print("实验 B：PTQ vs QAT 最终精度对比（用 INT4 放大差距）")
    print(f"运行设备：DEVICE = {DEVICE}"
          + ("（真实 GPU）" if DEVICE == "cuda" else "（CPU，逻辑完整）"))
    print(SEP)

    if not TORCH_AVAILABLE:
        print("环境：未安装 PyTorch，已用 NumPy 演示 STE 原理（实验 A）。")
        print("      安装 torch 后可运行完整 PTQ vs QAT 对比（实验 B）。")
        print()
        print(SEP)
        print("环境：未安装 PyTorch，已用 NumPy 演示核心原理。")
        print(SEP)
        return

    in_dim, hidden, n_classes = 8, 64, 3
    n_bits = 4   # INT4：低比特，量化误差大，凸显 QAT 优势
    torch.manual_seed(0)
    X_tr, y_tr = make_dataset(3000, in_dim, n_classes, seed=1)
    X_te, y_te = make_dataset(1000, in_dim, n_classes, seed=3)
    X_tr, y_tr = X_tr.to(DEVICE), y_tr.to(DEVICE)
    X_te, y_te = X_te.to(DEVICE), y_te.to(DEVICE)

    # ---- 1) FP32 训练基准（关闭量化）----
    model = QuantMLP(in_dim, hidden, n_classes, n_bits).to(DEVICE)
    model.set_quant(False)
    train(model, X_tr, y_tr, epochs=250)
    acc_fp32 = evaluate(model, X_te, y_te)

    # ---- 2) PTQ：直接打开量化评估（不再训练）----
    model.set_quant(True)
    acc_ptq = evaluate(model, X_te, y_te)

    # ---- 3) QAT：在打开量化的状态下继续微调 ----
    #    此时前向带伪量化，STE 让梯度回流，模型学习对量化鲁棒的权重
    train(model, X_tr, y_tr, epochs=150, lr=0.005)
    acc_qat = evaluate(model, X_te, y_te)

    print(f"  量化位宽：INT{n_bits}（qmax={2**(n_bits-1)-1}）")
    print(f"  [FP32 基准]           准确率 = {acc_fp32:.3%}")
    print(f"  [PTQ  直接量化]       准确率 = {acc_ptq:.3%}  (较FP32 {acc_fp32-acc_ptq:+.3%})")
    print(f"  [QAT  量化感知微调]   准确率 = {acc_qat:.3%}  (较FP32 {acc_fp32-acc_qat:+.3%})")
    print()
    print(f"  [结论] QAT 相比 PTQ 恢复了 {acc_qat - acc_ptq:+.3%} 的精度")
    ok = acc_qat >= acc_ptq
    print("  [结果] " + ("通过 ✓（QAT 精度 ≥ PTQ，量化感知训练有效）"
                        if ok else "本次随机种子下差距不明显（可增大 epoch/降低比特）"))

    print()
    print(SEP)
    if DEVICE == "cuda":
        print("环境：CUDA GPU 就绪，QAT 伪量化训练在 GPU 上真实运行。")
    else:
        print("环境：无 CUDA GPU，已在 CPU 上真实运行完整 QAT 流程。")
    print(SEP)


if __name__ == "__main__":
    main()
