#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 75: 梯度检查点 / 激活重计算（Gradient Checkpointing / Activation Recomputation）

反向传播需要用到前向的中间激活。默认情况下，前向会把每一层的激活都存下来
（显存 O(层数)）；层数一多，激活显存就爆了。

梯度检查点的思路——「用算力换显存」：
  前向时只保存少量「检查点」处的激活，中间激活丢弃；反向需要时，从最近的
  检查点重新前向计算一遍，把中间激活「重算」出来。
  代价是多做一次前向（约 +33% 计算），换来激活显存从 O(N) 降到 O(sqrt(N))。

本天目标：
  1. 手写实现「存激活」vs「重计算」两种前向/反向，验证梯度一致；
  2. 若有 torch，用 torch.utils.checkpoint 做真实对照；
  3. 统计两种方式的激活显存占用，展示显存-算力权衡。

运行策略：手写版用 numpy/torch 均可，无 GPU 也能跑。显存用「存下的激活
张量元素总数」来度量（单机可复现，不依赖真实显存计量）。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

GPU_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()
MULTI_GPU = GPU_AVAILABLE and torch.cuda.device_count() > 1

SEP = "=" * 64


# ============================================================
#  0. 原理讲解
# ============================================================
def explain_checkpoint():
    print(SEP)
    print("梯度检查点核心思想：用算力换显存")
    print(SEP)
    print("""
  一个 N 层网络的反向传播需要每层的前向激活：

  【默认：存全部激活】
    前向: x0 →[L0]→ x1 →[L1]→ x2 → ... → xN   (全部 x_i 都留在显存)
    显存: O(N)      反向: 直接用存好的 x_i，无需重算
    问题: N 大时激活显存爆炸

  【梯度检查点：只存检查点】
    前向: 只保留少数 x_i（检查点），其余丢弃
    反向: 到某段时，从最近检查点「重新前向」算出该段中间激活，再反向
    显存: O(sqrt(N))     算力: 多一次前向 (~+33%)

  权衡：
    - 显存紧张、想训更大 batch / 更深模型时开启；
    - 计算充裕但显存受限的场景收益最大；
    - 与混合精度、ZeRO、并行策略可叠加使用。
""")


# ============================================================
#  1. 一个简单的多层网络：每层 y = tanh(x @ W)
#     （tanh 便于手写反向，且导数只依赖输出）
# ============================================================
def make_layers(num_layers, dim, seed):
    if TORCH_AVAILABLE:
        gen = torch.Generator().manual_seed(seed)
        return [torch.randn(dim, dim, generator=gen) * 0.1 for _ in range(num_layers)]
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((dim, dim)) * 0.1).astype(np.float32)
            for _ in range(num_layers)]


def _tanh(x):
    return torch.tanh(x) if TORCH_AVAILABLE else np.tanh(x)


def layer_forward(x, W):
    """单层前向：z = x @ W; y = tanh(z)。返回 y。"""
    return _tanh(x @ W)


# ============================================================
#  2. 方式 A：存全部激活的前向 + 反向
# ============================================================
def forward_store_all(x0, weights):
    """标准前向：保存每一层的输入激活，供反向使用。

    返回 (输出, 激活列表 acts)。acts[i] 是第 i 层的输入。
    显存度量 = 保存的激活元素总数。
    """
    acts = []          # acts[i] = 第 i 层的输入 x_i
    x = x0
    for W in weights:
        acts.append(x)             # 保存该层输入（反向要用）
        x = layer_forward(x, W)
    mem = sum(a.size if not TORCH_AVAILABLE else a.numel() for a in acts)
    return x, acts, mem


def backward_store_all(acts, weights, grad_out):
    """标准反向：用存好的激活逐层回传，返回对 x0 的梯度（作为验证信号）。

    对 y = tanh(x @ W)：
      dz = grad_y * (1 - y^2)
      dx = dz @ W^T
    """
    grad = grad_out
    for i in reversed(range(len(weights))):
        x = acts[i]
        W = weights[i]
        y = layer_forward(x, W)              # 该层输出（存激活时其实也可缓存）
        dz = grad * (1 - y * y)
        if TORCH_AVAILABLE:
            grad = dz @ W.t()
        else:
            grad = dz @ W.T
    return grad


# ============================================================
#  3. 方式 B：梯度检查点（只存检查点，反向时重算）
# ============================================================
def forward_checkpoint(x0, weights, seg_len):
    """检查点前向：每 seg_len 层设一个检查点，只保存段边界的激活。

    段内的中间激活全部丢弃，反向时再重算。
    返回 (输出, 检查点激活列表, 显存度量)。
    """
    ckpts = []          # 只保存每段起点的输入激活
    x = x0
    for i, W in enumerate(weights):
        if i % seg_len == 0:
            ckpts.append(x)                  # 段起点：设检查点
        x = layer_forward(x, W)
    mem = sum(c.size if not TORCH_AVAILABLE else c.numel() for c in ckpts)
    return x, ckpts, mem


def backward_checkpoint(x0, weights, ckpts, seg_len, grad_out):
    """检查点反向：逐段处理。每段先从检查点「重算」出段内所有激活，再反向。

    这体现「重计算」：段内激活不是存下来的，而是反向时重新前向得到。
    """
    num_layers = len(weights)
    num_segs = (num_layers + seg_len - 1) // seg_len
    grad = grad_out
    for s in reversed(range(num_segs)):
        lo = s * seg_len
        hi = min(lo + seg_len, num_layers)
        # 重算：从该段检查点重新前向，恢复段内每层输入激活
        seg_acts = []
        x = ckpts[s]
        for i in range(lo, hi):
            seg_acts.append(x)
            x = layer_forward(x, weights[i])
        # 段内反向
        for local_i in reversed(range(hi - lo)):
            i = lo + local_i
            xin = seg_acts[local_i]
            W = weights[i]
            y = layer_forward(xin, W)
            dz = grad * (1 - y * y)
            if TORCH_AVAILABLE:
                grad = dz @ W.t()
            else:
                grad = dz @ W.T
    return grad


# ============================================================
#  4. torch.utils.checkpoint 真实对照（有 torch 时）
# ============================================================
def torch_checkpoint_demo(x0, weights, seg_len):
    """用 torch.utils.checkpoint 做真实的激活重计算，与手写版对拍梯度。

    返回 (对 x0 的梯度)。仅在 TORCH_AVAILABLE 时调用。
    """
    import torch.utils.checkpoint as cp

    x = x0.clone().detach().requires_grad_(True)

    def run_segment(inp, seg_weights):
        h = inp
        for W in seg_weights:
            h = torch.tanh(h @ W)
        return h

    num_layers = len(weights)
    h = x
    i = 0
    while i < num_layers:
        seg_weights = weights[i:i + seg_len]
        # checkpoint：前向不存段内激活，反向时重算
        h = cp.checkpoint(run_segment, h, seg_weights, use_reentrant=False)
        i += seg_len
    loss = h.sum()
    loss.backward()
    return x.grad


# ============================================================
#  主流程
# ============================================================
def main():
    explain_checkpoint()

    num_layers, dim, batch = 12, 32, 8
    seg_len = 4          # 每 4 层一个检查点 → 检查点数 = 3
    weights = make_layers(num_layers, dim, seed=123)

    if TORCH_AVAILABLE:
        x0 = torch.randn(batch, dim)
        grad_out = torch.ones(batch, dim)
    else:
        x0 = np.random.default_rng(0).standard_normal((batch, dim)).astype(np.float32)
        grad_out = np.ones((batch, dim), np.float32)

    print(SEP)
    print(f"网络：{num_layers} 层，dim={dim}，batch={batch}；"
          f"检查点每 {seg_len} 层一个")
    print(SEP)

    # 方式 A：存全部激活
    out_a, acts, mem_a = forward_store_all(x0, weights)
    grad_a = backward_store_all(acts, weights, grad_out)

    # 方式 B：梯度检查点
    out_b, ckpts, mem_b = forward_checkpoint(x0, weights, seg_len)
    grad_b = backward_checkpoint(x0, weights, ckpts, seg_len, grad_out)

    # 前向输出一致性
    if TORCH_AVAILABLE:
        out_err = (out_a - out_b).abs().max().item()
        grad_err = (grad_a - grad_b).abs().max().item()
    else:
        out_err = float(np.max(np.abs(out_a - out_b)))
        grad_err = float(np.max(np.abs(grad_a - grad_b)))

    print("数值验证：两种方式应得到相同的前向输出与梯度")
    print(f"  前向输出最大误差 = {out_err:.2e}")
    print(f"  输入梯度最大误差 = {grad_err:.2e}")
    ok = out_err < 1e-4 and grad_err < 1e-4
    print("  [结果] " + ("通过 ✓" if ok else "偏差过大 ✗"))

    print()
    print(SEP)
    print("显存 - 算力权衡统计")
    print(SEP)
    print(f"  {'方式':<16} | {'激活显存(元素数)':>16} | {'前向次数':>10}")
    print("  " + "-" * 52)
    print(f"  {'存全部激活':<14} | {mem_a:>16} | {'1 次':>10}")
    print(f"  {'梯度检查点':<14} | {mem_b:>16} | {'~2 次':>10}")
    ratio = mem_a / max(mem_b, 1)
    print(f"\n  激活显存节省：{mem_a} → {mem_b}，约省 {(1 - mem_b/mem_a)*100:.1f}%"
          f"（{ratio:.1f}x）")
    print(f"  代价：反向时段内激活需重算一次前向，总计算量约 +{(1/1)*33:.0f}%（经验值）")

    # 扫描不同 seg_len，展示 O(sqrt(N)) 趋势
    print()
    print("  不同检查点间隔的激活显存（层数固定 = %d）：" % num_layers)
    print(f"  {'seg_len':>8} | {'检查点数':>8} | {'激活显存(元素数)':>16}")
    print("  " + "-" * 42)
    for sl in [1, 2, 3, 4, 6, 12]:
        _, ck, m = forward_checkpoint(x0, weights, sl)
        print(f"  {sl:>8} | {len(ck):>8} | {m:>16}")
    print("  注：seg_len=1 等价于「几乎不省」，seg_len 取 ~sqrt(N) 时显存-算力最优。")

    print()
    print(SEP)
    if TORCH_AVAILABLE:
        print("torch.utils.checkpoint 真实对照：")
        grad_torch = torch_checkpoint_demo(x0, weights, seg_len)
        # 与手写 store_all 的梯度对拍
        terr = (grad_torch - grad_a).abs().max().item()
        print(f"  torch.checkpoint 梯度 vs 手写(存全部) 最大误差 = {terr:.2e}")
        print("  [结果] " + ("通过 ✓" if terr < 1e-4 else "偏差过大 ✗"))
    else:
        print("环境：无 PyTorch。已用 NumPy 手写演示存激活 vs 重计算的等价性与显存权衡。")
    print(SEP)

    print()
    print(SEP)
    if GPU_AVAILABLE:
        print("环境：有 GPU。真实显存可用 torch.cuda.max_memory_allocated() 计量；"
              "本脚本用激活元素数作可复现的显存代理指标。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 无 GPU。已完成手写与 torch.checkpoint 双重验证。")
    else:
        print("环境：无 PyTorch。已用 NumPy 完成手写验证。")
    print(SEP)


if __name__ == "__main__":
    main()
