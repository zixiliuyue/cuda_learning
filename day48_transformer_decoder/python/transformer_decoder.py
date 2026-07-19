#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 48: 从零搭一个 Transformer Decoder 层

用 PyTorch 手写一个 GPT 风格的 Decoder 层前向，覆盖 LLM 推理的最小构成单元：
  1. Multi-Head Self-Attention（含 causal mask，逐 token 只能看到自己和前面）
  2. Position-wise Feed-Forward Network（FFN，两层线性 + 激活）
  3. 残差连接（residual）与 LayerNorm（这里用 Pre-LN，主流 LLM 的做法）

核心目的：把 Day31/44 学过的注意力，组装成真正的 Transformer Decoder 层，
理解 “一个 LLM 就是 N 个这种层堆起来 + embedding + lm_head”。

运行策略（能真跑就真跑）：
  - 有 CUDA GPU：device=cuda，真实前向 + 显存/耗时统计；
  - 无 CUDA（纯 CPU）：device=cpu，逻辑完全一样照跑，只是不做 GPU 专属统计；
  - 连 torch 都没有：用 NumPy 演示同样的注意力核心公式，保证也能看懂原理。
"""

import math

SEP = "=" * 64

# ------------------------------------------------------------
# 环境探测：PyTorch 在 CPU 也能跑，因此 torch 在就用 torch
# ------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

import numpy as np


# ============================================================
#  PyTorch 实现：手写 Multi-Head Attention
# ============================================================
if TORCH_AVAILABLE:

    class MultiHeadSelfAttention(nn.Module):
        """手写多头自注意力（不用 nn.MultiheadAttention，看清每一步）。

        形状约定：
          x: (B, T, D)   B=batch, T=序列长度, D=模型维度 d_model
          切成 n_head 个头，每个头维度 d_head = D / n_head
        """

        def __init__(self, d_model: int, n_head: int):
            super().__init__()
            assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
            self.d_model = d_model
            self.n_head = n_head
            self.d_head = d_model // n_head
            # 一次性算出 Q/K/V，比三个独立 Linear 更省，qkv 输出 3*d_model
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)  # 输出投影

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            B, T, D = x.shape
            # (B,T,3D) -> 3 * (B,T,D)
            q, k, v = self.qkv(x).split(self.d_model, dim=2)
            # (B,T,D) -> (B, n_head, T, d_head)：把头拆出来并前移，便于批量矩阵乘
            q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

            # 注意力分数：QK^T / sqrt(d_head)  -> (B, n_head, T, T)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

            # causal mask：第 i 个 token 只能看到 <= i 的位置，上三角设为 -inf
            causal = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(causal == 0, float("-inf"))

            att = F.softmax(att, dim=-1)          # 每行归一化成概率
            out = att @ v                          # 加权求和 -> (B, n_head, T, d_head)

            # 合并多头：(B, n_head, T, d_head) -> (B, T, D)
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            return self.proj(out)

    class FeedForward(nn.Module):
        """位置无关的前馈网络：Linear -> GELU -> Linear，中间维度一般是 4*d_model。"""

        def __init__(self, d_model: int, d_ff: int):
            super().__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.fc2(F.gelu(self.fc1(x)))

    class DecoderLayer(nn.Module):
        """一个 GPT 风格 Decoder 层（Pre-LN 结构）。

        Pre-LN 前向（主流 LLM 采用，训练更稳定）：
            x = x + Attn(LN(x))
            x = x + FFN(LN(x))
        """

        def __init__(self, d_model: int, n_head: int, d_ff: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = MultiHeadSelfAttention(d_model, n_head)
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = FeedForward(d_model, d_ff)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.attn(self.ln1(x))   # 残差 + 注意力子层
            x = x + self.ffn(self.ln2(x))    # 残差 + FFN 子层
            return x


# ============================================================
#  无 torch 时：NumPy 演示注意力核心公式（causal + softmax）
# ============================================================
def numpy_causal_attention(x, Wq, Wk, Wv):
    """单头 causal self-attention 的 NumPy 版，讲清最核心的公式。

    x: (T, D)   Wq/Wk/Wv: (D, D)
    """
    T, D = x.shape
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    att = (q @ k.T) / math.sqrt(D)             # (T, T)
    mask = np.triu(np.ones((T, T)), k=1).astype(bool)  # 上三角(不含对角) 为 True
    att = np.where(mask, -np.inf, att)         # 未来位置置 -inf
    att = att - att.max(axis=-1, keepdims=True)
    e = np.exp(att)
    att = e / e.sum(axis=-1, keepdims=True)    # 逐行 softmax
    return att @ v                             # (T, D)


# ============================================================
#  讲解：Decoder 层结构
# ============================================================
def explain_decoder():
    print(SEP)
    print("Transformer Decoder 层结构（GPT 风格，Pre-LN）")
    print(SEP)
    print("""
  输入 x (B,T,D)
     │
     ├─────────────┐  残差直连
     │        ┌────▼────┐
     │        │ LayerNorm│
     │        └────┬────┘
     │      ┌──────▼───────┐
     │      │ Multi-Head    │  Q,K,V = xW；QK^T/√d_head
     │      │ Self-Attention│  + causal mask（看不到未来）
     │      │  + softmax    │  → 加权求和 V → 输出投影
     │      └──────┬───────┘
     ▼◄────────────┘  x = x + Attn(LN(x))
     │
     ├─────────────┐  残差直连
     │        ┌────▼────┐
     │        │ LayerNorm│
     │        └────┬────┘
     │      ┌──────▼───────┐
     │      │ FFN: Linear   │  d_model → 4*d_model
     │      │  → GELU       │  → d_model
     │      │  → Linear     │
     │      └──────┬───────┘
     ▼◄────────────┘  x = x + FFN(LN(x))
   输出 (B,T,D)

  一个完整 LLM ≈ Embedding + N × [DecoderLayer] + 最终 LN + lm_head。
  本天只搭“一层”，Day49 起在其上加 KV Cache、分页、动态组批。
""")


# ============================================================
#  主流程
# ============================================================
def run_torch():
    torch.manual_seed(0)
    B, T, D, n_head, d_ff = 2, 16, 128, 8, 512
    print(f"[环境] 使用 device = {DEVICE}")
    print(f"[配置] batch={B}, seq_len={T}, d_model={D}, n_head={n_head}, d_ff={d_ff}")

    layer = DecoderLayer(D, n_head, d_ff).to(DEVICE)
    x = torch.randn(B, T, D, device=DEVICE)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = layer(x)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"[GPU 统计] 前向耗时 = {ms:.3f} ms, 峰值显存 = {peak_mb:.2f} MB")
    else:
        y = layer(x)
        print("[CPU 说明] 逻辑与 GPU 完全一致，仅跳过 GPU 专属计时/显存统计。")

    print(f"[形状检查] 输入 {tuple(x.shape)} -> 输出 {tuple(y.shape)}  "
          + ("(一致 ✓)" if x.shape == y.shape else "(不一致 ✗)"))
    print(f"[数值检查] 输出 mean={y.mean().item():.4f}, std={y.std().item():.4f}, "
          f"含 NaN={bool(torch.isnan(y).any())}")

    # 验证 causal 性质：改动最后一个 token 的输入，不应影响前面 token 的输出
    x2 = x.clone()
    x2[:, -1, :] += 10.0
    with torch.no_grad():
        y2 = layer(x2)
    diff_prefix = (y2[:, :-1, :] - y[:, :-1, :]).abs().max().item()
    print(f"[因果性验证] 改末位 token 后，前缀输出最大变化 = {diff_prefix:.2e} "
          + ("(≈0，符合因果 ✓)" if diff_prefix < 1e-4 else "(异常 ✗)"))


def run_numpy():
    print(f"[环境] 未安装 PyTorch，用 NumPy 演示 causal 注意力核心公式 (device={DEVICE})")
    np.random.seed(0)
    T, D = 6, 8
    x = np.random.randn(T, D).astype(np.float32)
    Wq = np.random.randn(D, D).astype(np.float32) * 0.1
    Wk = np.random.randn(D, D).astype(np.float32) * 0.1
    Wv = np.random.randn(D, D).astype(np.float32) * 0.1
    out = numpy_causal_attention(x, Wq, Wk, Wv)
    print(f"[形状检查] 输入 {x.shape} -> 注意力输出 {out.shape}")

    # 因果性：第 0 个 token 的输出只应依赖它自己
    x2 = x.copy()
    x2[-1] += 10.0
    out2 = numpy_causal_attention(x2, Wq, Wk, Wv)
    diff_prefix = np.abs(out2[:-1] - out[:-1]).max()
    print(f"[因果性验证] 改末位 token 后，前缀输出最大变化 = {diff_prefix:.2e} "
          + ("(≈0，符合因果 ✓)" if diff_prefix < 1e-4 else "(异常 ✗)"))
    print("       完整多头 + FFN + 残差 + LayerNorm 版见 DecoderLayer（需安装 torch）。")


def main():
    explain_decoder()
    print(SEP)
    print("运行 Decoder 层前向")
    print(SEP)
    if TORCH_AVAILABLE:
        run_torch()
    else:
        run_numpy()

    print()
    print(SEP)
    if TORCH_AVAILABLE and DEVICE == "cuda":
        print("环境：PyTorch + CUDA GPU 就绪，已在 GPU 上运行真实前向并统计显存/耗时。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU。已在 CPU 上运行完整前向（逻辑一致）。")
    else:
        print("环境：未安装 PyTorch。已用 NumPy 演示注意力核心公式。")
        print("提示：`pip install torch` 后可运行完整多头 Decoder 层。")
    print(SEP)


if __name__ == "__main__":
    main()
