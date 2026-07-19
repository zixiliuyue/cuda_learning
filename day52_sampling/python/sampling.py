#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 52: 采样与解码策略 — greedy / temperature / top-k / top-p(nucleus)

LLM 推理时，模型每一步输出的是词表上的一个 logits 向量（未归一化分数）。
「怎么从这个分布里挑下一个 token」就是**解码策略**，它直接决定生成文本的
质量、多样性与稳定性。本天实现并对比四种最常见的策略：

  1. greedy      : 永远取概率最大的 token（确定性，易重复）
  2. temperature : 用温度 T 缩放 logits，控制分布的“尖锐/平缓”
  3. top-k       : 只在概率最高的 k 个 token 里采样
  4. top-p       : 只在累积概率达到 p 的最小 token 集合（nucleus）里采样

运行策略（有 CUDA 用 GPU，无 CUDA 用 CPU，都无 torch 则用 NumPy）：
  - device = "cuda" if torch.cuda.is_available() else "cpu"
  - 无论 GPU/CPU，完整逻辑都真实运行；仅在完全没有 torch 时退化为 NumPy 版。
"""

import math

# ------------------------------------------------------------
# 环境探测（有 CUDA 用 GPU，无 CUDA 用 CPU，都真实运行）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

import numpy as np

SEP = "=" * 64


# ============================================================
#  PyTorch 版：四种采样策略
# ============================================================
if TORCH_AVAILABLE:

    def softmax_probs(logits: "torch.Tensor") -> "torch.Tensor":
        """把 logits 转成概率分布（数值稳定：减最大值）。"""
        return torch.softmax(logits, dim=-1)

    def greedy_decode(logits: "torch.Tensor") -> int:
        """贪心解码：直接取 argmax，完全确定性。"""
        return int(torch.argmax(logits).item())

    def temperature_sample(logits: "torch.Tensor", temperature: float,
                           generator=None) -> int:
        """温度采样：logits / T 后再 softmax 采样。

        T < 1 使分布更尖锐（更接近 greedy）；T > 1 使分布更平缓（更随机）。
        """
        scaled = logits / max(temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, num_samples=1, generator=generator).item())

    def top_k_sample(logits: "torch.Tensor", k: int, temperature: float = 1.0,
                     generator=None) -> int:
        """top-k 采样：只保留概率最高的 k 个 token，其余置零后重新归一化。"""
        scaled = logits / max(temperature, 1e-6)
        k = min(k, scaled.numel())
        topk_vals, topk_idx = torch.topk(scaled, k)
        probs = torch.softmax(topk_vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=generator)
        return int(topk_idx[choice].item())

    def top_p_sample(logits: "torch.Tensor", p: float, temperature: float = 1.0,
                     generator=None) -> int:
        """top-p / nucleus 采样：取累积概率刚好达到 p 的最小 token 集合。"""
        scaled = logits / max(temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # 保留累积概率 <= p 的部分；至少保留第一个 token
        mask = cumulative <= p
        mask[0] = True
        kept_probs = sorted_probs[mask]
        kept_idx = sorted_idx[mask]
        kept_probs = kept_probs / kept_probs.sum()
        choice = torch.multinomial(kept_probs, num_samples=1, generator=generator)
        return int(kept_idx[choice].item())


# ============================================================
#  NumPy 版：完全无 torch 时的等价实现（原理一致）
# ============================================================
def np_softmax(logits, temperature=1.0):
    x = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def np_greedy(logits):
    return int(np.argmax(logits))


def np_temperature_sample(logits, temperature, rng):
    probs = np_softmax(logits, temperature)
    return int(rng.choice(len(probs), p=probs))


def np_top_k_sample(logits, k, temperature, rng):
    logits = np.asarray(logits, dtype=np.float64)
    k = min(k, len(logits))
    topk_idx = np.argpartition(-logits, k - 1)[:k]
    probs = np_softmax(logits[topk_idx], temperature)
    return int(topk_idx[rng.choice(len(topk_idx), p=probs)])


def np_top_p_sample(logits, p, temperature, rng):
    probs = np_softmax(logits, temperature)
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    mask = cumulative <= p
    mask[0] = True
    kept_idx = order[mask]
    kept_probs = probs[kept_idx]
    kept_probs = kept_probs / kept_probs.sum()
    return int(kept_idx[rng.choice(len(kept_idx), p=kept_probs)])


# ============================================================
#  策略讲解
# ============================================================
def explain_strategies():
    print(SEP)
    print("解码策略总览（LLM 每步都在做的选择）")
    print(SEP)
    print("""
  策略          做法                              特点
  ------------------------------------------------------------
  greedy        取 argmax                         确定性，最保守，易重复
  temperature   logits/T 再 softmax 采样          T↑更随机 / T↓更保守
  top-k         只在概率前 k 个里采样             截断长尾，控制风险
  top-p         在累积概率达 p 的最小集合里采样   自适应候选数（nucleus）

  关键直觉：
   - temperature 调「分布形状」，top-k/top-p 调「候选范围」，常配合使用。
   - top-p 比 top-k 更自适应：分布尖锐时候选少，分布平缓时候选多。
""")


def summarize_distribution(name, samples, vocab_size):
    """统计一批采样结果落在各 token 上的频次。"""
    counts = np.bincount(np.asarray(samples), minlength=vocab_size)
    top = np.argsort(-counts)[:5]
    frags = [f"t{int(i)}:{counts[i]}" for i in top if counts[i] > 0]
    unique = int(np.count_nonzero(counts))
    print(f"  {name:<24} 命中不同token数={unique:<3} 前5高频=[{', '.join(frags)}]")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_strategies()

    # 构造一个固定的 logits 分布：词表 10，制造一个明显的“主峰 + 长尾”
    vocab_size = 10
    base_logits = [6.0, 5.2, 4.5, 3.0, 2.5, 1.0, 0.5, 0.2, -1.0, -2.0]
    n_samples = 2000

    print(SEP)
    print(f"实验设置：vocab_size={vocab_size}, 每种随机策略采样 {n_samples} 次")
    print(f"运行设备：DEVICE = {DEVICE}"
          + ("（真实 GPU）" if DEVICE == "cuda" else "（CPU，逻辑完整）"))
    print(SEP)
    probs_preview = (np_softmax(base_logits)).round(4)
    print("原始 logits :", base_logits)
    print("对应概率     :", list(probs_preview))
    print()

    if TORCH_AVAILABLE:
        logits = torch.tensor(base_logits, dtype=torch.float32, device=DEVICE)
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(0)

        # greedy（确定性，采一次即可）
        g = greedy_decode(logits)
        print(f"  {'greedy':<24} 恒定选择 = t{g}")

        # temperature 对比
        for T in (0.5, 1.0, 1.5):
            samples = [temperature_sample(logits, T, gen) for _ in range(n_samples)]
            summarize_distribution(f"temperature(T={T})", samples, vocab_size)

        # top-k 对比
        for k in (1, 3, 5):
            samples = [top_k_sample(logits, k, 1.0, gen) for _ in range(n_samples)]
            summarize_distribution(f"top-k(k={k})", samples, vocab_size)

        # top-p 对比
        for p in (0.5, 0.8, 0.95):
            samples = [top_p_sample(logits, p, 1.0, gen) for _ in range(n_samples)]
            summarize_distribution(f"top-p(p={p})", samples, vocab_size)

    else:
        rng = np.random.default_rng(0)
        g = np_greedy(base_logits)
        print(f"  {'greedy':<24} 恒定选择 = t{g}")
        for T in (0.5, 1.0, 1.5):
            samples = [np_temperature_sample(base_logits, T, rng) for _ in range(n_samples)]
            summarize_distribution(f"temperature(T={T})", samples, vocab_size)
        for k in (1, 3, 5):
            samples = [np_top_k_sample(base_logits, k, 1.0, rng) for _ in range(n_samples)]
            summarize_distribution(f"top-k(k={k})", samples, vocab_size)
        for p in (0.5, 0.8, 0.95):
            samples = [np_top_p_sample(base_logits, p, 1.0, rng) for _ in range(n_samples)]
            summarize_distribution(f"top-p(p={p})", samples, vocab_size)

    print()
    print(SEP)
    print("观察要点：")
    print("  - greedy 永远是同一个 token（这里是 t0），最稳定但最单调。")
    print("  - temperature 越大，命中的不同 token 越多，分布越平。")
    print("  - top-k 的 k 越大，候选越多；k=1 等价于 greedy。")
    print("  - top-p 会自适应地选择候选集合大小，长尾被自然截断。")
    print(SEP)
    if DEVICE == "cuda":
        print("环境：CUDA GPU 就绪，采样在 GPU 上真实运行。")
    elif TORCH_AVAILABLE:
        print("环境：无 CUDA GPU，已在 CPU 上真实运行完整采样逻辑。")
    else:
        print("环境：未安装 PyTorch，已用 NumPy 演示等价原理。")
    print(SEP)


if __name__ == "__main__":
    main()
