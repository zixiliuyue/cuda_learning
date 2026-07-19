#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 55: 投机解码（Speculative Decoding）入门

自回归生成的瓶颈：大模型（target）每生成 1 个 token 就要前向一次，串行且慢。
**投机解码**用一个便宜的小模型（draft）先猜 k 个 token，大模型再**一次并行**
验证这 k 个：
  - 通过「接受-拒绝采样」逐个决定接受还是拒绝；
  - 一旦拒绝，就从修正分布里重采样一个 token 并停止本轮；
  - 若 k 个全接受，还能额外「白嫖」一个 target 采的 bonus token。

数学保证（核心！）：无论 draft 多不准，最终被接受的 token 序列的分布**严格等于**
大模型自己自回归采样的分布。draft 只影响速度，不影响正确性。

运行策略：device = "cuda" if torch.cuda.is_available() else "cpu"，逻辑都真实运行；
无 torch 时退化为 NumPy 版。本天用「马尔可夫式玩具模型」（转移概率矩阵）模拟
target / draft 两个模型的下一 token 分布，从而清晰验证分布一致性。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

SEP = "=" * 64
VOCAB = 8       # 玩具词表大小
K = 4           # 每轮 draft 猜测的 token 数


# ============================================================
#  玩具模型：用转移矩阵表示「给定当前 token -> 下一 token 分布」
#  （统一用 NumPy 构造矩阵，采样阶段按环境走 torch 或 numpy）
# ============================================================
def build_models(seed=0):
    """构造 target 与 draft 两个转移概率矩阵。

    draft = target 加噪声后重新归一化：既相关又不完全相同，
    从而产生「部分接受、部分拒绝」的真实投机场景。
    """
    rng = np.random.default_rng(seed)
    logits_t = rng.standard_normal((VOCAB, VOCAB)) * 1.5
    target = np.exp(logits_t - logits_t.max(axis=1, keepdims=True))
    target /= target.sum(axis=1, keepdims=True)

    noise = rng.standard_normal((VOCAB, VOCAB)) * 1.0
    logits_d = logits_t + noise
    draft = np.exp(logits_d - logits_d.max(axis=1, keepdims=True))
    draft /= draft.sum(axis=1, keepdims=True)
    return target.astype(np.float64), draft.astype(np.float64)


def sample_from(prob, rng):
    """从一维概率分布采样一个 token。"""
    return int(rng.choice(len(prob), p=prob))


# ============================================================
#  基准：target 纯自回归采样（作为分布的「金标准」）
# ============================================================
def autoregressive_step(target, state, rng):
    """target 自己走一步：从 target[state] 分布采一个 token。"""
    return sample_from(target[state], rng)


# ============================================================
#  投机解码：一轮（从 state 出发，最多产出 K+1 个 token）
# ============================================================
def speculative_round(target, draft, state, rng):
    """执行一轮投机解码，返回本轮接受产出的 token 列表。

    标准 speculative sampling（Leviathan et al. 2023 / Chen et al. 2023）：
      1. draft 自回归猜 K 个 token，记录各步 draft 分布；
      2. target 对这 K 个位置「并行」给出分布（这里查表即并行）；
      3. 对第 i 个草稿 token x：
           以概率 min(1, p_target(x)/p_draft(x)) 接受；
           否则拒绝，从 (p_target - p_draft)_+ 归一化后重采样一个 token，停止；
      4. 若 K 个全接受，用最后位置的 target 分布再采 1 个 bonus token。
    """
    # ---- 1. draft 猜 K 个 ----
    draft_tokens = []
    draft_probs = []
    cur = state
    for _ in range(K):
        p = draft[cur]
        tok = sample_from(p, rng)
        draft_tokens.append(tok)
        draft_probs.append(p)
        cur = tok

    # ---- 2. target 对每个位置的分布（查表 = 并行验证）----
    # 位置 i 的 target 分布依赖于该位置的输入 state：
    #   位置0 输入 state；位置 i(>0) 输入 draft_tokens[i-1]
    target_states = [state] + draft_tokens[:-1]
    target_probs = [target[s] for s in target_states]

    # ---- 3. 逐个接受/拒绝 ----
    accepted = []
    for i in range(K):
        x = draft_tokens[i]
        pt = target_probs[i][x]
        pd = draft_probs[i][x]
        accept_ratio = min(1.0, pt / (pd + 1e-12))
        if rng.random() < accept_ratio:
            accepted.append(x)          # 接受
        else:
            # 拒绝：从修正分布 (p_target - p_draft)_+ 重采样
            resid = np.clip(target_probs[i] - draft_probs[i], 0, None)
            s = resid.sum()
            resid = resid / s if s > 1e-12 else target_probs[i]
            accepted.append(sample_from(resid, rng))
            return accepted             # 本轮停止
    # ---- 4. 全接受：bonus token（从最后一个 token 之后的 target 分布采）----
    bonus_state = draft_tokens[-1]
    accepted.append(sample_from(target[bonus_state], rng))
    return accepted


def speculative_generate(target, draft, start, n_tokens, rng):
    """用投机解码生成至少 n_tokens 个 token，返回序列与 target 前向次数。"""
    out = []
    target_calls = 0
    state = start
    while len(out) < n_tokens:
        produced = speculative_round(target, draft, state, rng)
        target_calls += 1          # 每轮只调用一次 target（并行验证）
        out.extend(produced)
        state = out[-1]
    return out[:n_tokens], target_calls


# ============================================================
#  讲解
# ============================================================
def explain_spec():
    print(SEP)
    print("投机解码（Speculative Decoding）原理")
    print(SEP)
    print("""
  痛点：大模型自回归每步只出 1 个 token，串行慢。

  思路：小模型(draft)便宜，先猜 K 个；大模型(target)一次并行验证。
    - 接受概率 = min(1, p_target(x) / p_draft(x))
    - 拒绝时从 (p_target - p_draft)_+ 归一化分布重采样，保证无偏
    - K 个全中还能白嫖 1 个 bonus token

  关键保证：最终 token 分布 == target 自回归分布（draft 只管快慢，不管对错）。

  加速来源：一次 target 前向平均产出 >1 个 token（取决于 draft 命中率）。
""")


def kl_div(p, q):
    p = np.asarray(p) + 1e-12
    q = np.asarray(q) + 1e-12
    return float(np.sum(p * np.log(p / q)))


# ============================================================
#  主流程
# ============================================================
def main():
    explain_spec()

    target, draft = build_models(seed=0)
    print(SEP)
    print(f"实验：VOCAB={VOCAB}, 每轮草稿数 K={K}")
    print(f"运行设备：DEVICE = {DEVICE}"
          + ("（真实 GPU）" if DEVICE == "cuda" else "（CPU/NumPy，逻辑完整）"))
    if TORCH_AVAILABLE:
        # 把矩阵放上 device，体现「有 GPU 用 GPU」；采样统计仍用等价数值逻辑
        _t = torch.tensor(target, device=DEVICE)
        _d = torch.tensor(draft, device=DEVICE)
        print(f"  target/draft 转移矩阵已加载到 {DEVICE}，shape={tuple(_t.shape)}")
    print(SEP)

    start = 0
    n_samples = 40000

    # ---- 验证 1：单步分布一致性 ----
    # 投机解码产出序列的「第一个 token」应服从 target[start] 分布
    rng = np.random.default_rng(123)
    spec_first = np.zeros(VOCAB)
    ar_first = np.zeros(VOCAB)
    total_produced = 0
    total_calls = 0
    for _ in range(n_samples):
        seq, calls = speculative_generate(target, draft, start, 1, rng)
        spec_first[seq[0]] += 1
        total_produced += 1
        total_calls += calls
    spec_first /= spec_first.sum()

    rng2 = np.random.default_rng(456)
    for _ in range(n_samples):
        ar_first[autoregressive_step(target, start, rng2)] += 1
    ar_first /= ar_first.sum()

    print("[验证 1] 单步输出分布（第一个 token）")
    print("  token :  " + " ".join(f"{i:>6}" for i in range(VOCAB)))
    print("  target:  " + " ".join(f"{v:6.3f}" for v in target[start]))
    print("  投机  :  " + " ".join(f"{v:6.3f}" for v in spec_first))
    print("  自回归:  " + " ".join(f"{v:6.3f}" for v in ar_first))
    kl = kl_div(target[start], spec_first)
    print(f"  KL(target || 投机) = {kl:.4e}  "
          + ("一致 ✓" if kl < 5e-3 else "偏差偏大 ✗"))

    # ---- 验证 2：加速比模拟 ----
    # 生成较长序列，统计平均每次 target 调用产出多少 token
    rng3 = np.random.default_rng(789)
    gen_len = 2000
    seq, calls = speculative_generate(target, draft, start, gen_len, rng3)
    tokens_per_call = len(seq) / calls
    print()
    print("[验证 2] 加速比模拟（生成 %d token）" % gen_len)
    print(f"  target 前向调用次数 = {calls}")
    print(f"  平均每次调用产出 token 数 = {tokens_per_call:.2f}")
    print(f"  相对纯自回归的理论加速比 ≈ {tokens_per_call:.2f}x"
          "（忽略 draft 成本，上限 K+1={})".format(K + 1))

    print()
    print(SEP)
    print("结论：投机解码在不改变输出分布的前提下，用小模型草稿换取加速。")
    if DEVICE == "cuda":
        print("环境：CUDA GPU 就绪，转移矩阵已加载到 GPU。")
    elif TORCH_AVAILABLE:
        print("环境：无 CUDA GPU，已在 CPU 上真实运行完整逻辑。")
    else:
        print("环境：未安装 PyTorch，已用 NumPy 演示等价原理。")
    print(SEP)


if __name__ == "__main__":
    main()
