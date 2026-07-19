#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 73: MoE 并行与专家路由（Mixture of Experts）

MoE 用「稀疏激活」换「大参数量」：一层里放很多个专家（Expert，各是一个
MLP），但每个 token 只被路由到 top-k 个专家，因此计算量不随专家数线性增长。
这让模型总参数量可以极大（如上千亿），而单 token 的计算量仍可控。

本天实现一个简化 MoE 层的完整前向：
  1. gating 网络：为每个 token 打分，选出 top-k 专家（token 路由）；
  2. dispatch：把 token 分发到各自的专家（专家可分布在多 rank，即 expert parallel）；
  3. expert compute：每个专家对分到的 token 做 MLP；
  4. combine：按 gating 权重把各专家输出加权合并回每个 token。
并统计专家负载均衡（load balancing）情况。

运行策略：单进程模拟多 rank 的 expert parallel（专家分布在不同 rank，用列表
表示各 rank 持有的专家）。有 torch 用 torch，无则 numpy。
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


# ------------------------------------------------------------
# 后端无关工具
# ------------------------------------------------------------
def softmax_np(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================
#  0. 原理讲解
# ============================================================
def explain_moe():
    print(SEP)
    print("MoE 核心思想：稀疏激活 —— 大参数量，小计算量")
    print(SEP)
    print("""
  一层放 E 个专家（每个专家=一个 MLP），但每个 token 只走 top-k 个：

    token --> [gating 网络] --> 打分 --> 选 top-k 专家
                                          |
             +----------+----------+------+-----+
             v          v          v            v
          Expert0    Expert1    Expert2  ...  Expert{E-1}
          (rank0)    (rank0)    (rank1)       (rank{R-1})   <- expert parallel
             |          |          |            |
             +----------+----- combine（按 gating 权重加权求和）----+
                                   |
                                   v
                              token 输出

  关键点：
   - 稀疏：token 只激活 k/E 的专家，FLOPs 远小于 dense。
   - expert parallel：专家分布到多 rank，dispatch/combine 需 All-to-All 通信。
   - 负载均衡：若 gating 总把 token 送给少数专家，会造成部分 rank 过载、部分空闲，
     需要辅助损失（load balancing loss）来促使路由均匀。
""")


# ============================================================
#  1. 专家（Expert）：一个两层 MLP
# ============================================================
class Expert:
    """单个专家 = Linear -> ReLU -> Linear。分布在某个 rank 上。"""

    def __init__(self, expert_id, rank_id, dim, hidden, seed):
        self.expert_id = expert_id
        self.rank_id = rank_id
        if TORCH_AVAILABLE:
            g = torch.Generator().manual_seed(seed)
            self.W1 = torch.randn(dim, hidden, generator=g) * 0.1
            self.W2 = torch.randn(hidden, dim, generator=g) * 0.1
        else:
            rng = np.random.default_rng(seed)
            self.W1 = (rng.standard_normal((dim, hidden)) * 0.1).astype(np.float32)
            self.W2 = (rng.standard_normal((hidden, dim)) * 0.1).astype(np.float32)

    def forward(self, x):
        """x: [num_tokens, dim] -> [num_tokens, dim]"""
        if TORCH_AVAILABLE:
            h = torch.relu(x @ self.W1)
            return h @ self.W2
        h = np.maximum(x @ self.W1, 0.0)
        return h @ self.W2


# ============================================================
#  2. gating 网络：为每个 token 选 top-k 专家
# ============================================================
def gating_topk(x, gate_w, k):
    """返回每个 token 的 top-k 专家索引与归一化权重。

    x: [N, dim]，gate_w: [dim, E]。
    logits -> softmax -> 取 top-k -> 对 top-k 权重重新归一化。
    """
    if TORCH_AVAILABLE:
        logits = x @ gate_w                     # [N, E]
        probs = torch.softmax(logits, dim=-1)
        topk_val, topk_idx = torch.topk(probs, k, dim=-1)   # [N, k]
        topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True)  # 重新归一化
        return topk_idx, topk_val
    logits = x @ gate_w
    probs = softmax_np(logits, axis=-1)
    topk_idx = np.argsort(-probs, axis=-1)[:, :k]           # 降序取前 k
    rows = np.arange(x.shape[0])[:, None]
    topk_val = probs[rows, topk_idx]
    topk_val = topk_val / topk_val.sum(axis=-1, keepdims=True)
    return topk_idx, topk_val


# ============================================================
#  3. MoE 前向：dispatch -> expert compute -> combine
# ============================================================
def moe_forward(x, experts, gate_w, k, num_experts):
    """完整 MoE 前向，并统计每个专家分到的 token 数（负载）。

    dispatch：按 top-k 结果把 token 索引分组到各专家；
    combine：各专家输出乘以 gating 权重后加回对应 token。
    """
    N = x.shape[0]
    dim = x.shape[-1]
    topk_idx, topk_val = gating_topk(x, gate_w, k)

    # 输出缓冲
    if TORCH_AVAILABLE:
        out = torch.zeros(N, dim)
    else:
        out = np.zeros((N, dim), np.float32)

    load = [0] * num_experts     # 每个专家分到的 token 数

    # dispatch：为每个专家收集「哪些 token 的第几个槽位」路由到它
    for e in range(num_experts):
        # 找出所有把专家 e 选进 top-k 的 (token, slot)
        if TORCH_AVAILABLE:
            mask = (topk_idx == e)                       # [N, k] bool
            token_ids, slot_ids = torch.where(mask)
            if token_ids.numel() == 0:
                continue
            tokens = x[token_ids]                        # [n_e, dim]
            weights = topk_val[token_ids, slot_ids].unsqueeze(-1)  # [n_e, 1]
            y = experts[e].forward(tokens)               # 专家计算
            # combine：加权累加回对应 token
            out.index_add_(0, token_ids, y * weights)
            load[e] = int(token_ids.numel())
        else:
            mask = (topk_idx == e)
            token_ids, slot_ids = np.where(mask)
            if token_ids.size == 0:
                continue
            tokens = x[token_ids]
            weights = topk_val[token_ids, slot_ids][:, None]
            y = experts[e].forward(tokens)
            np.add.at(out, token_ids, y * weights)
            load[e] = int(token_ids.size)

    return out, load, topk_idx, topk_val


# ============================================================
#  4. 参考实现：dense（每个 token 走全部专家再按权重合并，仅用于对拍）
# ============================================================
def moe_reference(x, experts, gate_w, k, num_experts):
    """朴素参考：显式对每个 token 计算 top-k 专家加权和，逐 token 循环。

    与 moe_forward 的分组 dispatch 应得到相同结果（数值对拍）。
    """
    N, dim = x.shape[0], x.shape[-1]
    topk_idx, topk_val = gating_topk(x, gate_w, k)
    if TORCH_AVAILABLE:
        out = torch.zeros(N, dim)
    else:
        out = np.zeros((N, dim), np.float32)
    for t in range(N):
        for s in range(k):
            e = int(topk_idx[t, s])
            w = float(topk_val[t, s])
            xi = x[t:t + 1]
            y = experts[e].forward(xi)
            out[t] += (y * w)[0]
    return out


# ============================================================
#  主流程
# ============================================================
def main():
    explain_moe()

    N, dim, hidden = 32, 16, 32
    num_experts, num_ranks, k = 8, 4, 2

    # 专家分布到多 rank（expert parallel）：每 rank 持 num_experts/num_ranks 个专家
    per_rank = num_experts // num_ranks
    experts = []
    for e in range(num_experts):
        rank_id = e // per_rank
        experts.append(Expert(e, rank_id, dim, hidden, seed=200 + e))

    if TORCH_AVAILABLE:
        g = torch.Generator().manual_seed(7)
        x = torch.randn(N, dim, generator=g)
        gate_w = torch.randn(dim, num_experts, generator=g) * 0.1
    else:
        rng = np.random.default_rng(7)
        x = rng.standard_normal((N, dim)).astype(np.float32)
        gate_w = (rng.standard_normal((dim, num_experts)) * 0.1).astype(np.float32)

    print(SEP)
    print(f"配置：{N} 个 token，{num_experts} 个专家分布在 {num_ranks} 个 rank，"
          f"top-k={k}")
    print(SEP)
    print(f"  每个 rank 持有 {per_rank} 个专家（expert parallel）")
    print(f"  稀疏度：每 token 只激活 {k}/{num_experts} = {k/num_experts:.0%} 的专家")

    out, load, topk_idx, topk_val = moe_forward(x, experts, gate_w, k, num_experts)

    print()
    print(SEP)
    print("专家负载均衡统计（每个专家分到的 token 数）")
    print(SEP)
    total_assign = sum(load)
    ideal = total_assign / num_experts
    for e in range(num_experts):
        rank_id = e // per_rank
        bar = "#" * load[e]
        print(f"  Expert{e} (rank{rank_id}): {load[e]:>3} 个 token  {bar}")
    # 负载均衡度量：变异系数 CV = std / mean，越小越均衡
    load_arr = np.array(load, dtype=np.float64)
    cv = float(load_arr.std() / (load_arr.mean() + 1e-9))
    print(f"\n  理想均衡值 = {ideal:.1f} 个/专家；实际变异系数 CV = {cv:.3f}"
          f"（越小越均衡）")

    # 各 rank 负载
    print("\n  各 rank 总负载（All-to-All 后每 rank 的计算量）：")
    for r in range(num_ranks):
        rank_load = sum(load[e] for e in range(num_experts) if e // per_rank == r)
        print(f"    rank{r}: {rank_load} 个 token")

    print()
    print(SEP)
    print("数值验证：分组 dispatch 前向 == 逐 token 参考实现")
    print(SEP)
    ref = moe_reference(x, experts, gate_w, k, num_experts)
    if TORCH_AVAILABLE:
        err = (out - ref).abs().max().item()
    else:
        err = float(np.max(np.abs(out - ref)))
    print(f"  最大误差 = {err:.2e}")
    print("  [结果] " + ("通过 ✓" if err < 1e-4 else "偏差过大 ✗"))

    print()
    print(SEP)
    if MULTI_GPU:
        print("环境：多 GPU。真实 MoE expert parallel 用 All-to-All 分发 token；"
              "本脚本单进程模拟以保证可复现。")
    elif GPU_AVAILABLE:
        print("环境：单 GPU。expert parallel 需多卡 All-to-All，本脚本单进程模拟专家分布。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 无 GPU。已用 torch(CPU) 完成 MoE 前向与对拍。")
    else:
        print("环境：无 PyTorch。已用 NumPy 完成 MoE 前向与对拍。")
    print(SEP)


if __name__ == "__main__":
    main()
