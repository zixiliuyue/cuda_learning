#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 49: KV Cache 原理与实现

自回归生成（LLM 逐 token 吐字）时，每生成一个新 token，都要对**整段已有序列**做
注意力。若每步都从头重算所有历史 token 的 Key/Value，计算量是巨大的浪费——因为
历史 token 的 K/V 在权重不变时**永远不变**。

KV Cache 的核心思想：把每一步算出的 K、V 缓存起来，下一步只为**新 token**计算 Q/K/V，
再把新 K/V 追加进缓存，与全部历史做注意力。这样每步只算 1 个 token，而非 T 个。

本脚本：
  1. 手写一个带因果注意力的单层模型；
  2. 用两种方式做自回归生成：
       (a) 无 cache：每步都把 0..t 全序列重新前向；
       (b) 有 cache：每步只前向新 token，K/V 追加进缓存；
  3. 验证两者输出逐 token 完全一致；
  4. 统计两者的「K/V 投影计算的 token 次数」，看清计算量差异（O(T^2) vs O(T)）。

运行策略：有 CUDA 用 GPU，无 CUDA 用 CPU（逻辑一致），无 torch 用 NumPy 演示。
"""

import math

SEP = "=" * 64

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
#  PyTorch 实现：可选 KV Cache 的单头因果注意力模型
# ============================================================
if TORCH_AVAILABLE:

    class TinyCausalModel(nn.Module):
        """一个极简单头因果注意力模型，用于对比有/无 KV Cache。

        这里用固定（非随机重置）的线性层作为 Q/K/V/O 投影，保证两种生成方式
        用的是同一套权重，从而可以逐 token 比对输出是否一致。

        计数器 self.kv_token_count 记录「被做过 K/V 投影的 token 次数」，
        用来量化 KV Cache 省了多少重复计算。
        """

        def __init__(self, d_model: int):
            super().__init__()
            self.d_model = d_model
            self.Wq = nn.Linear(d_model, d_model, bias=False)
            self.Wk = nn.Linear(d_model, d_model, bias=False)
            self.Wv = nn.Linear(d_model, d_model, bias=False)
            self.Wo = nn.Linear(d_model, d_model, bias=False)
            self.kv_token_count = 0  # 统计 K/V 投影处理过的 token 数

        def _attention(self, q, k, v):
            # q:(Tq,D) k,v:(Tk,D)；因果：q 的第 i 行只能看 k 的前 i(相对全局) 个
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_model)  # (Tq,Tk)
            Tq, Tk = att.shape
            # 全序列前向时 Tq==Tk，需要下三角 mask；增量时 Tq==1 且能看到全部历史
            if Tq == Tk:
                mask = torch.tril(torch.ones(Tq, Tk, device=att.device))
                att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            return att @ v  # (Tq,D)

        # ---- 方式 (a)：无 cache，每次传入完整序列，全部重算 ----
        def forward_full(self, x_seq: "torch.Tensor") -> "torch.Tensor":
            """x_seq: (T, D)，返回最后一个位置的输出 (D,)。"""
            T = x_seq.shape[0]
            q = self.Wq(x_seq)
            k = self.Wk(x_seq)
            v = self.Wv(x_seq)
            self.kv_token_count += T  # 每步都对全部 T 个 token 做了 K/V 投影
            out = self._attention(q, k, v)  # (T, D)
            return self.Wo(out[-1])         # 只取最后一个位置

        # ---- 方式 (b)：有 cache，只算新 token，K/V 追加 ----
        def forward_cached(self, x_new: "torch.Tensor", cache: dict) -> "torch.Tensor":
            """x_new: (D,) 单个新 token；cache 内含累积的 'k','v' (T,D)。"""
            x_new = x_new.view(1, self.d_model)  # (1,D)
            q = self.Wq(x_new)                    # (1,D) 只算新 token 的 Q
            k_new = self.Wk(x_new)                # (1,D)
            v_new = self.Wv(x_new)
            self.kv_token_count += 1              # 只对 1 个新 token 做 K/V 投影

            if cache.get("k") is None:
                cache["k"], cache["v"] = k_new, v_new
            else:
                cache["k"] = torch.cat([cache["k"], k_new], dim=0)  # 追加历史
                cache["v"] = torch.cat([cache["v"], v_new], dim=0)

            out = self._attention(q, cache["k"], cache["v"])  # (1,D)，看到全部历史
            return self.Wo(out[0])


# ============================================================
#  NumPy 版：无 torch 时演示 KV Cache 等价逻辑
# ============================================================
class NumpyCausalModel:
    def __init__(self, d_model, seed=0):
        rng = np.random.default_rng(seed)
        self.d = d_model
        self.Wq = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.1
        self.Wk = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.1
        self.Wv = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.1
        self.Wo = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.1
        self.kv_token_count = 0

    @staticmethod
    def _softmax(a):
        a = a - a.max(axis=-1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=-1, keepdims=True)

    def forward_full(self, x_seq):
        T = x_seq.shape[0]
        q = x_seq @ self.Wq
        k = x_seq @ self.Wk
        v = x_seq @ self.Wv
        self.kv_token_count += T
        att = (q @ k.T) / math.sqrt(self.d)
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        att = np.where(mask, -np.inf, att)
        att = self._softmax(att)
        out = att @ v
        return (out[-1] @ self.Wo)

    def forward_cached(self, x_new, cache):
        x_new = x_new.reshape(1, self.d)
        q = x_new @ self.Wq
        k_new = x_new @ self.Wk
        v_new = x_new @ self.Wv
        self.kv_token_count += 1
        if cache.get("k") is None:
            cache["k"], cache["v"] = k_new, v_new
        else:
            cache["k"] = np.concatenate([cache["k"], k_new], axis=0)
            cache["v"] = np.concatenate([cache["v"], v_new], axis=0)
        att = (q @ cache["k"].T) / math.sqrt(self.d)
        att = self._softmax(att)
        out = att @ cache["v"]
        return (out[0] @ self.Wo)


# ============================================================
#  讲解
# ============================================================
def explain_kv_cache():
    print(SEP)
    print("KV Cache：为什么自回归生成需要它")
    print(SEP)
    print("""
  自回归生成第 t 步时，注意力需要历史所有 token 的 Key/Value：

  无 KV Cache（每步重算全序列）：
    step0: 算 [t0]             K/V 投影 1 个 token
    step1: 算 [t0,t1]          K/V 投影 2 个   ← t0 被重复算
    step2: 算 [t0,t1,t2]       K/V 投影 3 个   ← t0,t1 又被重复算
    ...    总计 1+2+...+T = O(T^2) 次 K/V 投影

  有 KV Cache（只算新 token，历史从缓存取）：
    step0: 算 [t0] → 存 K0,V0            1 个
    step1: 算 [t1] → 存 K1,V1，读 K0,V0  1 个
    step2: 算 [t2] → 存 K2,V2，读 K0..1  1 个
    ...    总计 T 次 K/V 投影 = O(T)

  代价：需要存 K/V 缓存，显存 ∝ 层数 × 序列长 × d_model × 2(K/V)。
  这正是后续 Day50 PagedAttention 要解决的「KV 显存」问题的根源。
""")


# ============================================================
#  主流程
# ============================================================
def run_torch():
    torch.manual_seed(0)
    D, T = 32, 12
    print(f"[环境] 使用 device = {DEVICE}")
    print(f"[配置] d_model={D}, 生成序列长度 T={T}")

    model = TinyCausalModel(D).to(DEVICE)
    model.eval()

    # 构造一串「已生成」的 token 表示（这里用随机向量模拟 embedding 序列）
    tokens = torch.randn(T, D, device=DEVICE)

    # ---- 方式 a：无 cache，逐步重算全序列 ----
    model.kv_token_count = 0
    outs_full = []
    with torch.no_grad():
        for t in range(T):
            y = model.forward_full(tokens[: t + 1])  # 传入 0..t 全序列
            outs_full.append(y)
    outs_full = torch.stack(outs_full)  # (T, D)
    cnt_full = model.kv_token_count

    # ---- 方式 b：有 cache，逐步只算新 token ----
    model.kv_token_count = 0
    cache = {"k": None, "v": None}
    outs_cached = []
    with torch.no_grad():
        for t in range(T):
            y = model.forward_cached(tokens[t], cache)
            outs_cached.append(y)
    outs_cached = torch.stack(outs_cached)  # (T, D)
    cnt_cached = model.kv_token_count

    max_err = (outs_full - outs_cached).abs().max().item()
    print(f"[一致性] 无cache vs 有cache 逐 token 最大误差 = {max_err:.2e} "
          + ("(一致 ✓)" if max_err < 1e-4 else "(不一致 ✗)"))
    print(f"[计算量] 无 cache 的 K/V 投影 token 次数 = {cnt_full}  (= 1+2+...+{T} = O(T^2))")
    print(f"[计算量] 有 cache 的 K/V 投影 token 次数 = {cnt_cached}  (= {T} = O(T))")
    print(f"[加速比] 重复计算减少 {cnt_full / cnt_cached:.1f}x（序列越长，收益越大）")
    print(f"[缓存尺寸] KV Cache 形状: K={tuple(cache['k'].shape)}, V={tuple(cache['v'].shape)}")


def run_numpy():
    print(f"[环境] 未安装 PyTorch，用 NumPy 演示 KV Cache 等价逻辑 (device={DEVICE})")
    D, T = 16, 10
    rng = np.random.default_rng(1)
    model = NumpyCausalModel(D)
    tokens = rng.standard_normal((T, D)).astype(np.float32)

    model.kv_token_count = 0
    outs_full = np.stack([model.forward_full(tokens[: t + 1]) for t in range(T)])
    cnt_full = model.kv_token_count

    model.kv_token_count = 0
    cache = {"k": None, "v": None}
    outs_cached = np.stack([model.forward_cached(tokens[t], cache) for t in range(T)])
    cnt_cached = model.kv_token_count

    max_err = np.abs(outs_full - outs_cached).max()
    print(f"[一致性] 无cache vs 有cache 逐 token 最大误差 = {max_err:.2e} "
          + ("(一致 ✓)" if max_err < 1e-4 else "(不一致 ✗)"))
    print(f"[计算量] 无 cache K/V 投影 token 次数 = {cnt_full} (O(T^2))")
    print(f"[计算量] 有 cache K/V 投影 token 次数 = {cnt_cached} (O(T))")
    print(f"[加速比] 重复计算减少 {cnt_full / cnt_cached:.1f}x")


def main():
    explain_kv_cache()
    print(SEP)
    print("对比实验：无 KV Cache vs 有 KV Cache")
    print(SEP)
    if TORCH_AVAILABLE:
        run_torch()
    else:
        run_numpy()

    print()
    print(SEP)
    if TORCH_AVAILABLE and DEVICE == "cuda":
        print("环境：PyTorch + CUDA GPU 就绪，已在 GPU 上运行。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU。已在 CPU 上运行完整逻辑（结果一致）。")
    else:
        print("环境：未安装 PyTorch。已用 NumPy 演示 KV Cache 原理。")
    print(SEP)


if __name__ == "__main__":
    main()
