#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 58: 阶段项目 — 迷你 LLM 推理引擎（第十阶段收官）

把第十阶段(Day48-58)学到的核心零件整合成一个能端到端跑的玩具级推理引擎：

  ┌─────────────────────────────────────────────────────────┐
  │  MiniLLM (Transformer Decoder)                            │
  │    Embedding → N × [MHA(带 KV Cache) + FFN] → LMHead      │
  ├─────────────────────────────────────────────────────────┤
  │  KV Cache          自回归时缓存历史 K/V，避免重复计算(Day49)│
  │  采样(top-k/top-p)  temperature + top-k + nucleus(Day52)   │
  │  批处理             多个 prompt 同时生成(Day51/56)          │
  │  性能报告           prefill/decode 计时 + tokens/s          │
  └─────────────────────────────────────────────────────────┘

这是一个**随机初始化**的玩具模型（不加载真实权重），目的在于把「推理引擎的
骨架」跑通、看清生成流程与性能构成，而不是生成有意义的文本。

运行策略（能不回退就不回退）：
  - 有 PyTorch：真实张量前向，有 CUDA 用 GPU，无 CUDA 用 CPU，逻辑完全一致；
  - 无 PyTorch：用 NumPy 实现同一套前向 + KV Cache + 采样，原理不打折。
"""

import time
import math

import numpy as np

# ------------------------------------------------------------
# 环境探测（优先真实 GPU，不轻易回退）
# ------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

SEP = "=" * 64

# 玩具模型超参（小到足以在 CPU 上秒跑）
VOCAB = 256
D_MODEL = 128
N_HEAD = 4
N_LAYER = 4
D_FF = 512
MAX_SEQ = 256
HEAD_DIM = D_MODEL // N_HEAD


# ============================================================
#  采样：temperature + top-k + top-p(nucleus)（Day52）
# ============================================================
def sample_logits_np(logits, temperature, top_k, top_p, rng):
    """从 logits (VOCAB,) 采样一个 token id（NumPy 版）。"""
    if temperature <= 0:                       # 贪心
        return int(np.argmax(logits))
    logits = logits / temperature
    # top-k：只保留最大的 k 个
    if top_k and top_k < len(logits):
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    # softmax
    m = np.max(logits)
    probs = np.exp(logits - m)
    probs /= probs.sum()
    # top-p：保留累积概率达到 p 的最小集合
    if top_p and top_p < 1.0:
        order = np.argsort(probs)[::-1]
        csum = np.cumsum(probs[order])
        cutoff = np.searchsorted(csum, top_p) + 1
        keep = order[:cutoff]
        mask = np.zeros_like(probs)
        mask[keep] = probs[keep]
        probs = mask / mask.sum()
    return int(rng.choice(len(probs), p=probs))


def sample_logits_torch(logits, temperature, top_k, top_p, generator):
    """从 logits (VOCAB,) 采样一个 token id（PyTorch 版）。"""
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    if top_p and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        csum = torch.cumsum(sorted_probs, dim=-1)
        keep = csum <= top_p
        keep[0] = True                          # 至少保留概率最高的一个
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
        probs = filtered / filtered.sum()
    return int(torch.multinomial(probs, 1, generator=generator).item())


# ============================================================
#  PyTorch 版迷你 LLM（Transformer Decoder + KV Cache）
# ============================================================
if TORCH_AVAILABLE:

    class TorchMiniLLM:
        """随机初始化的玩具 decoder。带 per-layer KV Cache。"""

        def __init__(self):
            g = torch.Generator().manual_seed(0)

            def randn(*shape):
                return torch.randn(*shape, generator=g) * 0.02

            self.emb = randn(VOCAB, D_MODEL).to(DEVICE)
            self.pos = randn(MAX_SEQ, D_MODEL).to(DEVICE)
            self.layers = []
            for _ in range(N_LAYER):
                self.layers.append({
                    "wq": randn(D_MODEL, D_MODEL).to(DEVICE),
                    "wk": randn(D_MODEL, D_MODEL).to(DEVICE),
                    "wv": randn(D_MODEL, D_MODEL).to(DEVICE),
                    "wo": randn(D_MODEL, D_MODEL).to(DEVICE),
                    "w1": randn(D_MODEL, D_FF).to(DEVICE),
                    "w2": randn(D_FF, D_MODEL).to(DEVICE),
                })
            self.lm_head = randn(D_MODEL, VOCAB).to(DEVICE)

        def _attn(self, layer, x, cache, start_pos):
            """x: (B, T, D)。cache: dict 持有历史 K/V。返回 (B, T, D)。"""
            B, T, _ = x.shape
            q = (x @ layer["wq"]).view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            k = (x @ layer["wk"]).view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            v = (x @ layer["wv"]).view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            # KV Cache：把当前 K/V 拼到历史后面
            if cache["k"] is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            Tk = k.shape[2]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)  # (B,H,T,Tk)
            # 因果 mask：query 位置 start_pos+i 只能看到 <= 它的 key
            qpos = torch.arange(start_pos, start_pos + T, device=x.device).view(T, 1)
            kpos = torch.arange(Tk, device=x.device).view(1, Tk)
            mask = kpos > qpos
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores, dim=-1)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D_MODEL)
            return out @ layer["wo"]

        def forward(self, tokens, caches, start_pos):
            """tokens: (B, T) long。返回最后位置的 logits (B, VOCAB)。"""
            x = self.emb[tokens] + self.pos[start_pos:start_pos + tokens.shape[1]]
            for layer, cache in zip(self.layers, caches):
                x = x + self._attn(layer, x, cache, start_pos)
                h = torch.relu(x @ layer["w1"]) @ layer["w2"]
                x = x + h
            logits = x[:, -1, :] @ self.lm_head          # 只取最后位置
            return logits

        def new_caches(self, batch):
            return [{"k": None, "v": None} for _ in range(N_LAYER)]


# ============================================================
#  NumPy 版迷你 LLM（无 torch 时的等价实现）
# ============================================================
class NumpyMiniLLM:
    def __init__(self):
        rng = np.random.default_rng(0)

        def randn(*shape):
            return (rng.standard_normal(shape) * 0.02).astype(np.float32)

        self.emb = randn(VOCAB, D_MODEL)
        self.pos = randn(MAX_SEQ, D_MODEL)
        self.layers = []
        for _ in range(N_LAYER):
            self.layers.append({
                "wq": randn(D_MODEL, D_MODEL), "wk": randn(D_MODEL, D_MODEL),
                "wv": randn(D_MODEL, D_MODEL), "wo": randn(D_MODEL, D_MODEL),
                "w1": randn(D_MODEL, D_FF), "w2": randn(D_FF, D_MODEL),
            })
        self.lm_head = randn(D_MODEL, VOCAB)

    def _attn(self, layer, x, cache, start_pos):
        B, T, _ = x.shape
        q = (x @ layer["wq"]).reshape(B, T, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
        k = (x @ layer["wk"]).reshape(B, T, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
        v = (x @ layer["wv"]).reshape(B, T, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
        if cache["k"] is not None:
            k = np.concatenate([cache["k"], k], axis=2)
            v = np.concatenate([cache["v"], v], axis=2)
        cache["k"], cache["v"] = k, v
        Tk = k.shape[2]
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(HEAD_DIM)
        qpos = np.arange(start_pos, start_pos + T).reshape(T, 1)
        kpos = np.arange(Tk).reshape(1, Tk)
        scores = np.where(kpos > qpos, -np.inf, scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn /= attn.sum(axis=-1, keepdims=True)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D_MODEL)
        return out @ layer["wo"]

    def forward(self, tokens, caches, start_pos):
        x = self.emb[tokens] + self.pos[start_pos:start_pos + tokens.shape[1]]
        for layer, cache in zip(self.layers, caches):
            x = x + self._attn(layer, x, cache, start_pos)
            h = np.maximum(x @ layer["w1"], 0.0) @ layer["w2"]
            x = x + h
        return x[:, -1, :] @ self.lm_head

    def new_caches(self, batch):
        return [{"k": None, "v": None} for _ in range(N_LAYER)]


# ============================================================
#  推理引擎：prefill + 逐 token decode（批处理 + KV Cache + 采样）
# ============================================================
class MiniEngine:
    """把模型包成一个引擎：generate(prompts, max_new) → 生成结果 + 性能。"""

    def __init__(self):
        self.backend = "torch" if TORCH_AVAILABLE else "numpy"
        self.model = TorchMiniLLM() if TORCH_AVAILABLE else NumpyMiniLLM()
        if TORCH_AVAILABLE:
            self.gen = torch.Generator(device=DEVICE).manual_seed(1234)
        self.rng = np.random.default_rng(1234)

    def _to_model(self, arr):
        if TORCH_AVAILABLE:
            return torch.tensor(arr, dtype=torch.long, device=DEVICE)
        return np.array(arr, dtype=np.int64)

    def _logits_row(self, logits, b):
        return logits[b] if not TORCH_AVAILABLE else logits[b]

    def _sample(self, row, temperature, top_k, top_p):
        if TORCH_AVAILABLE:
            return sample_logits_torch(row, temperature, top_k, top_p, self.gen)
        return sample_logits_np(row, temperature, top_k, top_p, self.rng)

    def _sync(self):
        if TORCH_AVAILABLE and DEVICE == "cuda":
            torch.cuda.synchronize()

    def generate(self, prompts, max_new=32, temperature=0.8, top_k=40, top_p=0.95):
        """prompts: List[List[int]]（等长以便批处理）。返回 (输出序列, 计时)。"""
        B = len(prompts)
        plen = len(prompts[0])
        caches = self.model.new_caches(B)
        outputs = [list(p) for p in prompts]

        # ---- Prefill：一次性喂入整段 prompt，填充 KV Cache ----
        self._sync(); t0 = time.perf_counter()
        tok = self._to_model(prompts)                       # (B, plen)
        logits = self.model.forward(tok, caches, start_pos=0)
        next_tokens = [self._sample(self._logits_row(logits, b),
                                    temperature, top_k, top_p) for b in range(B)]
        for b in range(B):
            outputs[b].append(next_tokens[b])
        self._sync(); prefill_t = time.perf_counter() - t0

        # ---- Decode：逐 token，每步只喂 1 个 token（靠 KV Cache 复用历史）----
        self._sync(); t1 = time.perf_counter()
        for step in range(max_new - 1):
            pos = plen + step
            tok = self._to_model([[next_tokens[b]] for b in range(B)])  # (B,1)
            logits = self.model.forward(tok, caches, start_pos=pos)
            next_tokens = [self._sample(self._logits_row(logits, b),
                                        temperature, top_k, top_p) for b in range(B)]
            for b in range(B):
                outputs[b].append(next_tokens[b])
        self._sync(); decode_t = time.perf_counter() - t1

        return outputs, {"prefill_s": prefill_t, "decode_s": decode_t,
                         "batch": B, "prompt_len": plen, "new_tokens": max_new}


# ============================================================
#  主流程
# ============================================================
def main():
    print(SEP)
    print("Day 58 收官项目：迷你 LLM 推理引擎")
    print(SEP)
    print("""
  整合的零件：
    Transformer Decoder  Embedding → N×[MHA + FFN] → LMHead
    KV Cache            自回归缓存历史 K/V，decode 每步只算 1 个新 token
    采样                temperature + top-k + top-p(nucleus)
    批处理              多个 prompt 同时 prefill/decode
    性能报告            prefill/decode 分段计时 + tokens/s
  注：随机权重的玩具模型，看的是"引擎骨架"与性能构成，不是文本质量。
""")

    if TORCH_AVAILABLE:
        print(f"[环境] PyTorch 就绪，设备 = {DEVICE.upper()}，backend = torch")
    else:
        print("[环境] 未安装 PyTorch，backend = numpy（逻辑等价，纯 CPU）")
    print(f"[模型] VOCAB={VOCAB} D_MODEL={D_MODEL} N_HEAD={N_HEAD} "
          f"N_LAYER={N_LAYER} D_FF={D_FF}")

    engine = MiniEngine()

    # 构造一批等长 prompt（随机 token id 即可，玩具演示）
    rng = np.random.default_rng(7)
    B, PLEN, NEW = 4, 16, 48
    prompts = [[int(t) for t in rng.integers(0, VOCAB, size=PLEN)] for _ in range(B)]

    print()
    print(SEP)
    print(f"生成：batch={B}, prompt_len={PLEN}, max_new={NEW}")
    print(SEP)
    outputs, perf = engine.generate(prompts, max_new=NEW,
                                    temperature=0.8, top_k=40, top_p=0.95)

    # 展示第 0 条的生成过程（prompt → 续写）
    seq0 = outputs[0]
    print(f"  [样例] 请求#0  prompt tokens: {seq0[:PLEN]}")
    print(f"                 生成 tokens : {seq0[PLEN:PLEN+16]} ...(共{NEW}个)")

    # ---- 性能报告 ----
    total_new = perf["batch"] * perf["new_tokens"]
    decode_new = perf["batch"] * (perf["new_tokens"] - 1)
    total_t = perf["prefill_s"] + perf["decode_s"]
    print()
    print(SEP)
    print("性能报告")
    print(SEP)
    print(f"  {'阶段':<10}{'耗时(ms)':>12}{'产出token':>12}{'吞吐(tok/s)':>14}")
    print("  " + "-" * 48)
    prefill_tok = perf["batch"]  # prefill 每个请求产出 1 个 token
    print(f"  {'Prefill':<10}{perf['prefill_s']*1e3:>12.2f}{prefill_tok:>12}"
          f"{prefill_tok/perf['prefill_s']:>14.1f}")
    print(f"  {'Decode':<10}{perf['decode_s']*1e3:>12.2f}{decode_new:>12}"
          f"{decode_new/perf['decode_s']:>14.1f}")
    print(f"  {'合计':<10}{total_t*1e3:>12.2f}{total_new:>12}"
          f"{total_new/total_t:>14.1f}")
    print()
    print(f"  → 端到端吞吐 ≈ {total_new/total_t:.0f} tokens/s "
          f"(batch={B}, 每请求 {NEW} tokens)")
    print("  说明：Prefill 一次处理整段 prompt(并行度高)；Decode 逐 token(受")
    print("       KV Cache 与访存带宽约束)，是自回归生成耗时的主体。")

    print()
    print(SEP)
    print("第十阶段 (Day48-58) 回顾：从一个 Decoder 层到能服务的推理引擎")
    print(SEP)
    stages = [
        ("48", "Transformer Decoder 层", "MHA + FFN 前向，LLM 推理基本单元"),
        ("49", "KV Cache", "缓存历史 K/V，decode 每步只算新 token"),
        ("50", "PagedAttention", "分页管理 KV Cache，消除显存碎片"),
        ("51", "Continuous Batching", "变长请求动态组批，提升吞吐"),
        ("52", "采样与解码策略", "greedy/top-k/top-p/temperature"),
        ("53", "推理量化", "INT8/FP16 权重量化 + 精度对齐"),
        ("54", "算子融合", "Fused QKV / LayerNorm+Residual 减 kernel"),
        ("55", "投机解码", "小模型草稿 + 大模型验证加速"),
        ("56", "推理服务化", "请求队列 + 批处理调度 + 延迟/吞吐权衡"),
        ("57", "vLLM/TRT-LLM 架构", "运行时派 vs 编译派的关键优化"),
        ("58", "迷你推理引擎(本天)", "整合 KVCache+批处理+采样 端到端生成"),
    ]
    print(f"  {'Day':>4} | {'主题':<22} | 核心要点")
    print("  " + "-" * 60)
    for d, topic, point in stages:
        print(f"  {d:>4} | {topic:<22} | {point}")
    print()
    print("  一句话总结：LLM 推理引擎 = Decoder 前向 + KV Cache(省重复计算)")
    print("             + 分页显存管理 + 动态批处理(榨吞吐) + 采样(控生成)")
    print("             + 服务化调度(平衡延迟/吞吐)。本天把这些串成了一条线。")

    print()
    print(SEP)
    if TORCH_AVAILABLE and DEVICE == "cuda":
        print("环境：GPU 就绪，以上为真实 GPU 端到端生成与性能报告。")
    elif TORCH_AVAILABLE:
        print("环境：PyTorch(CPU) 运行。逻辑与 GPU 一致，仅设备不同；")
        print("      tokens/s 绝对值不代表 GPU 性能。")
    else:
        print("环境：NumPy 演示。整套前向/KV Cache/采样均真实运行(纯 CPU)。")
    print(SEP)


if __name__ == "__main__":
    main()
