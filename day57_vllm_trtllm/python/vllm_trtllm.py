#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 57: 认识 vLLM / TensorRT-LLM 架构

Day56 我们做了朴素批处理，发现「攒批」在延迟和吞吐间存在权衡。工业界两大
主流推理引擎给出了两条不同的优化路线：

  vLLM          —— 运行时(runtime)派：PagedAttention + continuous batching，
                    核心解决 KV Cache 显存碎片 与 变长请求的调度效率。
  TensorRT-LLM  —— 编译(compile)派：把整个网络离线编译成高度融合、
                    针对特定 GPU/精度调优的引擎，追求极致单请求延迟与吞吐。

本天以「原理 + 可运行的数据结构演示」为主，不依赖真实 vllm/trt：
  1. 用一个 BlockManager 演示 PagedAttention 如何像 OS 分页一样管理 KV Cache，
     并量化它相比「连续预分配」节省了多少显存（消除内部碎片）；
  2. 用一个调度器演示 continuous batching：请求逐 token 推进、完成即退出、
     空位立刻被新请求填补，对比朴素 static batching 的 GPU 利用率；
  3. 打印 vLLM vs TensorRT-LLM 的架构对比表与关键优化点。

运行策略：
  - 有 torch 则打印设备信息（GPU/CPU），本天演示本身是纯 Python 数据结构，
    在任何环境都真实运行、结果可复现；
  - 若本机恰好装了 vllm，会做一次可选探测并打印版本（无则跳过，不报错）。
"""

import random

import numpy as np

# ------------------------------------------------------------
# 环境探测（原理演示为主，探测仅用于打印信息）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

VLLM_AVAILABLE = False
try:
    import vllm  # noqa: F401
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  1. PagedAttention：像 OS 分页一样管理 KV Cache
# ============================================================
class BlockManager:
    """简化版 PagedAttention 块管理器。

    背景：自回归生成时每个请求的 KV Cache 随生成长度增长，且长度事先未知。
    朴素做法是按「最大长度」为每个请求连续预分配 → 大量显存被短请求浪费
    （内部碎片）。

    PagedAttention 的思路（借鉴操作系统虚拟内存分页）：
      - 把 KV Cache 切成固定大小的 block（如每块 16 个 token）；
      - 每个请求维护一张 block_table（逻辑块 → 物理块 的映射）；
      - 需要时才按块分配，用完归还，物理块可被任意请求复用。
    好处：几乎消除内部碎片，显存利用率接近 100%，从而能容纳更多并发请求。
    """

    def __init__(self, total_blocks, block_size):
        self.block_size = block_size
        self.total_blocks = total_blocks
        self.free = list(range(total_blocks))   # 空闲物理块池
        self.tables = {}                         # req_id -> [物理块号...]
        self.token_counts = {}                   # req_id -> 已生成 token 数
        self.peak_used = 0

    def _alloc_one(self):
        if not self.free:
            raise MemoryError("KV Cache 物理块耗尽（真实系统会触发抢占/换出）")
        return self.free.pop()

    def append_token(self, req_id):
        """给某请求追加一个 token 的 KV，必要时才分配新物理块。"""
        table = self.tables.setdefault(req_id, [])
        used_in_last = self.token_counts.get(req_id, 0) % self.block_size
        if used_in_last == 0:          # 上一块正好填满 → 需要新块
            table.append(self._alloc_one())
        self.token_counts[req_id] = self.token_counts.get(req_id, 0) + 1
        self.peak_used = max(self.peak_used, self.total_blocks - len(self.free))

    def free_req(self, req_id):
        """请求结束：归还它占用的所有物理块。"""
        for blk in self.tables.pop(req_id, []):
            self.free.append(blk)
        self.token_counts.pop(req_id, None)


def demo_paged_attention():
    print(SEP)
    print("演示 1：PagedAttention vs 连续预分配（KV Cache 显存利用率）")
    print(SEP)

    block_size = 16
    max_len = 512                 # 系统支持的最大序列长度
    # 一批变长请求：真实长度远小于 max_len（这正是碎片的来源）
    seq_lens = [37, 200, 8, 128, 64, 15, 300, 22]
    n = len(seq_lens)

    # 方案A：连续预分配 —— 每个请求都按 max_len 预留
    naive_tokens = n * max_len
    # 方案B：PagedAttention —— 按块(向上取整)分配，只为真实长度买单
    paged_tokens = sum(
        ((L + block_size - 1) // block_size) * block_size for L in seq_lens
    )
    actual_tokens = sum(seq_lens)

    print(f"  请求数={n}, 最大长度={max_len}, block_size={block_size}")
    print(f"  各请求真实长度: {seq_lens}")
    print()
    print(f"  {'方案':<22}{'占用token槽':>12}{'相对真实浪费':>14}")
    print("  " + "-" * 48)
    print(f"  {'连续预分配(按max_len)':<20}{naive_tokens:>12}"
          f"{(naive_tokens/actual_tokens - 1)*100:>12.0f}%")
    print(f"  {'PagedAttention(按块)':<20}{paged_tokens:>12}"
          f"{(paged_tokens/actual_tokens - 1)*100:>12.0f}%")
    print(f"  {'理论真实需求':<22}{actual_tokens:>12}{0:>12}%")
    print()
    saved = (1 - paged_tokens / naive_tokens) * 100
    print(f"  → PagedAttention 相比连续预分配节省显存约 {saved:.1f}%")
    print("    (省下的显存可容纳更多并发请求 → 吞吐进一步提升)")

    # 用 BlockManager 真实跑一遍：模拟逐 token 生成 + 请求结束归还
    print()
    print("  用 BlockManager 真实模拟按块分配/归还：")
    total_blocks = (max_len // block_size) * n
    mgr = BlockManager(total_blocks=total_blocks, block_size=block_size)
    for rid, L in enumerate(seq_lens):
        for _ in range(L):
            mgr.append_token(rid)
    print(f"    峰值占用物理块 = {mgr.peak_used}/{total_blocks}"
          f"  (每块={block_size} token)")
    for rid in list(mgr.tables.keys()):
        mgr.free_req(rid)
    print(f"    全部请求结束后空闲块 = {len(mgr.free)}/{total_blocks} (完全回收)")


# ============================================================
#  2. Continuous Batching vs Static Batching：GPU 利用率
# ============================================================
def demo_continuous_batching():
    print()
    print(SEP)
    print("演示 2：Continuous Batching vs Static Batching（GPU 利用率）")
    print(SEP)

    random.seed(0)
    # 一批请求，各自要生成的 token 数不同（变长输出）
    gen_lens = [random.randint(5, 40) for _ in range(8)]
    batch = len(gen_lens)
    print(f"  {batch} 个请求，各自输出长度: {gen_lens}")

    # --- Static batching：整批一起跑，必须等最长的那个结束 ---
    # 每一步都占用 batch 个 slot，即使短请求早已算完仍空转到最长结束
    max_len = max(gen_lens)
    static_steps = max_len
    static_slot_capacity = batch * static_steps          # 总 slot·step
    static_useful = sum(gen_lens)                         # 真正有用的计算
    static_util = static_useful / static_slot_capacity

    # --- Continuous batching：完成即退出，空位立刻被"新请求"填补 ---
    # 这里模拟：每一步统计仍在跑的请求数，完成的立即腾出 slot。
    # 有用计算总量不变，但被压缩到更少的 slot·step 里 → 利用率更高。
    remaining = sorted(gen_lens)
    step = 0
    active_slot_steps = 0
    live = list(gen_lens)
    while any(t > 0 for t in live):
        active = sum(1 for t in live if t > 0)
        active_slot_steps += active           # 只为还在跑的请求计费
        live = [t - 1 for t in live]
        step += 1
    cont_steps = step
    cont_util = static_useful / active_slot_steps

    print()
    print(f"  {'策略':<20}{'总步数':>8}{'slot·step':>12}{'GPU利用率':>12}")
    print("  " + "-" * 52)
    print(f"  {'Static batching':<20}{static_steps:>8}"
          f"{static_slot_capacity:>12}{static_util*100:>11.1f}%")
    print(f"  {'Continuous batching':<20}{cont_steps:>8}"
          f"{active_slot_steps:>12}{cont_util*100:>11.1f}%")
    print()
    print("  说明：static batching 里短请求算完后 slot 仍被占着空转到最长结束；")
    print("       continuous batching 中完成的请求立即退出，空位被新请求填补，")
    print(f"       GPU 利用率从 {static_util*100:.0f}% 提升到 {cont_util*100:.0f}%。")
    print("       (真实 vLLM 里新请求会源源不断地补进来，利用率可长期接近满载)")


# ============================================================
#  3. 架构对比表
# ============================================================
def print_architecture_comparison():
    print()
    print(SEP)
    print("vLLM vs TensorRT-LLM：架构对比")
    print(SEP)
    print("""
  维度            vLLM(运行时派)            TensorRT-LLM(编译派)
  ------------------------------------------------------------------
  核心思想        运行时动态调度            离线编译成静态引擎
  KV Cache        PagedAttention(分页)      连续/分页(较新版本支持)
  批处理          Continuous batching       In-flight batching(类似)
  优化时机        运行时(Python 灵活)       编译期(离线, 面向具体GPU)
  算子            PagedAttention CUDA核      深度融合 + 内核自动选择
  精度            FP16/BF16/FP8/量化        FP16/BF16/FP8/INT8/INT4
  易用性          高(pip 装, HF 权重直用)   中(需 build engine 步骤)
  灵活性          高(改调度/加采样容易)     低(引擎编好后相对固定)
  典型优势        高吞吐 + 易部署            极致单请求延迟 + 峰值吞吐
  典型场景        通用在线服务/多变负载     固定模型的极致性能部署
""")
    print("  关键优化点小结：")
    print("   vLLM :")
    print("     - PagedAttention：分页 KV Cache，消除碎片，显存利用率↑")
    print("     - Continuous batching：token 级调度，完成即退出，GPU 利用率↑")
    print("     - 支持前缀共享(prefix caching)：相同前缀的 KV 块可复用")
    print("   TensorRT-LLM :")
    print("     - 编译期算子融合(QKV/GEMM/LayerNorm)，减少 kernel 启动")
    print("     - 面向具体 GPU/精度的内核自动选择与调优")
    print("     - 支持 INT8/INT4/FP8 量化与 in-flight batching")
    print()
    print("  一句话：vLLM 用『聪明的运行时调度』榨吞吐且易用；")
    print("          TensorRT-LLM 用『离线深度编译』榨极致延迟与峰值性能。")


# ============================================================
#  主流程
# ============================================================
def main():
    demo_paged_attention()
    demo_continuous_batching()
    print_architecture_comparison()

    print()
    print(SEP)
    if TORCH_AVAILABLE:
        print(f"环境：PyTorch 就绪，设备 = {DEVICE.upper()}。")
    else:
        print("环境：未安装 PyTorch。本天为纯数据结构原理演示，不受影响。")
    if VLLM_AVAILABLE:
        print(f"检测到本机安装了 vLLM v{getattr(vllm, '__version__', '?')}，"
              f"可进一步用真实引擎实验。")
    else:
        print("未检测到 vLLM（正常）。本天以原理演示为主；")
        print("如需真实体验：Linux + NVIDIA GPU 上 `pip install vllm` 后跑官方示例。")
    print(SEP)


if __name__ == "__main__":
    main()
