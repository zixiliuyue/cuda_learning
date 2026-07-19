#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 50: PagedAttention（vLLM 核心思想）

Day49 里 KV Cache 是「一段连续增长的张量」。真实服务中有很多并发请求、长度各异，
若给每个请求都预留「最大长度」的连续显存，会造成严重浪费与碎片：
  - 内部碎片：预留 2048，实际只用了 100，其余全浪费；
  - 外部碎片：显存里到处是大小不一的空洞，凑不出一块连续大空间给新请求。

PagedAttention 借鉴操作系统「虚拟内存分页」：
  - 把 KV Cache 切成固定大小的 **block/page**（如每块存 4 个 token 的 K/V）；
  - 物理显存是一堆等大的物理块池；
  - 每个请求维护一张 **block table**，把「逻辑块号」映射到「物理块号」；
  - 需要时按块分配，用完归还，物理块可被任意请求复用 → 几乎零外部碎片。

本脚本实现一个简化版分页 KV Cache 管理器，演示：
  1. 固定块大小、物理块池、block table 映射；
  2. 逐 token 追加时按需分配新块；
  3. 对比「连续预分配」的碎片/浪费 vs 「分页」的利用率；
  4. 请求结束归还物理块，块被后续请求复用。

运行策略：有 torch 用真实张量存 KV（GPU/CPU 自动），无 torch 用 NumPy。核心管理逻辑纯 Python。
"""

SEP = "=" * 64

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

import numpy as np


# ============================================================
#  物理块池：一整块预分配显存，切成 num_blocks 个等大物理块
# ============================================================
class KVBlockPool:
    """固定大小物理块池。

    形状约定（简化为单层单头）：
      每个物理块存 block_size 个 token 的 K 和 V，各维度 d_model。
      整池张量形状: (num_blocks, 2, block_size, d_model)  # 2 = K/V
    """

    def __init__(self, num_blocks: int, block_size: int, d_model: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.d_model = d_model
        if TORCH_AVAILABLE:
            self.storage = torch.zeros(num_blocks, 2, block_size, d_model, device=DEVICE)
        else:
            self.storage = np.zeros((num_blocks, 2, block_size, d_model), dtype=np.float32)
        # 空闲物理块列表（栈）：初始全部空闲
        self.free_blocks = list(range(num_blocks))

    def allocate(self) -> int:
        """分配一个物理块，返回物理块号；无空闲则抛异常（模拟 OOM）。"""
        if not self.free_blocks:
            raise MemoryError("物理块池已耗尽（模拟显存 OOM）")
        return self.free_blocks.pop()

    def free(self, block_id: int):
        """归还物理块，供后续复用。"""
        self.free_blocks.append(block_id)

    def num_free(self) -> int:
        return len(self.free_blocks)


# ============================================================
#  单个请求的分页 KV Cache：维护 block table + 当前长度
# ============================================================
class PagedSequence:
    """一个请求的 KV Cache，用 block table 记录逻辑块->物理块的映射。"""

    def __init__(self, seq_id: int, pool: KVBlockPool):
        self.seq_id = seq_id
        self.pool = pool
        self.block_table = []   # 逻辑块 i -> 物理块 block_table[i]
        self.length = 0         # 已存 token 数

    def append_token(self, k_vec, v_vec):
        """追加一个新 token 的 K/V。当前块满了就按需申请新物理块。"""
        block_size = self.pool.block_size
        offset_in_block = self.length % block_size
        if offset_in_block == 0:
            # 需要一个新逻辑块 -> 向池申请一个物理块
            phys = self.pool.allocate()
            self.block_table.append(phys)
        logical_block = self.length // block_size
        phys = self.block_table[logical_block]
        # 写入物理块池对应位置
        self.pool.storage[phys, 0, offset_in_block] = k_vec  # K
        self.pool.storage[phys, 1, offset_in_block] = v_vec  # V
        self.length += 1

    def gather_kv(self):
        """按 block table 收集完整的 K/V 序列 (length, d_model)，用于做注意力。"""
        ks, vs = [], []
        for i in range(self.length):
            phys = self.block_table[i // self.pool.block_size]
            off = i % self.pool.block_size
            ks.append(self.pool.storage[phys, 0, off])
            vs.append(self.pool.storage[phys, 1, off])
        if TORCH_AVAILABLE:
            return torch.stack(ks), torch.stack(vs)
        return np.stack(ks), np.stack(vs)

    def num_blocks_used(self) -> int:
        return len(self.block_table)

    def release(self):
        """请求结束，归还所有物理块。"""
        for phys in self.block_table:
            self.pool.free(phys)
        self.block_table = []
        self.length = 0


# ============================================================
#  讲解
# ============================================================
def explain_paged_attention():
    print(SEP)
    print("PagedAttention：像操作系统分页一样管理 KV Cache")
    print(SEP)
    print("""
  连续预分配（朴素 KV Cache）的问题：
    请求A [██████░░░░░░░░░░░░░░░░]  预留 max_len，实际只用一点 → 内部碎片
    请求B [████████████░░░░░░░░░░]
    释放A后显存出现空洞，新请求要连续大块 → 外部碎片，凑不出来

  PagedAttention（固定块 + block table）：
    物理块池:  [P0][P1][P2][P3][P4][P5][P6][P7] ...   (等大，可复用)

    请求A 逻辑块:  L0 L1 L2         block table: [P3, P0, P5]
    请求B 逻辑块:  L0 L1            block table: [P1, P4]
                     └── 逻辑连续，物理可分散，按需分配 ──┘

    读取时按 block table 把分散的物理块「聚合」成逻辑上连续的 K/V。

  好处：
    - 按块分配，最多浪费「不足一块」的尾部 → 内部碎片极小；
    - 物理块等大可任意复用 → 几乎无外部碎片；
    - 显存利用率大幅提升，能容纳更多并发请求（vLLM 高吞吐的关键）。
""")


# ============================================================
#  演示 1：碎片对比（连续预分配 vs 分页）
# ============================================================
def demo_fragmentation(block_size, d_model, total_token_capacity):
    print(SEP)
    print("演示 1：显存利用率对比（连续预分配 vs 分页）")
    print(SEP)
    # 模拟一批变长请求的实际长度
    seq_lens = [3, 17, 5, 40, 1, 9]
    max_len = 64  # 朴素方案给每个请求预留的最大长度

    # 连续预分配：每个请求都占 max_len 个 token 的空间
    reserved = len(seq_lens) * max_len
    actually_used = sum(seq_lens)
    waste_naive = reserved - actually_used
    print(f"  请求实际长度: {seq_lens}，共 {actually_used} tokens")
    print(f"  [连续预分配] 每请求预留 max_len={max_len} → 预留 {reserved} tokens, "
          f"浪费 {waste_naive} ({waste_naive / reserved * 100:.0f}%)")

    # 分页：每个请求按 ceil(len/block_size) 个块分配
    import math
    blocks_needed = [math.ceil(l / block_size) for l in seq_lens]
    tokens_allocated = sum(b * block_size for b in blocks_needed)
    waste_paged = tokens_allocated - actually_used
    print(f"  [分页 block={block_size}] 各请求用块数 {blocks_needed}，"
          f"分配 {tokens_allocated} tokens, 浪费 {waste_paged} "
          f"({waste_paged / tokens_allocated * 100:.0f}%，仅尾部内部碎片)")
    print(f"  [结论] 分页把浪费从 {waste_naive} 降到 {waste_paged} tokens，利用率大幅提升 ✓")


# ============================================================
#  演示 2：真实分页管理器逐 token 追加 + 复用
# ============================================================
def demo_pool_manager(block_size, d_model):
    print()
    print(SEP)
    print("演示 2：分页 KV Cache 管理器（block table 映射 + 物理块复用）")
    print(SEP)

    def rand_vec():
        if TORCH_AVAILABLE:
            return torch.randn(d_model, device=DEVICE)
        return np.random.randn(d_model).astype(np.float32)

    pool = KVBlockPool(num_blocks=8, block_size=block_size, d_model=d_model)
    print(f"  物理块池: {pool.num_blocks} 块, 每块 {block_size} tokens, 初始空闲 {pool.num_free()}")

    # 请求 A：生成 10 个 token
    seqA = PagedSequence(seq_id=0, pool=pool)
    stored_k = []
    for _ in range(10):
        k, v = rand_vec(), rand_vec()
        stored_k.append(k)
        seqA.append_token(k, v)
    print(f"  请求A 生成 10 tokens -> 用了 {seqA.num_blocks_used()} 个物理块, "
          f"block_table={seqA.block_table}, 剩余空闲 {pool.num_free()}")

    # 验证：按 block table 聚合出的 K 与写入顺序一致
    gathered_k, _ = seqA.gather_kv()
    if TORCH_AVAILABLE:
        err = (gathered_k - torch.stack(stored_k)).abs().max().item()
    else:
        err = np.abs(gathered_k - np.stack(stored_k)).max()
    print(f"  [聚合验证] 按 block_table 还原的 K 与写入的最大误差 = {err:.2e} "
          + ("(分散物理块正确聚合 ✓)" if err < 1e-5 else "(错误 ✗)"))

    # 请求 A 结束，归还物理块
    freed_before = pool.num_free()
    seqA.release()
    print(f"  请求A 结束归还 -> 空闲块 {freed_before} → {pool.num_free()}（物理块回池可复用）")

    # 请求 B 复用刚归还的物理块
    seqB = PagedSequence(seq_id=1, pool=pool)
    for _ in range(6):
        seqB.append_token(rand_vec(), rand_vec())
    print(f"  请求B 生成 6 tokens -> 用了 {seqB.num_blocks_used()} 个物理块, "
          f"block_table={seqB.block_table}（复用了A释放的块）, 剩余空闲 {pool.num_free()}")


def main():
    if TORCH_AVAILABLE:
        torch.manual_seed(0)
    else:
        np.random.seed(0)

    explain_paged_attention()
    block_size, d_model = 4, 16
    demo_fragmentation(block_size, d_model, total_token_capacity=None)
    demo_pool_manager(block_size, d_model)

    print()
    print(SEP)
    if TORCH_AVAILABLE and DEVICE == "cuda":
        print("环境：PyTorch + CUDA GPU 就绪，物理块池分配在 GPU 显存上。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU。物理块池在 CPU 内存上（逻辑一致）。")
    else:
        print("环境：未安装 PyTorch。已用 NumPy 演示分页管理器（逻辑一致）。")
    print(SEP)


if __name__ == "__main__":
    main()
