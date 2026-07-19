# Day 50: PagedAttention（vLLM 核心思想）

> 🚀 **第十阶段 · LLM 推理优化**。Day49 的 KV Cache 是一段连续增长的张量。可当成百上千个变长请求并发时，「给每个请求预留连续显存」会造成严重浪费与碎片。本天学习 vLLM 的核心创新——**把 KV Cache 分页管理**，像操作系统的虚拟内存一样。

## 学习目标

1. **识别** 连续预分配 KV Cache 的两类问题：内部碎片与外部碎片
2. **实现** 简化版分页 KV Cache 管理器：固定块大小、物理块池、block table 映射
3. **验证** 逻辑块→物理块的映射能正确聚合出连续 K/V，物理块可跨请求复用
4. **量化** 分页相比连续预分配在显存利用率上的提升

---

## 前置依赖

- **Day 49 KV Cache**：本天管理的正是 KV Cache 的物理存储
- **操作系统虚拟内存/分页**（可选）：PagedAttention 直接借鉴了这一思想

---

## 核心概念

### 连续预分配的碎片问题

```
  请求A [██████░░░░░░░░░░░░░░░░]  预留 max_len，实际用一点 → 内部碎片(浪费)
  请求B [████████████░░░░░░░░░░]
  释放A后留下空洞，新请求要连续大块 → 外部碎片(凑不出连续空间)
```

### PagedAttention：固定块 + block table

```
  物理块池:  [P0][P1][P2][P3][P4][P5][P6][P7] ...   (等大，可复用)

  请求A 逻辑块:  L0 L1 L2      block table: [P3, P0, P5]
  请求B 逻辑块:  L0 L1         block table: [P1, P4]
                   └─ 逻辑连续，物理分散，按需分配 ─┘

  读取时按 block table 把分散的物理块「聚合」成逻辑上连续的 K/V。
```

- **block/page**：固定大小，存 `block_size` 个 token 的 K/V。
- **物理块池**：一整块预分配显存切成等大块，用空闲列表管理。
- **block table**：每个请求一张，记录逻辑块号→物理块号。
- **按需分配**：当前块写满才申请新块；请求结束归还物理块供复用。

### 好处

- 内部碎片最多「不足一块」的尾部；
- 物理块等大可任意复用，几乎无外部碎片；
- 显存利用率大幅提升 → 同样显存容纳更多并发 → 更高吞吐。

---

## 文件结构

```
day50_paged_attention/
├── python/paged_attention.py  # 物理块池 + block table + 碎片对比，含无 torch 的 NumPy 演示
├── build.sh                   # Linux/WSL2 运行脚本
├── build.bat                  # Windows 运行脚本
└── README.md                  # 本文件
```

---

## 编译与运行

```bash
cd day50_paged_attention
pip install torch        # 可选；无 torch 会退化为 NumPy 演示
python python/paged_attention.py
```

Windows：`.\build.bat`

---

## 知识点总结

1. **固定块大小**：所有物理块等大，是可复用、无外部碎片的前提。
2. **物理块池 + 空闲列表**：分配即从空闲列表取块，释放即归还，O(1) 管理。
3. **block table**：逻辑连续、物理分散的映射表，是「分页」的核心数据结构。
4. **按需分配**：只有当前块写满才申请新块，避免一次性预留 max_len。
5. **聚合读取**：注意力计算前按 block table 把分散物理块聚合成连续 K/V。
6. **跨请求复用**：请求结束归还的物理块可被任意新请求使用。

---

## 验收标准

- 程序打印碎片对比：分页把浪费 token 数显著降低，显示利用率提升。
- 打印请求 A 的 block table，聚合还原的 K 与写入顺序最大误差 < 1e-5，显示「正确聚合 ✓」。
- 请求 A 释放后空闲块数回升，请求 B 复用了释放的物理块。
- 你能解释内部碎片与外部碎片的区别，以及分页为何能同时缓解两者。

---

## 动手挑战

1. **改 block_size**：把块大小从 4 改成 1 / 8 / 16，观察内部碎片与 block table 长度的权衡。
2. **模拟 OOM**：把物理块池调小，让并发请求耗尽块池，捕获 `MemoryError` 并实现「等待/抢占」。
3. **接上注意力**：用 `gather_kv()` 得到的 K/V 真正做一次因果注意力，接回 Day49。
4. **前缀共享**：让两个请求共享相同前缀的物理块（copy-on-write），体会 vLLM 的 prefix caching。

---

## 常见问题

- **Q：分页会让注意力变慢吗？**
  A：读取时需按 block table 聚合，有少量间接寻址开销；vLLM 用定制 CUDA kernel 直接在分页布局上算注意力，几乎无损，换来的是吞吐大幅提升。

- **Q：block_size 怎么选？**
  A：太小则 block table 长、管理开销大；太大则尾部内部碎片大。vLLM 默认 16，是经验折中。

- **Q：这和操作系统分页是一回事吗？**
  A：思想一致——逻辑地址（逻辑块）经页表（block table）映射到物理地址（物理块），从而消除外部碎片。

---

## 延伸阅读

- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM 论文)](https://arxiv.org/abs/2309.06180)
- [vLLM 官方文档](https://docs.vllm.ai/)
