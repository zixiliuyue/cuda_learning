# Day 49: KV Cache 原理与实现

> 🚀 **第十阶段 · LLM 推理优化**。Day48 搭好了一个 Decoder 层，但真正推理时是**逐 token** 生成的。本天回答一个关键问题：为什么每生成一个字都重算整段历史是巨大浪费，以及 KV Cache 如何把它从 O(T²) 降到 O(T)。

## 学习目标

1. **理解** 自回归生成为何需要缓存历史 token 的 Key/Value
2. **实现** 两种生成方式：无 cache（每步重算全序列）与有 cache（只算新 token）
3. **验证** 两者输出逐 token **完全一致**（KV Cache 不改变结果，只省计算）
4. **量化** 两者的 K/V 投影计算量差异（O(T²) vs O(T)）与显存代价

---

## 前置依赖

- **Day 48 Transformer Decoder 层**：本天在其因果注意力基础上加缓存
- **Day 31 注意力**：理解 Q/K/V 的角色——Q 来自当前 token，K/V 来自全部历史

---

## 核心概念

### 为什么能缓存？

注意力里，历史 token 的 **Key/Value 只由它自己的表示和（不变的）权重决定**，与后续生成无关。所以一旦算出就可以永久缓存，后面每步直接取用。而 **Query 只需要当前新 token 的**。

```
  无 KV Cache（每步重算全序列）        有 KV Cache（只算新 token）
  step0: 算 [t0]        1 个            step0: 算 t0 → 存 K0,V0        1 个
  step1: 算 [t0,t1]     2 个 (重算t0)   step1: 算 t1 → 存 K1,V1        1 个
  step2: 算 [t0,t1,t2]  3 个 (重算t0,1) step2: 算 t2 → 存 K2,V2        1 个
  ...                                   ...
  合计 1+2+...+T = O(T^2)               合计 T = O(T)
```

### 代价：KV Cache 的显存

```
  KV Cache 显存 ≈ 层数 × 序列长度 × d_model × 2(K和V) × 精度字节数 × batch
```

序列越长、并发请求越多，KV Cache 越吃显存——这就是 **Day50 PagedAttention** 要解决的问题。

---

## 文件结构

```
day49_kv_cache/
├── python/kv_cache.py   # 有/无 KV Cache 的自回归生成对比，含无 torch 的 NumPy 演示
├── build.sh             # Linux/WSL2 运行脚本
├── build.bat            # Windows 运行脚本
└── README.md            # 本文件
```

---

## 编译与运行

```bash
cd day49_kv_cache
pip install torch        # 可选；无 torch 会退化为 NumPy 演示
python python/kv_cache.py
```

Windows：`.\build.bat`

---

## 知识点总结

1. **K/V 可缓存**：历史 token 的 K/V 不随生成变化，算一次存下来即可。
2. **Q 只需当前**：每步只需新 token 的 Query，与缓存的历史 K/V 做注意力。
3. **结果等价**：KV Cache 是纯粹的计算复用，不改变数值（本天用逐 token 误差验证）。
4. **计算量**：无 cache O(T²)，有 cache O(T)，序列越长收益越大。
5. **显存换计算**：代价是保存 K/V 缓存，占用随序列长与并发数线性增长。
6. **增量注意力**：增量步 Q 只有 1 行，能看到全部历史 K/V，无需重新 mask。

---

## 验收标准

- 程序打印无 cache / 有 cache 两种方式，逐 token 输出最大误差 < 1e-4，显示「一致 ✓」。
- 打印两者 K/V 投影 token 次数：无 cache = `1+2+...+T`，有 cache = `T`，并给出加速比。
- 你能解释「为什么 KV Cache 不改变生成结果，只减少重复计算」。

---

## 动手挑战

1. **画计算量曲线**：把 T 从 8 增到 256，记录两种方式的 K/V 次数，画出 O(T²) 与 O(T) 曲线。
2. **多层缓存**：把模型堆成 N 层，为每层各维护一份 KV Cache，观察显存随层数线性增长。
3. **估算显存**：按 `层数×T×d_model×2×2字节`（fp16）估算 7B 模型在 T=4096 时的 KV Cache 大小。
4. **加多头**：把单头改成多头，缓存形状变为 `(n_head, T, d_head)`，验证一致性仍成立。

---

## 常见问题

- **Q：KV Cache 会影响生成质量吗？**
  A：不会。它是无损的计算复用，本天用逐 token 误差 < 1e-4 证明两者等价。

- **Q：为什么增量步不需要 causal mask？**
  A：增量步 Query 只有当前这 1 个 token，它本就应该看到全部历史 K/V，没有「未来」可屏蔽。

- **Q：KV Cache 存在哪？**
  A：显存里。它是长上下文/高并发推理的主要显存开销，催生了分页（Day50）等管理技术。

---

## 延伸阅读

- [Transformers KV Cache 解析 (HuggingFace)](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [Efficiently Scaling Transformer Inference (Google)](https://arxiv.org/abs/2211.05102)
