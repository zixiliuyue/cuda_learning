# Day 52: 采样与解码策略 — greedy / temperature / top-k / top-p

> 🧠 **第十阶段（LLM 推理）开篇**。前面你学会了写 kernel、融合算子。从本天起进入 LLM 推理主线：模型每一步吐出的是词表上的一个 **logits 向量**，「怎么从中挑下一个 token」就是**解码策略**，它直接决定生成质量、多样性与稳定性。

## 学习目标

1. **实现** greedy / temperature / top-k / top-p(nucleus) 四种采样策略
2. **理解** 温度调「分布形状」、top-k/top-p 调「候选范围」的区别
3. **观察** 同一 logits 分布下，各策略选出的 token 分布差异
4. **掌握** 「有 CUDA 用 GPU、无 CUDA 用 CPU」的推理代码写法（`device` 切换）

---

## 前置依赖

- **Day 31 Softmax / Flash Attention**：采样前要先把 logits 过 softmax 变概率
- **Day 9 归约**：top-k / top-p 里的排序与累加本质是归约类操作
- 基础概率知识：多项式采样 `multinomial`

---

## 核心概念

### 从 logits 到 token 的一步

```
模型输出 logits ──/T──> softmax ──> 概率分布 ──(截断/采样)──> 下一个 token
```

| 策略 | 做法 | 特点 |
|------|------|------|
| greedy | 取 `argmax` | 确定性，最保守，易重复 |
| temperature | `logits/T` 再 softmax 采样 | T↑更随机 / T↓更保守 |
| top-k | 只在概率前 k 个里采样 | 截断长尾，控制风险 |
| top-p | 在累积概率达 p 的最小集合里采样 | 自适应候选数（nucleus） |

### 关键直觉

- **temperature 调分布形状**：`T<1` 让分布更尖锐（接近 greedy），`T>1` 更平缓（更随机）。
- **top-k / top-p 调候选范围**：把长尾的低概率 token 直接排除，避免采到离谱结果。
- **top-p 比 top-k 更自适应**：分布尖锐时候选自然变少，分布平缓时候选自然变多。
- 实践中三者常配合使用（如 `T=0.8, top_p=0.95`）。

---

## 文件结构

```
day52_sampling/
├── python/sampling.py   # 四种采样策略 + 分布对比，含无 GPU/CPU/NumPy 三级运行
├── build.sh             # Linux/WSL2/macOS 运行脚本
├── build.bat            # Windows 运行脚本
└── README.md            # 本文件
```

> 本阶段以 **PyTorch** 为主。代码用 `device = "cuda" if torch.cuda.is_available() else "cpu"`：**有 CUDA 跑 GPU，无 CUDA 跑 CPU，逻辑都完整真实运行**；完全没有 torch 时才退化为 NumPy 演示原理。

---

## 编译与运行

### Linux / WSL2 / macOS

```bash
cd day52_sampling
pip install torch          # 可选，无 GPU 也能用 CPU 跑
python python/sampling.py
```

### Windows

```powershell
cd day52_sampling
.\build.bat
```

---

## 知识点总结

1. **greedy** 是 `argmax`，`top-k(k=1)` 与它等价。
2. **temperature** 通过缩放 logits 改变分布的尖锐程度。
3. **top-k** 用 `torch.topk` 保留前 k 个，重新归一化后 `multinomial` 采样。
4. **top-p** 先排序再算累积概率，保留累积到 p 的最小集合（至少保留 1 个）。
5. **device 切换**：一行 `device` 判断即可让同一份代码在 GPU/CPU 上运行。
6. **可复现性**：用 `torch.Generator().manual_seed()` 固定随机种子。

---

## 验收标准

- 程序打印四种策略在同一 logits 上的采样分布对比。
- greedy 恒定选中最高分 token（本例 t0）。
- temperature 越大，命中的不同 token 数越多。
- top-k 的 k 越大候选越多；`k=1` 退化为 greedy。
- top-p 随 p 增大候选集合变大，长尾被自然截断。
- 无 GPU 时在 CPU 上完整运行，exit code 0。

---

## 动手挑战

1. **组合策略**：实现 `temperature + top-p` 组合（先按 T 缩放，再做 nucleus 截断）。
2. **重复惩罚**：加入 repetition penalty，对已生成过的 token 降低其 logits。
3. **批量采样**：把单条 logits 扩成 `[batch, vocab]`，一次并行采样多条。

---

## 常见问题

- **Q：为什么不直接都用 greedy？**
  A：greedy 每步只看局部最优，容易陷入重复循环，生成文本单调。采样能提升多样性。

- **Q：temperature 设成 0 会怎样？**
  A：数学上等价于 greedy（分布退化为 one-hot），代码里需做除零保护。

---

## 延伸阅读

- [The Curious Case of Neural Text Degeneration（top-p 原论文）](https://arxiv.org/abs/1904.09751)
- [Hugging Face — How to generate text](https://huggingface.co/blog/how-to-generate)
