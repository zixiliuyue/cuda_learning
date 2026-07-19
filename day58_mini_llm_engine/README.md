# Day 58: 阶段项目 — 迷你 LLM 推理引擎

> 🏁 **第十阶段（LLM 推理优化）收官项目**。把 Day48-57 学到的核心零件整合成一个能端到端跑的玩具级推理引擎：Transformer Decoder + KV Cache + top-k/top-p 采样 + 批处理 + 性能报告。

## 学习目标

1. **整合** Transformer Decoder 前向、KV Cache、采样、批处理为一个引擎
2. **实现** prefill（一次处理整段 prompt）+ decode（逐 token 自回归）两阶段生成
3. **实现** temperature + top-k + top-p(nucleus) 采样
4. **产出** 性能报告：prefill/decode 分段计时 + tokens/s，理解耗时构成

---

## 前置依赖

- **Day 48 Transformer Decoder**：本引擎的模型骨架（MHA + FFN）
- **Day 49 KV Cache**：decode 每步只算新 token 的关键
- **Day 52 采样与解码**：temperature / top-k / top-p 的实现
- **Day 51/56 批处理与服务化**：多 prompt 同时生成的批处理思路
- **Day 57 vLLM/TRT-LLM**：本引擎是那些工业引擎的极简教学版

---

## 核心概念

### 引擎骨架

```
  ┌─────────────────────────────────────────────────────────┐
  │  MiniLLM (Transformer Decoder)                            │
  │    Embedding → N × [MHA(带 KV Cache) + FFN] → LMHead      │
  ├─────────────────────────────────────────────────────────┤
  │  KV Cache          自回归时缓存历史 K/V，避免重复计算       │
  │  采样(top-k/top-p)  temperature + top-k + nucleus          │
  │  批处理             多个 prompt 同时生成                    │
  │  性能报告           prefill/decode 计时 + tokens/s          │
  └─────────────────────────────────────────────────────────┘
```

### Prefill vs Decode：自回归生成的两个阶段

| 阶段 | 输入 | 特点 | 瓶颈 |
|------|------|------|------|
| **Prefill** | 整段 prompt（T 个 token 一次） | 并行度高，一次填满 KV Cache | 计算密集 |
| **Decode** | 每步 1 个新 token | 靠 KV Cache 复用历史，逐 token 串行 | 访存/带宽密集 |

> Decode 是生成耗时的主体：每生成一个 token 就要走一遍全部层，但每步只算 1 个新位置（历史 K/V 已缓存）。这正是 KV Cache 的价值——把 O(T²) 的重复计算降为每步 O(T)。

### 采样策略（Day52 回顾）

- **temperature**：`logits / T`，T↑ 更随机，T→0 趋近贪心。
- **top-k**：只在概率最高的 k 个 token 中采样。
- **top-p (nucleus)**：保留累积概率达到 p 的最小 token 集合，动态截断。

> 本天是**随机初始化的玩具模型**，看的是「引擎骨架」与「性能构成」，不是文本质量。

---

## 第十阶段（Day48-58）回顾表

| Day | 主题 | 核心要点 |
|-----|------|----------|
| 48 | Transformer Decoder 层 | MHA + FFN 前向，LLM 推理基本单元 |
| 49 | KV Cache | 缓存历史 K/V，decode 每步只算新 token |
| 50 | PagedAttention | 分页管理 KV Cache，消除显存碎片 |
| 51 | Continuous Batching | 变长请求动态组批，提升吞吐 |
| 52 | 采样与解码策略 | greedy / top-k / top-p / temperature |
| 53 | 推理量化 | INT8/FP16 权重量化 + 精度对齐 |
| 54 | 算子融合 | Fused QKV / LayerNorm+Residual 减 kernel 启动 |
| 55 | 投机解码 | 小模型草稿 + 大模型验证加速 |
| 56 | 推理服务化 | 请求队列 + 批处理调度 + 延迟/吞吐权衡 |
| 57 | vLLM / TRT-LLM 架构 | 运行时派 vs 编译派的关键优化 |
| 58 | 迷你推理引擎（本天） | 整合 KV Cache + 批处理 + 采样，端到端生成 |

**一句话总结**：LLM 推理引擎 = Decoder 前向 + KV Cache（省重复计算）+ 分页显存管理 + 动态批处理（榨吞吐）+ 采样（控生成）+ 服务化调度（平衡延迟/吞吐）。本天把这些串成了一条线。

---

## 文件结构

```
day58_mini_llm_engine/
├── python/mini_llm_engine.py   # 迷你推理引擎：Decoder+KVCache+采样+批处理+性能报告
├── build.sh                    # Linux/WSL2 运行脚本
├── build.bat                   # Windows 运行脚本
└── README.md                   # 本文件
```

> 本阶段以 PyTorch 为主：**有 CUDA 用 GPU，无 CUDA 自动用 CPU**，逻辑完全一致；连 PyTorch 都没有时用 NumPy 实现同一套前向 + KV Cache + 采样，原理不打折。

---

## 编译与运行

### Linux / WSL2（推荐，有 NVIDIA GPU）

```bash
cd day58_mini_llm_engine
pip install torch          # 有 GPU 则自动用 GPU
python python/mini_llm_engine.py
```

### Windows

```powershell
cd day58_mini_llm_engine
.\build.bat
```

> 引擎骨架、KV Cache、采样、批处理、性能计时在**任何环境都真实运行**；只有张量运算的设备在无 GPU 时落到 CPU/NumPy。

---

## 知识点总结

1. **两阶段生成**：prefill 一次填满 KV Cache，decode 逐 token 复用。
2. **KV Cache 拼接**：每步把新 K/V `cat` 到历史后面，因果 mask 用 query/key 位置比较生成。
3. **批处理**：多个 prompt 打包成 `(B, T)`，一次前向服务整批。
4. **采样流水线**：temperature → top-k 截断 → softmax → top-p 截断 → 多项式采样。
5. **性能构成**：decode 通常是生成耗时主体，tokens/s 是核心指标。
6. **工程映射**：本引擎的每个零件都对应 vLLM/TRT-LLM 的一个真实模块。

---

## 验收标准

- 程序端到端跑完并打印生成 token 序列、prefill/decode 分段耗时与 tokens/s。
- 你能解释 prefill 与 decode 的区别，以及为何 decode 是耗时主体。
- 无 GPU 时程序照常跑完（exit code 0），NumPy 与 PyTorch 路径逻辑一致。
- 你能对照回顾表复述第十阶段每一天在引擎里扮演的角色。

---

## 动手挑战

1. **加 EOS 提前停止**：约定某个 token 为结束符，某请求采到它就停止，配合 continuous batching 让它提前退出。
2. **变长 prompt**：支持不等长 prompt（左侧 padding + attention mask），逼近真实场景。
3. **接量化**：把线性层权重做 INT8 量化（Day53），对比 tokens/s 与显存占用。
4. **加投机解码**：用一个更小的草稿模型（Day55）一次猜多个 token，验证加速。
5. **接服务化**：把引擎塞进 Day56 的调度器，做成能并发接请求的 serving。

---

## 常见问题

- **Q：为什么生成的 token 看起来是乱的？**
  A：这是随机初始化的玩具模型，没有训练，输出无语义。本天目标是跑通引擎骨架与性能构成，不是文本质量。想要有意义的输出需加载真实预训练权重。

- **Q：为什么 decode 比 prefill 慢那么多（总时间上）？**
  A：prefill 一次并行处理整段 prompt，decode 要串行走 `max_new-1` 步、每步一遍全部层。decode 的串行性正是 LLM 推理优化（KV Cache、投机解码、continuous batching）的主战场。

- **Q：CPU 上的 tokens/s 能代表 GPU 吗？**
  A：不能。绝对数值仅供无 GPU 环境理解流程；GPU 上吞吐会高很多，且 batch 越大优势越明显。

---

## 延伸阅读

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [vLLM 论文：PagedAttention](https://arxiv.org/abs/2309.06180)
- [How to generate text（HuggingFace 采样策略详解）](https://huggingface.co/blog/how-to-generate)
- [nanoGPT（极简 GPT 实现，很好的参考）](https://github.com/karpathy/nanoGPT)
