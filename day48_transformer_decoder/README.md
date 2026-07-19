# Day 48: 从零搭一个 Transformer Decoder 层

> 🚀 **第十阶段（LLM 推理优化）开篇**。前面用 Triton 学会了写融合算子，从本天起我们把视角抬到「一个能推理的语言模型」。第一步：用 PyTorch 手写 LLM 的最小构成单元——一个 GPT 风格的 Decoder 层。

## 学习目标

1. **实现** 手写 Multi-Head Self-Attention（含 causal mask），不依赖 `nn.MultiheadAttention`
2. **组装** Attention + FFN + 残差 + LayerNorm 成一个完整 Decoder 层（Pre-LN 结构）
3. **验证** 单层 Decoder 前向的输出形状、数值健康度与**因果性**（改未来 token 不影响过去输出）
4. **理解** 「一个 LLM = Embedding + N × DecoderLayer + lm_head」这一整体图景

---

## 前置依赖

- **Day 31 / Day 44 注意力**：本天的 MHA 是它们的组装版，QK^T/√d、softmax、加权求和一脉相承
- **Day 26 混合精度**：LLM 推理常用 fp16/bf16，理解精度有助于后续显存优化
- **线性代数基础**：矩阵乘法与广播，理解 (B, n_head, T, d_head) 的形状变换

---

## 核心概念

### Decoder 层结构（GPT 风格，Pre-LN）

```
  输入 x (B,T,D)
     │
     ├─────────────┐  残差直连
     │        ┌────▼────┐
     │        │ LayerNorm│
     │        └────┬────┘
     │      ┌──────▼───────┐
     │      │ Multi-Head    │  Q,K,V = xW；QK^T/√d_head
     │      │ Self-Attention│  + causal mask（看不到未来）
     │      │  + softmax    │  → 加权求和 V → 输出投影
     │      └──────┬───────┘
     ▼◄────────────┘  x = x + Attn(LN(x))
     │
     ├─────────────┐  残差直连
     │        ┌────▼────┐
     │        │ LayerNorm│
     │        └────┬────┘
     │      ┌──────▼───────┐
     │      │ FFN: Linear   │  d_model → 4*d_model → d_model
     │      │  → GELU       │
     │      └──────┬───────┘
     ▼◄────────────┘  x = x + FFN(LN(x))
   输出 (B,T,D)
```

### Causal Mask：为什么解码器只能看过去

自回归生成时，预测第 `i` 个 token 只能依赖 `0..i` 的信息，不能偷看未来。做法是在注意力分数矩阵 (T,T) 上，把上三角（未来位置）设为 `-inf`，softmax 后这些位置权重变 0：

```
        key位置→  0    1    2    3
  query 0       [ ✓  -inf -inf -inf ]
        1       [ ✓   ✓  -inf -inf ]
        2       [ ✓   ✓   ✓  -inf ]
        3       [ ✓   ✓   ✓   ✓   ]
```

### Pre-LN vs Post-LN

主流 LLM（GPT-2 及之后）采用 **Pre-LN**：`x = x + Sublayer(LN(x))`，梯度更稳定、更好训练。本天即用此结构。

---

## 文件结构

```
day48_transformer_decoder/
├── python/transformer_decoder.py  # 手写 MHA + FFN + Decoder 层，含无 torch 的 NumPy 演示
├── build.sh                       # Linux/WSL2 运行脚本
├── build.bat                      # Windows 运行脚本
└── README.md                      # 本文件
```

> 本阶段以 PyTorch 为主。**PyTorch 在 CPU 也能跑**，因此脚本在无 GPU 时仍会完整运行前向逻辑，只跳过 GPU 专属的显存/耗时统计。

---

## 编译与运行

### 通用（有无 GPU 均可）

```bash
cd day48_transformer_decoder
pip install torch          # 可选；没有 torch 会退化为 NumPy 原理演示
python python/transformer_decoder.py
```

### Windows

```powershell
cd day48_transformer_decoder
.\build.bat
```

---

## 知识点总结

1. **QKV 合并投影**：用一个 `Linear(d_model, 3*d_model)` 一次算出 Q/K/V，比三个独立层更高效。
2. **多头拆分**：`(B,T,D) → (B, n_head, T, d_head)`，各头独立做注意力后再合并。
3. **缩放点积**：除以 `√d_head` 防止分数过大导致 softmax 梯度消失。
4. **causal mask**：上三角置 `-inf`，实现「只看过去」。
5. **残差 + LayerNorm**：Pre-LN 让深层网络训练稳定。
6. **FFN 4× 扩张**：中间维度通常是 `4*d_model`，是模型主要参数量所在。

---

## 验收标准

- 程序打印 Decoder 层 ASCII 结构图，你能逐子层解释数据流。
- 输出形状与输入一致 `(B,T,D)`，显示「一致 ✓」，输出不含 NaN。
- **因果性验证**：修改最后一个 token 的输入后，前缀 token 的输出变化 ≈ 0，显示「符合因果 ✓」。
- 有 GPU 时额外打印前向耗时与峰值显存；无 GPU 时逻辑照跑。

---

## 动手挑战

1. **堆叠多层**：把 `DecoderLayer` 堆 6 层组成一个 mini-GPT，观察输出仍是 `(B,T,D)`。
2. **换激活**：把 FFN 的 `GELU` 换成 `SiLU`（LLaMA 用的），对比数值分布。
3. **加位置编码**：在输入端加 RoPE 或可学习位置编码，思考它如何影响注意力。
4. **打印注意力图**：把某个头的 softmax 权重矩阵打印出来，直观看到下三角结构。

---

## 常见问题

- **Q：没有 GPU 能学吗？**
  A：完全可以。PyTorch 在 CPU 上照样跑完整前向，本天的逻辑与数值验证都不依赖 GPU。

- **Q：为什么用 Pre-LN 而不是原始 Transformer 的 Post-LN？**
  A：Pre-LN 把 LayerNorm 放在子层输入端，残差路径是「干净」的恒等映射，深层训练更稳定，已成 LLM 主流。

- **Q：`nn.MultiheadAttention` 不是现成的吗？**
  A：是，但手写一遍能看清每一步形状变换与 mask 逻辑，为后续 KV Cache / PagedAttention 打基础。

---

## 延伸阅读

- [The Illustrated GPT-2 (Jay Alammar)](https://jalammar.github.io/illustrated-gpt2/)
- [nanoGPT (Andrej Karpathy)](https://github.com/karpathy/nanoGPT)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
