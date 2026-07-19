# Day 55: 投机解码（Speculative Decoding）入门

> 🚀 **LLM 推理提速的明星技术**。大模型自回归每步只出 1 个 token，串行且慢。投机解码用便宜的**小模型（draft）**先猜 K 个 token，**大模型（target）一次并行验证**，在**不改变输出分布**的前提下换取 2~3x 加速。vLLM / TensorRT-LLM 均已内置。

## 学习目标

1. **理解** 「draft 草稿 + target 验证」的投机解码流程
2. **实现** 接受-拒绝采样：接受概率 `min(1, p_target/p_draft)`，拒绝时从修正分布重采样
3. **验证** 投机解码的输出分布严格等于 target 自回归分布（KL≈0）
4. **模拟** 加速比：平均每次 target 前向产出多少 token

---

## 前置依赖

- **Day 52 采样策略**：接受/拒绝、重采样都建立在概率采样之上
- 基础概率：接受-拒绝采样、KL 散度
- 自回归生成的基本概念

---

## 核心概念

### 流程（一轮）

```
1. draft 自回归猜 K 个 token，记录每步 draft 分布 p_draft
2. target 对这 K 个位置一次并行给出分布 p_target（查表即“并行”）
3. 逐个验证第 i 个草稿 token x：
     以概率 min(1, p_target(x)/p_draft(x)) 接受
     否则拒绝：从 (p_target - p_draft)_+ 归一化分布重采样一个，停止本轮
4. 若 K 个全接受：从最后位置的 target 分布再采 1 个 bonus token
```

### 为什么分布不变（关键）

接受-拒绝 + 修正分布重采样，是标准的 **speculative sampling** 构造，可以数学证明：
最终被接受 token 的边缘分布**严格等于** `p_target`。也就是说——

> **draft 只影响速度，不影响正确性。** draft 再差，输出分布也和大模型自回归一致。

### 加速来源

一次 target 前向能**并行验证 K 个位置**，平均产出 `>1` 个 token（取决于 draft 命中率），
因此单位 target 调用产出更多 token，理论加速上限为 `K+1`。

---

## 文件结构

```
day55_speculative_decoding/
├── python/speculative_decoding.py   # 投机解码流程 + 分布一致性 + 加速比模拟
├── build.sh                         # Linux/WSL2/macOS 运行脚本
├── build.bat                        # Windows 运行脚本
└── README.md                        # 本文件
```

> 以 **PyTorch** 为主，`device = "cuda" if torch.cuda.is_available() else "cpu"`：有 CUDA 把转移矩阵放 GPU，无 CUDA 用 CPU，逻辑都真实运行；无 torch 时退化为 NumPy。本天用「马尔可夫玩具模型」（转移概率矩阵）扮演 target/draft，便于清晰验证分布一致性。

---

## 编译与运行

### Linux / WSL2 / macOS

```bash
cd day55_speculative_decoding
pip install torch
python python/speculative_decoding.py
```

### Windows

```powershell
cd day55_speculative_decoding
.\build.bat
```

---

## 知识点总结

1. **draft/target 分工**：小模型猜、大模型验，验证可并行。
2. **接受概率** `min(1, p_target/p_draft)`：draft 高估时可能拒绝。
3. **拒绝重采样** 从 `(p_target - p_draft)_+` 归一化分布采，保证无偏。
4. **bonus token**：全接受时白嫖一个，进一步提速。
5. **分布不变性**：投机解码输出分布 == target 自回归分布（用 KL 验证）。
6. **加速比** 取决于 draft 与 target 的相似度（命中率），上限 `K+1`。

---

## 验收标准

- 打印 target / 投机 / 自回归 三者的单步输出分布，肉眼接近。
- `KL(target || 投机) < 5e-3`，显示「一致 ✓」（分布不变性得证）。
- 加速比模拟中「平均每次 target 调用产出 token 数」> 1。
- 无 GPU 时在 CPU 上完整运行，exit code 0。

---

## 动手挑战

1. **改 K**：把 K 从 4 改成 2 / 8，观察加速比与 draft 命中率的关系。
2. **调 draft 质量**：把 draft 的噪声调小/调大，看接受率如何变化。
3. **贪心版**：把 target/draft 都改成 greedy，实现 exact-match 的 verify（无采样）。

---

## 常见问题

- **Q：draft 很差会怎样？**
  A：接受率低，加速比降到接近 1x，但**输出分布依然正确**——最坏只是没提速。

- **Q：draft 模型从哪来？**
  A：可用同族小模型、蒸馏模型，或 Medusa/EAGLE 这类「自投机」头共享主干。

---

## 延伸阅读

- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al.)](https://arxiv.org/abs/2211.17192)
- [Accelerating LLM Decoding with Speculative Sampling (Chen et al.)](https://arxiv.org/abs/2302.01318)
