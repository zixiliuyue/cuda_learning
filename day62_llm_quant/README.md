# Day 62: 大模型量化 — GPTQ / AWQ / SmoothQuant 原理

> 🧩 **量化专题收官**。承接 [Day59-61](../day59_quant_basics/)。前三天在小模型上打牢量化基础，本天进入 LLM 量化：核心矛盾是**激活离群通道 (outlier channels)**——个别特征维度的激活值极大，直接量化会毒化整体精度。三种主流方法 SmoothQuant / AWQ / GPTQ 各有巧思。

## 学习目标

1. **理解** LLM 激活离群通道为何让直接量化失效
2. **推导** SmoothQuant 的数学等价性 `XW^T = (X/s)(Ws)^T` 与平滑因子公式
3. **复现** SmoothQuant 核心步骤并验证量化误差显著下降
4. **对比** SmoothQuant / AWQ / GPTQ 三者的量化对象、思想、速度与精度

---

## 前置依赖

- **Day59 量化基础**：对称量化、per-tensor/per-channel
- **Day60 PTQ 校准**：本天量化都在训练后进行（PTQ 范畴）
- **Day53 推理量化**：权重量化的显存收益

---

## 核心概念

### 痛点：激活离群通道

```
   激活 X（按通道看幅度）：
     通道:  0    1    2 *离群*  3    4    5
     |X|:  1.2  0.9  ██████89   1.1  0.8  1.0
                      ↑ 一个巨大的离群通道撑大 per-tensor scale
                        -> 其余通道被压到极少几个整数台阶 -> 精度崩溃
```

Transformer 的激活里存在系统性的离群通道（GPT-3/OPT 等尤为明显），而**权重分布相对平稳**。这一「激活难量化、权重好量化」的不对称，正是三种方法的出发点。

### SmoothQuant：把难度从激活迁移到权重

关键在于线性层的一个**数学恒等式**（s 为 per-channel 向量，按输入通道广播）：

```
   Y = X · W^T = (X / s) · (W · s)^T = X' · W'^T
```

- 激活离群通道 `j` 除以较大的 `s_j` → 被压平，变得好量化；
- 代价是权重列 `j` 乘以 `s_j` 变大，但权重原本平稳，扛得住；
- 平滑因子：`s_j = (max_i|X_ij|)^α / (max_k|W_kj|)^(1-α)`，`α`（默认 0.5）控制迁移多少难度到权重侧。
- 目标 **W8A8**：权重和激活都量化成 INT8，可用 INT8 GEMM 加速。

### AWQ：按激活重要性保护权重

- 观察：激活大的输入通道对应的权重更「重要」，量化时应被保护；
- 做法：放大重要通道的权重再量化、量化后除回，等价于给它们**更细的有效台阶**；
- 只量化权重（**W4A16**），激活保持 FP16，无需反缩放激活；
- ⚠️ 关键前提：缩放只在**比 per-channel 更粗**的量化粒度（per-tensor / per-group）下才有效——若权重按列 per-channel 量化，按列缩放会被 scale 精确抵消（本天代码注释有说明）。

### GPTQ：Hessian 误差补偿

- 逐列量化权重，每量化一列，用 **Hessian 二阶信息** 计算它引入的误差，并把误差「转嫁」给尚未量化的列做补偿；
- 精度最高（**W4A16**），但要算 Hessian 及其逆，最慢。

---

## 文件结构

```
day62_llm_quant/
├── python/llm_quant.py   # SmoothQuant/AWQ 复现 + 三者对比表
├── build.sh              # Linux/WSL2 运行脚本
├── build.bat             # Windows 运行脚本
└── README.md             # 本文件
```

---

## 编译与运行

### Linux / WSL2

```bash
cd day62_llm_quant
pip install torch          # 可选，无 torch 会用 NumPy 复现
python python/llm_quant.py
```

### Windows

```powershell
cd day62_llm_quant
.\build.bat
```

> 有 CUDA 用 GPU（矩阵乘法在 GPU 上验证数值一致），无 CUDA 用 CPU；连 PyTorch 都没有则用 NumPy 复现同样的 SmoothQuant/AWQ 数学。无 GPU 也能 exit 0。

---

## 知识点总结

1. **离群通道**：LLM 激活的系统性大值通道，是直接量化失效的元凶。
2. **SmoothQuant 恒等式**：`XW^T=(X/s)(Ws)^T`，量化前重分配难度，数学等价。
3. **平滑因子 α**：α 越大越多难度迁到权重；本天实验里 α 增大时误差单调下降。
4. **AWQ 保护重要权重**：按激活幅度缩放，仅权重量化；需粗于 per-channel 的粒度。
5. **GPTQ 误差补偿**：逐列量化 + Hessian 补偿，精度最高但最慢。
6. **量化对象差异**：SmoothQuant=W8A8，AWQ/GPTQ=W4A16（仅权重）。

---

## 验收标准

- 程序打印激活离群通道 ASCII 图与 SmoothQuant 恒等式，你能解释难度迁移原理。
- 实验中 SmoothQuant 的输出相对误差随 α 增大而下降，最优时约为基线的 1/5，显示「通过 ✓」。
- AWQ 式在适中 α（如 0.3）取得最佳权重量化误差，你能解释过度缩放为何变差。
- 你能填出 GPTQ/AWQ/SmoothQuant 对比表的「量化对象 / 思想 / 速度 / 精度」。

---

## 动手挑战

1. **调离群幅度**：把离群通道倍数从 40 改成 5 / 100，观察 SmoothQuant 收益变化。
2. **per-group 量化**：给 AWQ 实现 group_size=64 的分组量化，看保护效果是否更稳。
3. **迷你 GPTQ**：实现最简版逐列量化 + 误差补偿（`W[:,j+1:] -= err * H_inv[...]`），对比其精度。
4. **真实模型**：用 `transformers` 加载一个小 LLM，统计某个 Linear 层激活的按通道最大值，找出真实离群通道。

---

## 常见问题

- **Q：SmoothQuant 的 s 是训练出来的吗？**
  A：不是。s 用校准集统计的激活/权重最大值离线算出，属于 PTQ，无需梯度训练。

- **Q：为什么 AWQ 用 per-channel 权重量化时缩放没用？**
  A：按列 scale 会把按列缩放精确抵消（`(W·s)/scale` 与 `W/scale'` 等价）。AWQ 的收益依赖 per-tensor/per-group 这种共享 scale 的粗粒度。

- **Q：三者能组合吗？**
  A：能。实践中常先 SmoothQuant 平滑激活，再对权重用 GPTQ/AWQ，兼顾激活与权重量化质量。

---

## 延伸阅读

- [SmoothQuant (Xiao et al., 2022)](https://arxiv.org/abs/2211.10438)
- [AWQ: Activation-aware Weight Quantization (Lin et al., 2023)](https://arxiv.org/abs/2306.00978)
- [GPTQ (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323)
