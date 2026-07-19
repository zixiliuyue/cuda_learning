# Day 53: 推理量化 — INT8/FP16 权重量化

> 🗜️ **LLM 推理阶段核心优化之一**。LLM 权重动辄数十亿参数，FP32/FP16 存储与带宽是推理瓶颈。**权重量化**把浮点权重压成低比特整数（INT8），推理时反量化回浮点计算，显存/带宽显著下降，精度损失可控。

## 学习目标

1. **实现** INT8 对称量化 / 反量化（手写 scale 计算与 clamp）
2. **对比** per-tensor 与 per-channel 两种量化粒度的精度差异
3. **验证** 量化前后权重重建误差、层输出误差与「显存」节省
4. **理解** 异常值（outlier）为什么让 per-tensor 精度变差

---

## 前置依赖

- **Day 52 采样**：同属 LLM 推理阶段
- 线性层 `y = W x` 的基本矩阵乘
- 后续 **Day 59-61**（量化基础/PTQ/QAT）会在本天基础上深入

---

## 核心概念

### 对称量化公式

```
量化   : q     = clamp(round(w / scale), -127, 127)   # 存成 INT8
反量化 : w_hat = q * scale                            # 计算时还原为浮点
scale  : 对称量化取 max(|w|) / 127（零点固定为 0）
```

### 两种粒度

| 粒度 | scale 数量 | 特点 |
|------|-----------|------|
| per-tensor | 整个张量 1 个 | 简单，但异常值会拉大 scale，小权重精度损失严重 |
| per-channel | 每个输出通道（行）1 个 | 更贴合各通道范围，误差更小，实践主流 |

### 关键直觉：异常值的危害

per-tensor 用**全局** `max(|w|)` 当 scale，只要有一个异常大的权重，scale 就被撑大，
导致大量小权重被量化到很少的几个整数级别上，精度骤降。per-channel 让每行独立，
异常值只影响它所在的那一行，因此整体误差小得多。

### 显存节省

| 精度 | 每权重字节 | 相对 FP32 |
|------|-----------|-----------|
| FP32 | 4 B | 1x |
| FP16 | 2 B | 2x |
| INT8 | 1 B | 4x |

---

## 文件结构

```
day53_inference_quant/
├── python/inference_quant.py   # INT8 per-tensor/per-channel 量化 + 精度验证
├── build.sh                    # Linux/WSL2/macOS 运行脚本
├── build.bat                   # Windows 运行脚本
└── README.md                   # 本文件
```

> 以 **PyTorch** 为主，`device = "cuda" if torch.cuda.is_available() else "cpu"`：有 CUDA 跑 GPU、无 CUDA 跑 CPU，逻辑都完整真实运行；无 torch 时退化为 NumPy。

---

## 编译与运行

### Linux / WSL2 / macOS

```bash
cd day53_inference_quant
pip install torch
python python/inference_quant.py
```

### Windows

```powershell
cd day53_inference_quant
.\build.bat
```

---

## 知识点总结

1. **对称量化** 零点固定为 0，`scale = max(|w|)/127`，实现简单。
2. **per-channel** 沿输出维度分别计算 scale，误差明显小于 per-tensor。
3. **异常值** 是 per-tensor 精度差的主因，也是 LLM 量化（如 SmoothQuant）要解决的问题。
4. **误差评估** 常用 MSE + 最大绝对误差，同时看权重重建误差与层输出误差。
5. **FP16** 是最简单的“量化”（半精度），2x 节省且几乎无损。
6. **反量化** 在计算前还原为浮点，真正的推理内核会用 INT8 GEMM 直接算。

---

## 验收标准

- 程序打印 per-tensor / per-channel / FP16 的权重重建误差与层输出误差。
- per-channel 的误差明显小于 per-tensor（异常值场景下差距更大）。
- 打印 FP32/FP16/INT8 存储占用，INT8 约为 FP32 的 1/4。
- per-channel 层输出相对最大误差 < 5%，显示「通过 ✓」。
- 无 GPU 时在 CPU 上完整运行，exit code 0。

---

## 动手挑战

1. **非对称量化**：实现带零点 zero-point 的非对称量化，处理非零均值分布。
2. **激活量化**：不仅量化权重，也量化输入激活 x，做 INT8×INT8 的 GEMM 模拟。
3. **outlier 抑制**：把异常权重裁剪或单独用 FP16 保存，观察 per-tensor 误差变化。

---

## 常见问题

- **Q：为什么反量化后还要用浮点算？**
  A：本天为教学用浮点还原。真实高性能推理会用 INT8 Tensor Core 直接做整数矩阵乘，累加到 INT32 再反量化。

- **Q：量化一定掉精度吗？**
  A：会有损失，但 per-channel + 校准（PTQ）后通常可控制在可接受范围；FP16 几乎无损。

---

## 延伸阅读

- [LLM.int8()（异常值分解）](https://arxiv.org/abs/2208.07339)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [PyTorch 量化文档](https://pytorch.org/docs/stable/quantization.html)
