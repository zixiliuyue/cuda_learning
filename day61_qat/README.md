# Day 61: QAT 量化感知训练入门

> 承接 [Day60 PTQ](../day60_ptq_calibration/)。PTQ 训练完才量化，模型「没料到」自己会被量化；低比特（如 INT4）下精度掉得多。**QAT** 在训练时就插入伪量化节点，让模型主动学习对量化鲁棒的权重。

## 学习目标

1. **理解** 为什么 `round()` 的梯度处处为 0，直接量化训练会梯度断裂
2. **实现** STE（直通估计器）：前向量化引入误差、反向梯度直通
3. **搭建** 带伪量化节点的 `QuantLinear` / `QuantMLP`
4. **对比** 在 INT4 低比特下 PTQ 与 QAT 的最终精度差异

---

## 前置依赖

- **Day60 PTQ**：本天用 PTQ 作为对照基准
- **Day59 量化基础**：对称量化的 scale/round/clamp
- **反向传播 / autograd**：QAT 的核心是自定义 `autograd.Function` 的 backward

---

## 核心概念

### 问题：round 的梯度处处为 0

```
   round(x)
     ^
   3 |          ┌────
   2 |      ┌───┘        台阶之间斜率为 0 -> 梯度=0
   1 |  ┌───┘             -> 权重永远不更新 -> 无法训练
     └──┴───┴───┴──> x
```

量化必须用 `round`（取整），但取整是阶梯函数，梯度几乎处处为 0。若直接把量化放进训练前向，反向传播时梯度被清零，权重纹丝不动。

### 解法：STE 直通估计器

```
  前向： x ──> [ quant -> dequant ] ──> x_hat      （引入真实量化误差）
  反向： grad_x <── [ 视为恒等, 梯度≈1 ] <── grad_x_hat   （直通）
```

- **前向**：老老实实做「量化→反量化」，让模型体验量化误差 `x_hat = round(x/s)*s`。
- **反向**：假装量化节点是恒等函数，梯度直接穿过（`d x_hat / d x ≈ 1`）。
- **饱和裁剪**：被 clamp 到范围外（饱和）的位置，梯度置 0——那里已经饱和，继续推大无意义。

本天用 `torch.autograd.Function` 手写这个前向/反向不一致的节点，这正是 PyTorch/TensorRT QAT 工具背后的机制。

### PTQ vs QAT

| | PTQ | QAT |
|--|-----|-----|
| 何时量化 | 训练结束后 | 训练过程中 |
| 训练成本 | 零（只需校准） | 需额外训练/微调 |
| 精度 | 低比特下掉得多 | 更高，接近 FP32 |
| 适用 | 8bit 常够用 | 4bit/敏感模型必备 |

---

## 文件结构

```
day61_qat/
├── python/qat.py     # FakeQuantize(STE) + QuantMLP + PTQ vs QAT 对比
├── build.sh          # Linux/WSL2 运行脚本
├── build.bat         # Windows 运行脚本
└── README.md         # 本文件
```

---

## 编译与运行

### Linux / WSL2

```bash
cd day61_qat
pip install torch
python python/qat.py
```

### Windows

```powershell
cd day61_qat
.\build.bat
```

> 有 CUDA 用 GPU，无 CUDA 用 CPU，均真实训练；连 PyTorch 都没有则用 NumPy 演示「STE 前向有误差/反向直通」的核心原理（实验 A）。无 GPU 也能 exit 0。

---

## 知识点总结

1. **梯度断裂**：`round` 梯度处处为 0，量化无法直接进入训练。
2. **STE**：前向真量化、反向当恒等，梯度直通，让量化节点可训练。
3. **伪量化节点 (fake quantize)**：quant→dequant，仍是浮点张量，但带量化误差。
4. **饱和位置梯度置 0**：超出量化范围的值梯度截断，稳定训练。
5. **QAT 流程**：先 FP32 预训练 → 打开伪量化 → 继续微调若干 epoch。
6. **收益场景**：INT8 时 PTQ 常已够用；INT4/敏感模型上 QAT 优势明显。

---

## 验收标准

- 实验 A：无 STE 时权重停在 0（梯度断裂），有 STE 时量化值成功逼近 target。
- 实验 B：INT4 下 PTQ 相对 FP32 有可见精度下降，QAT 恢复大部分精度。
- QAT 准确率 ≥ PTQ，显示「通过 ✓」。
- 你能画出 round 的阶梯图并解释 STE 为何能让梯度回流。

---

## 动手挑战

1. **调比特**：把 `n_bits` 改成 8 / 2，观察 PTQ 与 QAT 的差距如何随比特变化。
2. **只量化权重**：修改 `QuantLinear`，只对权重伪量化不量化激活，看精度差异。
3. **learnable scale**：把 scale 变成可学习参数（LSQ 思想），让反向也更新 scale。
4. **BN 折叠**：加一个 BatchNorm 并尝试在量化前把它折叠进 Linear 权重。

---

## 常见问题

- **Q：QAT 一定比 PTQ 好吗？**
  A：精度上通常是，但要额外训练成本。INT8 时 PTQ 往往已够用，没必要上 QAT。

- **Q：STE 反向真的用梯度=1 合理吗？**
  A：这是一种有偏但实践极有效的近似。也有 LSQ、PACT 等改进让 scale/clip 阈值可学习，进一步提升精度。

- **Q：为什么用 INT4 演示？**
  A：INT8 量化误差小，PTQ 与 QAT 差距不明显；INT4 误差大，能清晰放大 QAT 的收益。

---

## 延伸阅读

- [Bengio et al.《Estimating or Propagating Gradients Through Stochastic Neurons》(STE 起源)](https://arxiv.org/abs/1308.3432)
- [Jacob et al.《Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference》(QAT 经典)](https://arxiv.org/abs/1712.05877)
- [PyTorch Quantization-Aware Training 教程](https://pytorch.org/docs/stable/quantization.html#quantization-aware-training)
