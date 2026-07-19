# Day 66: 端侧部署实战 — FP16/INT8 + 精度验证

> 🏁 **第十一阶段（模型压缩与部署）收官**。把本阶段的量化/剪枝/蒸馏/TensorRT 串成一条完整链路，用一个可跑的小模型走一遍 **训练 → 量化(FP16/INT8) → 部署对比 → 精度验证**，形成端侧部署的工程闭环。

## 学习目标

1. **走通** 训练(FP32) → 量化(FP16/INT8) → 部署对比 的完整链路
2. **实现** INT8 对称 per-tensor 量化（PTQ + 校准）的可跑模拟
3. **对比** FP32/FP16/INT8 三种精度的「精度 / 模型大小 / 推理耗时」
4. **执行** 部署前的精度验证，判断量化掉点是否可接受
5. **总结** 第十一阶段（Day 59-66）的知识地图

---

## 前置依赖

- **Day 59-62 量化**：本天的 INT8 校准与量化直接沿用
- **Day 63 剪枝 / Day 64 蒸馏**：压缩三板斧的另外两块
- **Day 65 TensorRT**：本天的「导出 → Engine」链路承接其原理

---

## 核心概念

### 部署三角：精度 vs 大小 vs 速度

| 精度 | 每参数字节 | 相对大小 | 典型速度 | 精度损失 | 使用条件 |
|------|-----------|---------|---------|---------|---------|
| FP32 | 4 B | 1.0x | 1.0x | 无（基线） | 训练/高精度需求 |
| FP16 | 2 B | 0.5x | ~2x | 极小 | 主流部署默认 |
| INT8 | 1 B | 0.25x | ~2-4x | 小（需校准） | 端侧/高吞吐 |

### 端侧部署完整链路

```
训练(FP32) → 量化(FP16/INT8) → 导出(ONNX) → 引擎(TensorRT) → 部署
               │
               ├─ FP16：直接转半精度，几乎无损，一行 flag。
               └─ INT8：PTQ 需校准集统计激活范围求 scale；掉点大时上 QAT。
```

### INT8 对称量化（PTQ）

用校准集前向一遍，统计每个 Linear 输入激活的范围，求对称量化 scale（zero_point=0）：

```python
scale = tensor.abs().max() / 127        # int8 对称，qmax=127
q = clamp(round(x / scale), -128, 127)  # 量化
deq = q * scale                          # 反量化（fake quant，模拟精度损失）
```

权重同理按 per-tensor 量化。掉点大时改 **per-channel**（每输出通道一个 scale）或上 **QAT**。

### 精度验证（部署最后一关）

量化后必须用验证集核对 top-1 / 关键指标，确认掉点在可接受范围（工业界常要求 < 1%）。本脚本对 FP16/INT8 分别打印掉点并给出「通过 / 需关注」结论。

> **说明**：脚本用 PyTorch fake-quant 模拟数值精度，无需 TensorRT。真实的 FP16/INT8 **加速**来自 TensorRT 的专用低精度 kernel；模拟里 INT8 因多了量化算子反而更慢，这不代表真实部署的速度。

---

## 文件结构

```
day66_edge_deploy/
├── python/edge_deploy.py   # 训练->量化->部署对比->精度验证，含无 GPU 模拟
├── build.sh                # Linux/WSL2 运行脚本
├── build.bat               # Windows 运行脚本
└── README.md               # 本文件（含第十一阶段回顾表）
```

> 有 CUDA 用 GPU（真半精度）；无 CUDA 用 CPU（fp16 舍入 + fake-quant 模拟）；无 torch 用 NumPy 演示 INT8 量化数值流程。

---

## 编译与运行

### Linux / WSL2

```bash
cd day66_edge_deploy
python python/edge_deploy.py
```

### Windows

```powershell
cd day66_edge_deploy
.\build.bat
```

> 无 GPU 也能完整跑完，exit code 0。

---

## 知识点总结

1. **部署三角**：精度 / 模型大小 / 推理速度需权衡，端侧优先小而快。
2. **FP16**：字节减半、约 2x 加速、几乎无损，主流部署默认精度。
3. **INT8**：4x 压缩、更高吞吐，但需校准集统计 scale，掉点大时上 QAT。
4. **PTQ 流程**：训练 → 校准（统计激活范围）→ 量化 → 精度验证。
5. **精度验证**：部署最后一道关，掉点须控制在指标可接受范围（常 <1%）。
6. **完整链路**：训练 → 量化 → ONNX 导出 → TensorRT Engine → 端侧推理。

---

## 验收标准

- 程序打印 FP32/FP16/INT8 三精度对比表（精度 / 大小 / 耗时）。
- 模型大小按 4B/2B/1B 呈现 1x / 0.5x / 0.25x 的比例关系。
- 精度验证给出 FP16/INT8 的掉点与「通过/需关注」结论。
- 结尾打印第十一阶段（Day 59-66）回顾表，你能讲清压缩三板斧与部署链路。

---

## 第十一阶段（Day 59-66）模型压缩与部署 · 回顾表

| Day | 主题 | 核心内容 |
|-----|------|---------|
| 59 | 量化基础 | 对称/非对称、per-tensor/per-channel、量化/反量化 |
| 60 | PTQ + 校准 | 训练后量化，用校准集统计范围求 scale |
| 61 | QAT | 量化感知训练，插入伪量化节点让模型适应量化 |
| 62 | 大模型量化 | GPTQ / AWQ / SmoothQuant 权重量化原理 |
| 63 | 剪枝 | 结构化 vs 非结构化 + 剪枝后微调恢复精度 |
| 64 | 知识蒸馏 | teacher→student，软标签 + 温度 T 的 KD loss |
| 65 | TensorRT 深入 | ONNX → Engine → Plugin 完整部署链路 |
| 66 | 端侧部署实战 | FP32/FP16/INT8 对比 + 精度验证（收官） |

> **主线**：量化(59-62) + 剪枝(63) + 蒸馏(64) = 模型压缩三板斧；TensorRT(65) + 端侧部署(66) = 把压缩后的模型高效落地。
> **承接**：Day 24 TensorRT 概念、Day 53 推理量化。
> **下一阶段**：第十二阶段 分布式训练（Day 67-80）。

---

## 动手挑战

1. **per-channel 量化**：把 INT8 权重量化改为每输出通道一个 scale，对比精度提升。
2. **叠加压缩**：先对模型做 Day 63 的剪枝，再走本天量化，观察大小/精度的复合收益。
3. **真实引擎**：在有 GPU + TensorRT 的机器上，用 Day 65 骨架把模型构建成 FP16/INT8 Engine，实测真实加速比。
4. **蒸馏 + 量化**：用 Day 64 蒸馏得到小 student，再量化部署，走完整端侧流程。

---

## 常见问题

- **Q：为什么模拟里 INT8 比 FP32 还慢？**
  A：fake-quant 是「量化+反量化」的额外算子，只为模拟数值精度，不代表真实速度。真实 INT8 加速来自 TensorRT/cuDNN 的 INT8 kernel。

- **Q：INT8 一定要校准吗？**
  A：PTQ 需要校准集统计激活范围。若无标签数据，也可用少量无标签样本；掉点大时用 QAT。

- **Q：FP16 会溢出吗？**
  A：FP16 动态范围较小，激活/梯度过大可能溢出。推理一般安全；训练需配合 loss scaling（见 Day 26 混合精度）。

---

## 延伸阅读

- [NVIDIA INT8 量化白皮书](https://arxiv.org/abs/2004.09602)
- [PyTorch Quantization 文档](https://pytorch.org/docs/stable/quantization.html)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/)
- [Integer Quantization for DL Inference (NVIDIA)](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)
