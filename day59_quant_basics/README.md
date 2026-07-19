# Day 59: 量化基础 — 对称/非对称、per-tensor/per-channel

> 🧩 **第十一阶段（模型压缩与部署）开篇**。承接 [Day53 推理量化](../day53_inference_quant/) 与 [Day24 TensorRT](../day24_tensorrt_intro/)。Day53 我们初步写了 INT8 对称权重量化，本天把「量化」拆到最底层，讲透四个正交的设计维度。

## 学习目标

1. **推导** 对称量化（scale）与非对称量化（scale + zero_point）的量化/反量化公式
2. **实现** 四种量化：对称/非对称 × per-tensor/per-channel
3. **对比** 在带离群值的数据上，per-tensor 与 per-channel 的误差差异
4. **理解** 为什么权重常用对称、激活常用非对称，以及离群值为何毒化 per-tensor

---

## 前置依赖

- **Day53 推理量化**：本天是其深化，Day53 只讲了对称 per-tensor/per-channel
- **基础线性代数**：量化本质是把浮点区间线性映射到整数区间
- **无需 CUDA C++**：本阶段以 PyTorch 为主，有 GPU 用 GPU，无 GPU 用 CPU/NumPy 真实运行

---

## 核心概念

### 对称量化（权重常用，零点固定为 0）

```
scale = max(|x|) / qmax            (8bit: qmax = 2^7 - 1 = 127)
q     = clamp(round(x / scale), -127, 127)      # 存 INT8
x_hat = q * scale                               # 反量化
```

### 非对称量化（激活常用，零点可偏移）

```
scale = (max - min) / (2^bits - 1)              (8bit: 分母 = 255)
zp    = round(0 - min / scale)                  # 浮点 0 映射到的整数
q     = clamp(round(x / scale) + zp, 0, 255)
x_hat = (q - zp) * scale
```

数轴直观（非对称把浮点区间 `[min, max]` 线性铺满整数区间 `[0, 255]`）：

```
   浮点  min ────────── 0 ─────────────── max
          │            │                  │
   整数   0 ────────── zp ─────────────── 255
```

- **对称**只需 1 个参数（scale），但若数据单边分布（如 ReLU 后全为正），整数区间的负半段被浪费，误差偏大。
- **非对称**用零点把 0 对齐到整数区间内部，能用满全部台阶，单边分布下误差更小。

### per-tensor vs per-channel

| 粒度 | 参数量 | 优点 | 缺点 |
|------|--------|------|------|
| per-tensor | 整张量 1 组 | 最省事、kernel 简单 | 一个离群值拉大 scale，毒化所有值 |
| per-channel | 每行 1 组 | 隔离离群值，误差小 | 参数稍多，kernel 略复杂 |

**离群值毒化机制**：per-tensor 的 `scale = max(|x|)/127`，一个异常大的权重会把 scale 抬高，导致大量小权重被压到同一整数台阶，量化台阶变粗、误差骤增。per-channel 把离群值限制在它所在的那一行，其他行不受影响。

---

## 文件结构

```
day59_quant_basics/
├── python/quant_basics.py   # 四种量化实现 + 对比表 + 离群值实验
├── build.sh                 # Linux/WSL2 运行脚本
├── build.bat                # Windows 运行脚本
└── README.md                # 本文件
```

---

## 编译与运行

### Linux / WSL2

```bash
cd day59_quant_basics
pip install torch          # 可选：有 NVIDIA GPU 则自动用 GPU
python python/quant_basics.py
```

### Windows

```powershell
cd day59_quant_basics
.\build.bat
```

> 脚本自动检测环境：有 CUDA 用 GPU，无 CUDA 用 CPU，均真实运行完整量化逻辑；连 PyTorch 都没有则用 NumPy 演示等价原理。无 GPU 也能正常运行并 exit 0。

---

## 知识点总结

1. **对称 vs 非对称**：对称零点固定为 0（1 参数），非对称有零点偏移（2 参数）。
2. **零点 zero_point**：非对称量化中「浮点 0 对应的整数」，`zp = round(0 - min/scale)`。
3. **量化范围**：对称 8bit 用 `[-127,127]`，非对称用 `[0,255]`。
4. **per-tensor 毒化**：离群值抬高全局 scale，量化台阶变粗，小值误差骤增。
5. **per-channel 隔离**：每行独立 scale，离群值只影响所在行，是权重量化主流。
6. **选型经验**：权重分布关于 0 对称 → 对称；激活（尤其 ReLU 后）单边 → 非对称。

---

## 验收标准

- 程序打印「量化前后对比表」，你能逐行解释原始值 → 整数台阶 → 反量化值的映射。
- 实验 1 中非对称量化的 MSE 明显小于对称（单边分布场景）。
- 实验 2 中 per-tensor 的 MSE 是 per-channel 的数十倍，显示「通过 ✓」。
- 你能说清离群值毒化 per-tensor 的机制，以及权重/激活的量化选型经验。

---

## 动手挑战

1. **改比特数**：把 `n_bits` 从 8 改成 4，观察量化台阶变粗、误差增大多少。
2. **对称做激活**：用对称量化去量化 ReLU 激活（全正），对比非对称，量化验证浪费的负半段。
3. **per-channel 非对称**：仿照 `sym_quant_per_channel` 写一个非对称 per-channel 版本。
4. **离群值扫描**：把注入的离群值从 6.0 逐步调到 100，画出 per-tensor MSE 随离群值增大的趋势。

---

## 常见问题

- **Q：为什么对称量化用 127 而不是 128？**
  A：INT8 范围是 `[-128,127]`，为保持关于 0 对称、避免 -128 无正对应，通常只用 `[-127,127]`。

- **Q：per-channel 一定沿行吗？**
  A：对 Linear 权重 `[out,in]` 通常沿输出通道（行）；卷积则沿输出通道维。关键是「让共享 scale 的元素分布尽量一致」。

---

## 延伸阅读

- [PyTorch Quantization 文档](https://pytorch.org/docs/stable/quantization.html)
- [Google《Quantizing deep convolutional networks for efficient inference》白皮书](https://arxiv.org/abs/1806.08342)
