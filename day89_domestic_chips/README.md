# Day 89: 国产 AI 芯片适配概览 — 昇腾 / 寒武纪 / 壁仞

> 🌏 **第十三阶段（通信·存储·工程化）视野拓展专段**。CUDA 生态是 NVIDIA 的护城河，但国产算力适配的需求在快速增长。本天以**原理为主**，梳理三家主流国产 AI 芯片的架构、编程模型，以及从 CUDA 生态迁移的典型路径与工作量。

## 学习目标

1. **了解**昇腾（CANN/AscendCL）、寒武纪（MLU/BANG C）、壁仞的架构与编程模型
2. **建立** CUDA 生态各组件（cuBLAS/cuDNN/NCCL/TensorRT...）→ 国产对应物的映射表
3. **理解**一个 CUDA kernel 迁移到昇腾/寒武纪时「什么变了、什么没变」
4. **掌握**三层迁移策略（框架后端级 / 图编译级 / 算子级）的工作量与覆盖率权衡

---

## 前置依赖

- **Day 87 dispatcher**：框架后端级迁移的技术底座——注册新 dispatch key + kernel
- **Day 39 NCCL / Day 40 CUDA 扩展 / Day 41-47 Triton**：这些正是迁移中「算子级」最费力的部分
- **Day 24/65 TensorRT**：对照国产图编译器（CANN GE / MagicMind）

---

## 核心概念

### 三家芯片速览

| 维度 | 华为昇腾 | 寒武纪 | 壁仞 |
|------|----------|--------|------|
| 代表产品 | Ascend 910/310 | MLU 370/590 | BR100/BR104 |
| 架构 | 达芬奇（Cube 矩阵单元） | MLUarch（私有） | 通用 GPGPU（SIMT+张量核） |
| 软件栈 | CANN | Neuware | 壁仞软件栈 |
| 编程语言 | AscendC / TBE | BANG C（类 CUDA C） | 类 CUDA C |
| 运行时 | AscendCL(ACL) | CNRT | BIRENSUPA |
| 框架适配 | torch_npu / MindSpore | Cambricon PyTorch | 壁仞 PyTorch |
| 编程范式 | 算子+图编译 | 核函数为主 | SIMT，迁移成本低 |

**一句话记忆**：昇腾自成体系（达芬奇+CANN）；寒武纪 BANG C 写 kernel 最像 CUDA；壁仞主打「类 CUDA、迁移成本低」。

### CUDA 生态 → 国产对应物

| 类别 | CUDA | 昇腾 | 寒武纪 | 壁仞 |
|------|------|------|--------|------|
| 编程语言 | CUDA C | AscendC | BANG C | 类 CUDA C |
| 运行时 | CUDA Runtime | AscendCL | CNRT | BIRENSUPA RT |
| DNN 库 | cuDNN | CANN 算子库 | CNNL | 壁仞 DNN |
| 集合通信 | NCCL | HCCL | CNCL | 壁仞 CCL |
| 图/编译 | TensorRT | CANN GE | MagicMind | 壁仞图编译 |
| 框架后端 | torch.cuda | torch_npu | Cambricon PyTorch | 壁仞 PyTorch |

### 迁移中「变与不变」

计算逻辑（如加法）不变，变的是**内存层次管理**与**并行组织方式**：
- CUDA shared memory ≈ 寒武纪 NRAM ≈ 昇腾 Local Buffer；
- 昇腾把编程模型从「线程」上升到「数据分块(Tiling) + 多级缓冲流水」。

### 三层迁移策略

| 层次 | 做法 | 工作量 | 覆盖率 |
|------|------|--------|--------|
| ① 框架后端级 | `torch.cuda`→`torch_npu`，`.to('npu')` | 最小 | 高（标准算子） |
| ② 图/编译级 | 导 ONNX → CANN GE / MagicMind 生成引擎 | 中 | 中高 |
| ③ 算子级 | BANG C / AscendC 手写非标准算子 | 大 | 取决于算子数 |

---

## 文件结构

```
day89_domestic_chips/
├── python/domestic_chips.py   # 架构对比 + 生态映射 + 迁移示意 + 策略选择
├── build.sh                   # Linux/WSL2 运行脚本
├── build.bat                  # Windows 运行脚本
└── README.md                  # 本文件
```

> 本天为**纯知识梳理**，不依赖任何国产芯片 SDK（多为 Linux + 特定硬件），任意环境可运行。

---

## 编译与运行

### Linux / WSL2 或 Windows

```bash
cd day89_domestic_chips
python python/domestic_chips.py
```

```powershell
cd day89_domestic_chips
.\build.bat
```

> 若本机装了 `torch_npu`（昇腾适配层），脚本会提示可做真实迁移实验；否则纯知识梳理，稳定运行。

---

## 知识点总结

1. **护城河的本质**：CUDA 的壁垒不只是硬件，更是 cuDNN/NCCL/TensorRT 等成熟生态。
2. **迁移查表**：先在生态映射表定位依赖的 CUDA 组件，再找对应替代物。
3. **通信与编译是重灾区**：NCCL→HCCL/CNCL、TensorRT→GE/MagicMind 往往需重点适配。
4. **越靠框架层越省力**：后端级迁移改动最小，算子级手写最费力。
5. **dispatcher 是底座**：国产后端本质是注册新的 dispatch key + kernel（呼应 Day 87）。
6. **精度对齐必做**：算子实现差异、累加顺序不同会导致数值偏差，迁移后要逐层验证。

---

## 验收标准

- 你能说出三家芯片各自的架构名、编程语言与「一句话定位」。
- 给定一个依赖 NCCL + cuDNN + 自定义 CUDA 算子的项目，你能说出各部分迁移到昇腾的对应物与难度。
- 你能解释为什么「框架后端级」迁移最省力，以及它依赖的技术底座（dispatcher）。
- 你能指出迁移后为什么必须做「精度对齐」验证。

---

## 动手挑战

1. **补全映射表**：查资料补充「随机数库 cuRAND」「稀疏库 cuSPARSE」在三家的对应物。
2. **迁移评估**：拿你在 Day 58 / Day 80 的项目，列出迁移到昇腾时各组件属于三层策略的哪一层。
3. **算子清单**：统计你在 Day 40-47 写过的自定义算子，估算它们在算子级迁移中的工作量。

---

## 常见问题

- **Q：国产芯片能直接跑现有 PyTorch 代码吗？**
  A：标准算子经框架后端（如 torch_npu）大多能直接跑；自定义/融合算子与部分通信策略需额外适配。

- **Q：迁移后性能能对齐 NVIDIA 吗？**
  A：取决于算子成熟度与编译优化。标准网络差距在缩小，但极致优化的自定义算子仍需针对性调优。

- **Q：学 CUDA 对国产芯片有用吗？**
  A：非常有用。并行思想、内存层次、算子融合是通用的；BANG C/AscendC 的片上内存管理与 CUDA shared memory 高度类比。

---

## 延伸阅读

- [华为昇腾 CANN 文档](https://www.hiascend.com/software/cann)
- [寒武纪 BANG C 编程指南](https://www.cambricon.com/)
- [PyTorch 昇腾适配 torch_npu](https://gitee.com/ascend/pytorch)
- [ONNX（跨框架/跨硬件的中间表示）](https://onnx.ai/)
