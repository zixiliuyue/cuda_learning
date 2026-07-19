# Day 90: 总收官 — AI Infra 能力图谱 + 项目梳理

> 🎓 **全课程收官（第十三阶段 & 90 天终点）**。本天不学新知识，而是把 90 天、13 个阶段的能力系统性收束成一张**AI Infra 能力图谱**，对照三个真实目标岗位盘点技能覆盖，并给出**简历级项目清单**与进阶路线。这既是终点，也是新起点。

## 学习目标

1. **回顾** 13 个阶段的核心能力与代表产出，建立完整知识地图
2. **对照** 训练优化 / 端侧部署 / 推理服务化三个岗位 JD，评估自身技能覆盖
3. **组织**散落的 Day 产出为 6 个可写进简历的项目
4. **规划**结课之后的进阶学习路线

---

## 90 天全景回顾（13 个阶段）

| 阶段 | 天数 | 主题 | 核心能力 |
|------|------|------|----------|
| 一 | Day 1-5 | CUDA 基础 | 执行/内存模型、错误处理、设备查询 |
| 二 | Day 6-12 | 内存优化 | 共享内存/合并访问/占用率/归约/流 |
| 三 | Day 13-20 | 算法实战 | cuBLAS/FFT/排序/Scan/BFS/蒙特卡洛/调优 |
| 四 | Day 21-26 | 框架集成 | PyTorch扩展/Numba/OpenCV/TensorRT/多GPU/AMP |
| 五 | Day 27-30 | 综合项目 | N-Body/光追/推理引擎/阶段总结 |
| 六 | Day 31-35 | 进阶专题 | Flash Attention/cuSPARSE/IPC/内存池/Nsight |
| 七 | Day 36-38 | Python 封装 | pybind11/ctypes/Cython 三种路线 |
| 八 | Day 39-40 | 扩展实战 | NCCL AllReduce / PyTorch CUDA 扩展 |
| 九 | Day 41-47 | Triton 算子 | 融合算子/自动调优/Flash Attn/接入PyTorch |
| 十 | Day 48-58 | LLM 推理优化 | KV Cache/PagedAttn/批处理/采样/投机/服务化 |
| 十一 | Day 59-66 | 模型压缩部署 | 量化(PTQ/QAT/GPTQ)/剪枝/蒸馏/TensorRT |
| 十二 | Day 67-80 | 分布式训练 | DDP/ZeRO/TP/PP/MoE/3D并行/FSDP/profiling |
| 十三 | Day 81-90 | 通信·存储·工程化 | NCCL算法/RDMA/CUTLASS/Checkpoint/GDS/排查 |

**能力主线**：能跑通 → 跑得快 → 会用库 → 融入框架 → 做项目 → 前沿专题 → 交付 Python → 分布式/推理/压缩 → 通信存储工程化。

---

## 三个目标岗位 × 技能覆盖

### JD1 训练优化工程师

| 能力项 | 对应天数 |
|--------|----------|
| 分布式并行（DP/TP/PP/MoE） | Day 67-74 |
| 显存优化（ZeRO/梯度检查点） | Day 69/75 |
| 通信优化（NCCL/overlap） | Day 39/78/81 |
| 混合精度训练（AMP） | Day 26 |
| Profiling 与瓶颈定位 | Day 35/79 |
| 自定义高性能算子（CUDA/Triton） | Day 40-47 |

### JD2 模型端侧部署工程师

| 能力项 | 对应天数 |
|--------|----------|
| 量化（PTQ/QAT/GPTQ/AWQ） | Day 59-62 |
| 剪枝 / 知识蒸馏 | Day 63/64 |
| TensorRT（ONNX/Engine/Plugin） | Day 24/65/66 |
| 推理算子融合 | Day 43-45/54 |
| 精度对齐与验证 | Day 60/66 |
| 端侧/国产芯片适配 | Day 89 |

### JD3 推理服务化工程师

| 能力项 | 对应天数 |
|--------|----------|
| KV Cache / PagedAttention | Day 49/50 |
| Continuous/Dynamic Batching | Day 51/56 |
| 采样与解码策略 | Day 52 |
| 投机解码 | Day 55 |
| Triton 融合算子 | Day 41-47 |
| vLLM/TensorRT-LLM 架构 | Day 57/58 |

> 三个岗位的技能几乎全部被 90 天课程覆盖，且大量交叉复用（Triton 算子 / Flash Attention / 量化三个岗位都吃得到）。

---

## 简历级项目清单

1. **高性能 CUDA 算子库**（Day 6/9/31/40）
   手写并优化 Tiled GEMM / 归约 / Flash Attention / LayerNorm，含 shared memory、合并访问、warp-shuffle 优化，PyTorch 扩展交付。

2. **Triton 融合算子集**（Day 41-47）
   Fused Softmax/LayerNorm/Flash Attention/MLP，`@autotune` 自动调优，接入 `torch.compile`，对比 cuBLAS/手写 CUDA。

3. **迷你 LLM 推理引擎**（Day 48-58）
   整合 KV Cache + PagedAttention + Continuous Batching + 多种采样 + 投机解码，端到端引擎并产出吞吐/延迟报告。

4. **模型压缩与端侧部署链路**（Day 59-66）
   PTQ/QAT 量化 + 剪枝 + 蒸馏，ONNX→TensorRT Engine（FP16/INT8），自定义 Plugin，完整精度对齐验证。

5. **分布式训练框架实践**（Day 67-80）
   DDP/ZeRO/TP/PP/MoE/3D 并行最小可跑实现 + 梯度检查点 + 通信计算重叠 + profiling 体系。

6. **AI Infra 工程化工具箱**（Day 81-90）
   Ring/Tree AllReduce 模拟、Checkpoint 优化、GDS IO 路径分析、NaN 钩子/OOM 排查工具、迷你 autograd 引擎。

> **写简历建议**：每个项目突出量化指标（加速比/显存节省/吞吐提升），并说明用到的优化手段与问题定位过程。

---

## 文件结构

```
day90_capstone/
├── python/capstone.py   # 能力图谱 + JD 对照 + 简历项目 + 进阶路线
├── build.sh             # Linux/WSL2 运行脚本
├── build.bat            # Windows 运行脚本
└── README.md            # 本文件（全课程收官总结）
```

---

## 编译与运行

```bash
cd day90_capstone
python python/capstone.py
```

```powershell
cd day90_capstone
.\build.bat
```

> 纯 Python 打印，任意环境稳定运行。

---

## 结课之后 — 进阶学习路线

1. **深入源码**：读 vLLM / TensorRT-LLM / Megatron-LM / DeepSpeed，从「会用」到「能改」。
2. **真实硬件实战**：多卡/多机跑真正的分布式训练与 RDMA/NVLink，把「单机模拟」升级为「真集群」。
3. **前沿算子**：跟进 FlashAttention-3 / FP8 训练 / MoE kernel / CUTLASS 3.x。
4. **系统工程**：K8s + GPU 调度、弹性训练、故障恢复、可观测性，向平台级 AI Infra 演进。
5. **参与开源**：给主流框架提 PR，是最快的成长与背书方式。

---

## 验收标准

- 你能不看图谱，口述 13 个阶段的主线与各自核心能力。
- 给定任一目标岗位 JD，你能指出自己在课程中对应的产出与项目。
- 你能挑 2-3 个项目，用「量化指标 + 优化手段 + 问题定位」的方式讲清楚。
- 你为自己规划了明确的下一步进阶方向。

---

## 动手挑战

1. **写一份简历**：用本天的 6 个项目模板，结合你的实际产出写一版 AI Infra 简历。
2. **打磨一个项目**：选一个最感兴趣的项目深挖到可开源的程度（含 README + benchmark）。
3. **模拟面试**：针对三个 JD 各准备 5 个高频面试题，用课程知识作答。

---

## 结语

> AI Infra 的核心是**「让算力被高效地用起来」**——算子、并行、通信、存储、工程化，每一环都是杠杆。
>
> 90 天从 `Hello CUDA` 到分布式训练与推理引擎，你已经走完从「写 kernel」到「构建系统」的完整路径。这是终点，更是新的起点。🎉

---

## 延伸阅读

- [vLLM](https://github.com/vllm-project/vllm) · [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) · [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [CUTLASS](https://github.com/NVIDIA/cutlass) · [FlashAttention](https://github.com/Dao-AILab/flash-attention)
