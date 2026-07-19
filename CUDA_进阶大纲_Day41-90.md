# CUDA 进阶大纲 — AI Infra 专精路线 (Day 41-90)

> **定位**：在前 40 天「通用 CUDA 基础」之上，向 **AI Infra（大模型训练/推理/部署）** 方向专精。
> **前置**：完成 Day 1-40，掌握 CUDA C++ 手写 Kernel、共享内存/合并访问优化、Flash Attention 原理、PyTorch CUDA 扩展、多 GPU + NCCL 入门。
> **目标岗位对齐**：AI Infra 训练优化工程师 / 模型端侧部署 / AI Infra 推理优化工程师。

---

## 设计原则

1. **承接而非重复**：每个新阶段都建立在前 40 天的 CUDA 基础上（Flash Attention→推理算子，AMP→量化，多GPU/NCCL→分布式并行）。
2. **由易到难、由单点到系统**：现代算子写法(Triton) → 单卡推理优化 → 模型压缩部署 → 分布式训练 → 通信/存储/工程化。
3. **代码优先，能不回退就不回退**：优先写可在**单卡 GPU** 上真实运行的代码；确实需要多卡/多机或特殊硬件的（分布式、RDMA、GPUDirect），以**原理为主 + 单机可跑的最小实现/模拟为辅**。
4. **沿用现有 Day 模板**：每天一个独立目录，含 `README.md`（学习目标/前置依赖/核心概念/文件结构/运行/知识点/验收标准/动手挑战/常见问题/延伸阅读）+ 可运行代码。

---

## 扩展路线图总览

| 阶段 | 天数 | 主题 | GPU 需求 | 主要命中岗位 |
|------|------|------|---------|------------|
| **第九阶段：现代算子编程** | Day 41-47 | Triton 融合算子（从 CUDA 过渡到主流写法） | 单卡即可 | 三家通用 |
| **第十阶段：LLM 推理优化** | Day 48-58 | KV Cache、PagedAttention、量化、服务化 | 单卡即可 | 推理优化 / 端侧 |
| **第十一阶段：模型压缩与部署** | Day 59-66 | 量化/剪枝/蒸馏 + TensorRT 深入 | 单卡即可 | 端侧部署 |
| **第十二阶段：分布式训练** | Day 67-80 | 数据/张量/流水线/MoE 并行、ZeRO、Megatron/DeepSpeed | 单卡演示+多卡最佳 | 训练优化 |
| **第十三阶段：通信·存储·工程化** | Day 81-90 | NCCL 算法、RDMA/NVLink、Checkpoint/存储、troubleshooting、PyTorch 底层 | 部分需多机(原理层) | 训练优化 |

---

## 第九阶段：现代算子编程 — Triton (Day 41-47)

> **阶段目标**：CUDA C++ 写融合算子太重，Triton 是业界主流（JD3 明确要求）。这是从"手写 CUDA"到"高效写融合算子"的关键桥梁。
> **承接**：对照 Day 2（向量加法）、Day 6（Tiling）、Day 9（归约）、Day 31（Flash Attention）的 CUDA 版本，用 Triton 重写并对比。

### Day 41: Triton 入门 — 编程模型 vs CUDA
- **目标**: 理解 Triton 的 program/block 编程模型与 CUDA thread/block 的异同；写出第一个 Triton kernel
- **实战**: Triton 版向量加法 + Softmax，与 Day2/Day31 的 CUDA 版对照

### Day 42: Triton 矩阵乘与自动调优
- **目标**: 用 Triton 写分块矩阵乘，掌握 `@triton.autotune` 自动搜索 block 配置
- **实战**: Triton GEMM，性能对比 cuBLAS（Day13）与手写 Tiled（Day6）

### Day 43: Triton 融合算子 — Fused Softmax / LayerNorm
- **目标**: 掌握"一次读入、片上完成多步计算"的融合思想，减少访存往返
- **实战**: Fused Softmax、Fused LayerNorm，对照 common/ 的 CUDA 版

### Day 44: Triton Flash Attention
- **目标**: 用 Triton 实现 tiled + online softmax 注意力，体会比 CUDA 手写简洁多少
- **实战**: Triton Flash Attention，对照 Day31

### Day 45: 算子融合进阶 — Fused MLP / GELU+Linear
- **目标**: 理解 Transformer FFN 中的融合机会，减少中间张量落地
- **实战**: Fused GELU+Linear、Fused Bias+Dropout

### Day 46: Triton 接入 PyTorch
- **目标**: 把 Triton kernel 封装成 `torch.autograd.Function` / custom op，接入 `torch.compile`
- **实战**: 用 Triton 算子替换 nn.Module 中的标准算子并验证梯度

### Day 47: 阶段小结 — Triton vs CUDA vs cuBLAS
- **目标**: 系统对比三种算子实现方式的开发效率与性能
- **实战**: 综合性能对比报告

---

## 第十阶段：LLM 推理优化 (Day 48-58)

> **阶段目标**：从"一个 Transformer 层"逐步搭到"能服务的推理引擎"，命中 JD3 核心。这是当前课程最大空白。
> **承接**：Day31/44 的注意力、Day26 的混合精度、Day29 的推理引擎雏形。

### Day 48: 从零搭一个 Transformer Decoder 层
- **目标**: 手写 Multi-Head Attention + FFN 前向，理解 LLM 推理的基本单元
- **实战**: 单层 Decoder 前向（GPU）

### Day 49: KV Cache 原理与实现
- **目标**: 理解自回归生成为何需要 KV Cache，如何避免重复计算
- **实战**: 带 KV Cache 的逐 token 生成

### Day 50: PagedAttention（vLLM 核心思想）
- **目标**: 理解 KV Cache 显存碎片问题与分页管理
- **实战**: 简化版分页 KV Cache 管理器

### Day 51: Continuous / Dynamic Batching
- **目标**: 理解变长请求如何动态组批以提升吞吐
- **实战**: 请求调度 + 动态批处理模拟

### Day 52: 采样与解码策略
- **目标**: 实现 greedy / top-k / top-p / temperature 采样
- **实战**: GPU 采样 kernel

### Day 53: 推理量化 — INT8/FP16 权重量化
- **目标**: 掌握权重量化流程与推理精度对齐
- **实战**: 量化线性层 + 精度验证

### Day 54: 算子融合在推理中的应用
- **目标**: Fused QKV 投影、Fused LayerNorm+Residual 减少 kernel 启动
- **实战**: 融合算子加速单层推理

### Day 55: 投机解码（Speculative Decoding）入门
- **目标**: 理解小模型草稿 + 大模型验证的加速原理
- **实战**: 简化投机解码流程

### Day 56: 推理服务化 — 并发调度与低延迟
- **目标**: 理解 serving 的请求队列、批处理、延迟/吞吐权衡
- **实战**: 简易 serving（请求队列 + 批处理调度）

### Day 57: 认识 vLLM / TensorRT-LLM 架构
- **目标**: 读懂主流推理引擎的架构设计与关键优化
- **实战**: 跑通并 profiling 分析（原理为主）

### Day 58: 阶段项目 — 迷你 LLM 推理引擎
- **目标**: 整合 KV Cache + 批处理 + 采样，搭端到端推理引擎
- **交付物**: 迷你推理引擎 + 性能报告

---

## 第十一阶段：模型压缩与部署 (Day 59-66)

> **阶段目标**：补齐量化/剪枝/蒸馏实操 + TensorRT 完整链路，命中 JD2「端侧部署」新增重点。
> **承接**：Day24 TensorRT 概念、Day53 推理量化。

### Day 59: 量化基础 — 对称/非对称、per-tensor/per-channel
- **实战**: 手写量化/反量化 kernel

### Day 60: PTQ 训练后量化 + 校准
- **实战**: INT8 校准 + 精度评估

### Day 61: QAT 量化感知训练入门
- **实战**: 伪量化节点训练

### Day 62: 大模型量化 — GPTQ / AWQ / SmoothQuant 原理
- **实战**: 复现一种权重量化方法

### Day 63: 剪枝 — 结构化 vs 非结构化
- **实战**: 剪枝 + 微调恢复精度

### Day 64: 知识蒸馏
- **实战**: teacher→student 蒸馏

### Day 65: TensorRT 深入 — ONNX→Engine→Plugin
- **实战**: 自定义 TensorRT 插件（接 Day24）

### Day 66: 端侧部署实战 — FP16/INT8 Engine + 精度验证
- **交付物**: 完整部署链路

---

## 第十二阶段：分布式训练 (Day 67-80)

> **阶段目标**：命中 JD1「训练优化」核心，权重最高。单卡先做"并行原理的最小可跑实现"，有多卡则做真并行。
> **承接**：Day25 多 GPU、Day39 NCCL AllReduce。

### Day 67: 分布式基础 — 进程组与集合通信
- **实战**: torch.distributed 起步（单机多进程可跑）

### Day 68: 数据并行 DDP
- **实战**: DDP 训练 + 梯度同步

### Day 69: ZeRO-1/2/3 显存优化
- **实战**: DeepSpeed ZeRO 实操（原理 + 配置）

### Day 70: 张量并行（Tensor Parallel）
- **实战**: 手写 Column/Row 并行 Linear

### Day 71: 流水线并行（Pipeline Parallel）
- **实战**: micro-batch 流水线调度

### Day 72: 序列并行 / 上下文并行
- **实战**: 长序列切分

### Day 73: MoE 并行与专家路由
- **实战**: 简化 MoE 层 + 路由

### Day 74: 混合并行（3D 并行）
- **实战**: 组合 DP + TP + PP

### Day 75: 梯度检查点 / 激活重计算
- **实战**: 显存换算力

### Day 76: 认识 Megatron-LM 架构
- **实战**: 跑通并读核心代码（原理为主）

### Day 77: 认识 FSDP
- **实战**: FSDP 训练对比 DDP

### Day 78: 通信与计算重叠优化
- **实战**: overlap 优化

### Day 79: 训练性能监控与 profiling 体系
- **实战**: 搭建 profiling + 瓶颈定位

### Day 80: 阶段项目 — 多卡训练一个小模型
- **交付物**: 端到端训练 + 扩展性报告

---

## 第十三阶段：通信·存储·工程化 (Day 81-90)

> **阶段目标**：JD1 通信底层 + 存储专段（目前完全空白）+ 三家看重的工程素养。多机相关以原理 + 单机最小实现呈现。

### Day 81: NCCL 通信算法 — Ring / Tree AllReduce
- **实战**: 手写 Ring AllReduce 模拟

### Day 82: NVLink / RDMA / GPUDirect 原理
- **实战**: 带宽拓扑测量

### Day 83: CUTLASS 入门 — GEMM 模板库
- **实战**: 用 CUTLASS 写 GEMM（接前 40 天大纲遗留项）

### Day 84: Checkpoint 机制与性能优化
- **实战**: 异步/分片 checkpoint

### Day 85: 存储 — 并行文件系统 / 小文件 / 数据加载瓶颈
- **实战**: 数据 pipeline 优化

### Day 86: GPUDirect Storage 原理
- **实战**: IO 路径分析（原理为主）

### Day 87: PyTorch 底层 — Autograd 引擎 / C10 / dispatcher
- **实战**: 读源码 + 自定义 backend 概念

### Day 88: Linux 工程化 — coredump / GPU 异常 / compute-sanitizer
- **实战**: troubleshooting 实战

### Day 89: 国产 AI 芯片适配概览（昇腾/寒武纪/壁仞）
- **实战**: 迁移思路 + 生态对比（原理为主）

### Day 90: 总收官 — AI Infra 能力图谱 + 项目梳理
- **交付物**: 综合复盘 + 简历级项目清单

---

## 实施顺序

按见效速度与硬件可行性，落地顺序为：

1. **第九阶段 Triton (Day 41-47)** ← 优先，单卡可跑、三家通用
2. **第十阶段 LLM 推理 (Day 48-58)** ← 次之，单卡可跑、命中推理岗核心
3. 第十一阶段 模型压缩部署 (Day 59-66)
4. 第十二阶段 分布式训练 (Day 67-80)
5. 第十三阶段 通信·存储·工程化 (Day 81-90)

---

## 环境依赖补充

新阶段在原 `requirements.txt` 基础上新增（GPU 环境）：

```
triton>=2.1.0          # 第九阶段：Triton 算子（需 Linux + NVIDIA GPU；Windows 建议 WSL2）
torch>=2.2.0           # 第十~十三阶段：推理/训练主框架
transformers>=4.38.0   # 第十阶段：模型结构参考（可选）
```

> **平台说明**：Triton 官方主要支持 Linux。Windows 用户建议使用 WSL2 + CUDA。无 GPU 时，Triton/分布式相关代码以"原理讲解 + 代码骨架"形式学习，不强制回退。

---

*CUDA 90-Day Learning Project · AI Infra 专精扩展 · 承接前 40 天通用 CUDA 基础*
