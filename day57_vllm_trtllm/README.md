# Day 57: 认识 vLLM / TensorRT-LLM 架构

> 🚀 **第十阶段（LLM 推理优化）**。Day56 我们做了朴素批处理并看到延迟/吞吐的权衡。本天认识工业界两大主流推理引擎——**vLLM**（运行时调度派）与 **TensorRT-LLM**（编译派）——理解它们各自的核心优化。以**原理 + 可运行的数据结构演示**为主。

## 学习目标

1. **理解** PagedAttention 如何像操作系统分页一样管理 KV Cache，消除内部碎片
2. **量化** 分页相比「连续预分配」节省的显存比例（本例约 80%）
3. **理解** continuous batching 相比 static batching 如何提升 GPU 利用率
4. **对比** vLLM（运行时派）与 TensorRT-LLM（编译派）的架构取舍与关键优化点

---

## 前置依赖

- **Day 49 KV Cache**：理解自回归为何需要缓存 K/V，是 PagedAttention 的前提
- **Day 50 PagedAttention**：本天把分页思想用数据结构量化出来
- **Day 51 Continuous Batching**：本天对比它与 static batching 的利用率
- **Day 56 推理服务化**：本天是对 Day56 权衡问题的工业级解法

---

## 核心概念

### 1. PagedAttention：KV Cache 的「虚拟内存分页」

自回归生成时，每个请求的 KV Cache 随生成长度增长，且**最终长度事先未知**。朴素做法按最大长度为每个请求**连续预分配**，导致短请求浪费大量显存（**内部碎片**）。

PagedAttention 借鉴 OS 虚拟内存：

```
  KV Cache 切成固定大小 block(如 16 token/块)
  每个请求维护 block_table：逻辑块 → 物理块 的映射
  按需分配、用完归还，物理块可被任意请求复用
```

| 方案 | 显存占用 | 碎片 |
|------|----------|------|
| 连续预分配（按 max_len） | 高（为最坏情况买单） | 内部碎片严重 |
| PagedAttention（按块） | 低（只为真实长度买单） | 几乎无碎片 |

> 本天演示中：8 个变长请求，连续预分配浪费 429%，PagedAttention 仅 5%，**节省约 80% 显存**。省下的显存可容纳更多并发请求 → 吞吐进一步提升。

### 2. Continuous Batching：token 级调度

- **Static batching**：整批一起开始、一起结束，短请求算完后 slot 仍被占着**空转**到最长请求结束。
- **Continuous batching**：以 token 为粒度调度，**完成的请求立即退出**，空位被新请求**立刻填补**，GPU 利用率长期接近满载。

> 本天演示中：static 利用率约 73%，continuous 提升到 100%。

### 3. vLLM vs TensorRT-LLM：两条优化路线

| 维度 | vLLM（运行时派） | TensorRT-LLM（编译派） |
|------|------------------|------------------------|
| 核心思想 | 运行时动态调度 | 离线编译成静态引擎 |
| KV Cache | PagedAttention | 连续/分页（较新版本） |
| 批处理 | Continuous batching | In-flight batching |
| 优化时机 | 运行时（Python 灵活） | 编译期（面向具体 GPU） |
| 算子 | PagedAttention CUDA 核 | 深度融合 + 内核自动选择 |
| 精度 | FP16/BF16/FP8/量化 | FP16/BF16/FP8/INT8/INT4 |
| 易用性 | 高（pip 装，HF 权重直用） | 中（需 build engine） |
| 灵活性 | 高 | 低（引擎编好后固定） |
| 典型场景 | 通用在线服务/多变负载 | 固定模型的极致性能部署 |

**一句话**：vLLM 用「聪明的运行时调度」榨吞吐且易用；TensorRT-LLM 用「离线深度编译」榨极致延迟与峰值性能。

---

## 文件结构

```
day57_vllm_trtllm/
├── python/vllm_trtllm.py   # BlockManager(分页) + continuous batching 演示 + 对比表
├── build.sh                # Linux/WSL2 运行脚本
├── build.bat               # Windows 运行脚本
└── README.md               # 本文件
```

> 本天为**纯数据结构原理演示**，在任何环境都真实运行、结果可复现，不依赖真实 vllm/trt。若本机恰好装了 vllm 会做可选探测并打印版本。

---

## 编译与运行

### Linux / WSL2

```bash
cd day57_vllm_trtllm
python python/vllm_trtllm.py          # 原理演示，无需 GPU/vllm
pip install vllm                      # 可选：真实体验（需 NVIDIA GPU）
```

### Windows

```powershell
cd day57_vllm_trtllm
.\build.bat
```

> vLLM/TensorRT-LLM 官方主要支持 Linux + NVIDIA GPU。本天的**原理演示不依赖它们**，任何环境都能跑出一致结果。

---

## 知识点总结

1. **分页消除碎片**：固定 block + block_table 映射，按需分配，显存利用率接近满。
2. **block_table**：逻辑块→物理块的间接映射，是 PagedAttention 的核心数据结构。
3. **前缀共享**：相同前缀的 KV 块可跨请求复用（prefix caching），进一步省显存。
4. **token 级调度**：continuous batching 完成即退出、空位即填补，避免 slot 空转。
5. **两条路线**：vLLM 靠运行时灵活调度；TensorRT-LLM 靠编译期深度融合与量化。
6. **选型**：多变负载/快速迭代选 vLLM；固定模型追极致性能选 TensorRT-LLM。

---

## 验收标准

- 程序打印 PagedAttention 的显存节省比例，你能解释「内部碎片」从何而来。
- 程序打印 continuous vs static 的 GPU 利用率，你能说清 slot 空转的含义。
- 你能填出 vLLM vs TensorRT-LLM 对比表的至少 5 个维度差异。
- 你能说出各自的典型适用场景与一条核心优化点。

---

## 动手挑战

1. **改 block_size**：把 `block_size` 从 16 改成 8/32，观察显存占用与碎片如何变化（block 越大，最后一块的浪费越多）。
2. **加抢占**：让 `BlockManager` 在块耗尽时「换出」最长请求的 KV 到 CPU，模拟 vLLM 的 preemption。
3. **前缀共享**：给多个请求相同前缀，改造 BlockManager 让它们共享前缀块，量化额外节省。
4. **补流量**：在 continuous batching 里，请求完成后随机注入新请求，观察利用率能否长期维持满载。

---

## 常见问题

- **Q：PagedAttention 会不会因为分页而变慢？**
  A：会有轻微间接寻址开销，但专用 CUDA kernel 已高度优化；换来的显存收益（更大并发/更长上下文）远超这点开销。

- **Q：vLLM 和 TensorRT-LLM 能一起用吗？**
  A：可以组合思路，但通常二选一。也有项目（如 TensorRT-LLM 的 in-flight batching）在编译派里引入运行时调度的优点。

- **Q：为什么本天不跑真实 vLLM？**
  A：vLLM 需要 Linux + NVIDIA GPU 且下载大模型权重。本天用数据结构把**核心思想**讲透，等价且跨平台；有条件可自行 `pip install vllm` 体验。

---

## 延伸阅读

- [vLLM 论文：Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [TensorRT-LLM 官方仓库](https://github.com/NVIDIA/TensorRT-LLM)
- [Orca（continuous batching 起源）](https://www.usenix.org/conference/osdi22/presentation/yu)
