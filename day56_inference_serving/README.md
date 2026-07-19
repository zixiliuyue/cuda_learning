# Day 56: 推理服务化 — 并发调度与低延迟

> 🚀 **第十阶段（LLM 推理优化）**。模型训练好只是开始，真正对外服务时要面对**延迟**与**吞吐**的经典矛盾。本天用一个「请求队列 + 批处理调度器 + 统计」的简易 serving，把这个权衡量化出来。

## 学习目标

1. **理解** serving 的三大组件：请求队列、批处理调度器、性能统计
2. **实现** 用后台线程真实模拟并发请求到达 + 动态攒批调度
3. **量化** `max_batch` 对平均延迟 / P99 延迟 / 吞吐(req/s) 的影响
4. **说清** 为什么大 batch 提升吞吐却抬高延迟，以及 continuous batching 如何缓解

---

## 前置依赖

- **Day 48-51**：Transformer Decoder、KV Cache、PagedAttention、动态批处理是本天的模型/调度基础
- **Day 10 CUDA Streams**：并发与异步执行的思想在此延伸到「请求级并发」
- **Day 29 推理引擎雏形**：本天把单次推理扩展为可服务的多请求系统

---

## 核心概念

### 延迟 vs 吞吐：serving 的核心矛盾

```
  用户视角：低延迟   —— 我发一个请求，立刻给我结果
  平台视角：高吞吐   —— 同样的 GPU，尽可能多地服务请求

  批处理(batching)：一次矩阵乘服务整批请求
    batch↑ → GPU 利用率↑ → 吞吐↑
    batch↑ → 排队攒批时间↑ → 单请求延迟↑
```

| 组件 | 职责 |
|------|------|
| 请求队列 `Queue` | 缓冲并发到达的请求，削峰填谷 |
| 批处理调度器 | 从队列攒一批（受 `max_batch` / `max_wait` 约束）后统一前向 |
| batched 前向 | 一次矩阵乘服务整批，摊薄固定开销 → 提升吞吐 |
| 统计 | 端到端延迟（avg / P99）与吞吐（req/s） |

### 两个关键旋钮

- **`max_batch`**：攒批上限。越大吞吐越高，但请求等待越久、延迟越高。
- **`max_wait`**：最长攒批等待时间。低流量时给攒批设超时上限，避免空等。

### continuous batching（Day57 详解）

朴素批处理必须「整批一起开始、一起结束」，短请求要陪长请求一起等。vLLM 等引擎用 **continuous batching**：以 token 为粒度调度，新请求可随时插入正在运行的 batch，已完成的请求立即让出资源，从而同时兼顾低延迟与高吞吐。

---

## 文件结构

```
day56_inference_serving/
├── python/inference_serving.py   # 请求队列 + 批处理调度器 + 延迟/吞吐统计
├── build.sh                      # Linux/WSL2 运行脚本
├── build.bat                     # Windows 运行脚本
└── README.md                     # 本文件
```

> 本阶段以 PyTorch 为主：**有 CUDA 用 GPU，无 CUDA 自动用 CPU**，逻辑完全一致；连 PyTorch 都没有时用 NumPy 做等价 batched 运算，原理不打折。

---

## 编译与运行

### Linux / WSL2（推荐，有 NVIDIA GPU）

```bash
cd day56_inference_serving
pip install torch          # 有 GPU 则自动用 GPU
python python/inference_serving.py
```

### Windows

```powershell
cd day56_inference_serving
.\build.bat
```

> 请求的并发到达、队列、调度、统计**在任何环境都真实运行**；只有「模型前向」在无 GPU 时落到 CPU/NumPy。

---

## 知识点总结

1. **队列解耦**：请求到达与处理速率解耦，队列削峰填谷。
2. **动态攒批**：贪心攒批 = 等首个请求 + 时间窗内尽量多收 + 超时/满批即触发。
3. **摊薄开销**：batch 越大，kernel 启动/调度等固定成本被摊得越薄，吞吐越高。
4. **延迟代价**：吞吐提升以排队等待为代价，P99 延迟对攒批尤其敏感。
5. **P99 的意义**：平均延迟会掩盖长尾，服务质量常看 P99/P999。
6. **工程选择**：延迟敏感 → 小 batch + 短 max_wait；吞吐优先 → 大 batch；两者兼顾 → continuous batching。

---

## 验收标准

- 程序打印 `max_batch ∈ {1,4,16,64}` 下的平均延迟 / P99 延迟 / 吞吐对照表。
- 你能解释为何 `max_batch` 增大时吞吐上升、延迟同步上升。
- 无 GPU 时程序照常跑完（exit code 0），你能说清哪些部分真实运行、哪些落到 CPU。
- 你能复述 continuous batching 相比朴素批处理的优势。

---

## 动手挑战

1. **扫 max_wait**：固定 `max_batch=16`，把 `max_wait` 从 1ms 扫到 20ms，观察延迟/吞吐变化。
2. **变流量**：把 `arrival_rate` 调高/调低，看队列长度与延迟如何反应（可打印实时队列长度）。
3. **加 SLA**：给每个请求设一个延迟上限，统计「超时率」，找到满足 SLA 的最大 `max_batch`。
4. **priority 队列**：把普通队列换成优先级队列，让短请求插队，观察 P99 变化。

---

## 常见问题

- **Q：CPU 上的延迟/吞吐能代表 GPU 吗？**
  A：绝对数值不能，但「batch 增大→吞吐升、延迟升」的相对趋势成立，方法论一致。

- **Q：为什么用线程而不是 async？**
  A：线程 + 队列最直观地表达「生产者(请求到达) / 消费者(调度器)」模型；async 也可，原理相同。

- **Q：真实 LLM serving 还差什么？**
  A：变长序列、KV Cache 显存管理、continuous batching、抢占与调度公平性等，Day57/58 继续。

---

## 延伸阅读

- [vLLM: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Orca: 面向 Transformer 的分布式服务系统（continuous batching 起源）](https://www.usenix.org/conference/osdi22/presentation/yu)
- [NVIDIA Triton Inference Server 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/)
