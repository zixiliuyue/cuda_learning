# CUDA 40天学习项目 — 阶段自检 Checklist

> 每个阶段结束时用本清单自评。只有在能对每一条给出**肯定回答并能动手复现**时，才建议进入下一阶段。
> 打勾方式：把 `[ ]` 改为 `[x]`。配合每天 README 的「验收标准」使用。

---

## 第一阶段：CUDA 基础与核心概念 (Day 1-5)

- [ ] 我能独立编写一个 `__global__` Kernel 并用 `<<<grid, block>>>` 正确启动。
- [ ] 我能用 `blockIdx * blockDim + threadIdx` 推导任意线程的全局索引（一维与二维）。
- [ ] 我理解 Host 与 Device 内存的区别，能正确使用 `cudaMalloc / cudaMemcpy / cudaFree`。
- [ ] 我知道为什么要在 Kernel 启动后做错误检查，并会用 `CUDA_SAFE_CALL` 宏。
- [ ] 我能用 `cudaGetDeviceProperties` / `cudaDeviceGetAttribute` 读出自己 GPU 的 SM 数、共享内存、带宽。
- [ ] 我理解 Kernel 是**异步**的，知道 `cudaDeviceSynchronize()` 的作用。

**通过标准**：能从零写出并跑通向量加法，且带完整错误检查。

---

## 第二阶段：内存优化与高级特性 (Day 6-12)

- [ ] 我能解释共享内存为何比全局内存快，并能用 Tiling 改写矩阵乘法。
- [ ] 我能说出"合并访问"的定义，并能判断一段代码是否合并访问。
- [ ] 我理解占用率（Occupancy）的含义，知道寄存器/共享内存如何限制它。
- [ ] 我能写出共享内存树形归约，并知道 `__shfl_down_sync` 的 warp 级归约更快。
- [ ] 我能用多个 CUDA 流实现"拷贝-计算"重叠，并用 Event 精确计时。
- [ ] 我了解常量内存、纹理内存、动态并行、CUDA Graphs 各自的适用场景。

**通过标准**：能把一个朴素 Kernel 通过共享内存/合并访问优化并测出加速比。

---

## 第三阶段：算法实战与优化技巧 (Day 13-20)

- [ ] 我能调用 cuBLAS 做矩阵乘法，并与手写 Kernel 对比性能。
- [ ] 我能用 cuFFT 做 FFT/IFFT，并理解频域滤波流程。
- [ ] 我能实现 2D 卷积（高斯模糊 / Sobel）并处理边界。
- [ ] 我理解 Bitonic Sort / Blelloch Scan 的并行思想。
- [ ] 我能用 CSR 表示图并实现并行 BFS 前沿扩展。
- [ ] 我能用 cuRAND 做蒙特卡洛模拟，并会用 Roofline / Nsight 指标定位瓶颈。

**通过标准**：能独立选择合适的库或算法解决一个真实并行问题并调优。

---

## 第四阶段：主流框架集成与项目 (Day 21-26)

- [ ] 我能为 PyTorch 编写自定义 CUDA 算子并支持自动求导。
- [ ] 我能用 Numba 的 `@cuda.jit` 在 Python 中写 Kernel，并说出它与原生 CUDA 的对应关系。
- [ ] 我了解 OpenCV CUDA 模块（`GpuMat`）的基本用法。
- [ ] 我理解 TensorRT 的工作流与 FP16/INT8 量化概念。
- [ ] 我掌握多 GPU 的 `cudaSetDevice` 与数据切分策略。
- [ ] 我理解混合精度训练（AMP）与 FlashAttention 的基本原理。

**通过标准**：能把一个手写 CUDA Kernel 集成进 PyTorch 并验证结果正确。

---

## 第五阶段：综合实战与拓展 (Day 27-30)

- [ ] 我完成了 N-Body 模拟，并用共享内存 Tiling 做了优化。
- [ ] 我完成了简易 GPU 光线追踪渲染器。
- [ ] 我完成了简易推理引擎，手写了 Conv2D/ReLU/MaxPool/Linear/Softmax 中的关键算子。
- [ ] 我为至少一个项目产出了性能报告（含加速比与瓶颈分析）。
- [ ] 我能回顾前 30 天，说清每个阶段解决的核心问题。

**通过标准**：至少独立完成一个综合项目并能讲清优化过程。

---

## 第六阶段：进阶专题 (Day 31-35)

- [ ] 我能解释 Flash Attention 的 Tiling + Online Softmax 如何把显存从 O(N²) 降到 O(N)。
- [ ] 我了解稀疏矩阵格式（CSR/COO）并能用 cuSPARSE 做 SpMV。
- [ ] 我理解 CUDA IPC 如何在多进程间共享显存。
- [ ] 我理解频繁 `cudaMalloc/cudaFree` 的开销，并能实现一个简单内存池。
- [ ] 我能用 Nsight Systems/Compute 做时间线与 Kernel 级深度分析。

**通过标准**：能读懂 Nsight 报告并据此提出具体优化措施。

---

## 第七阶段：Python 封装 (Day 36-38)

- [ ] 我能用 pybind11 把 CUDA Kernel 封装成 Python 模块。
- [ ] 我能用 ctypes 调用导出为 C 风格 API 的 CUDA 动态库。
- [ ] 我能用 Cython 包装 CUDA C 风格函数。
- [ ] 我能对比 pybind11 / ctypes / Cython 三种方式的易用性、性能开销与适用场景（见大纲对比表）。

**通过标准**：能根据项目场景选择合适的 Python↔CUDA 封装方式并说明理由。

---

## 第八阶段：扩展实战 (Day 39-40)

- [ ] 我理解 NCCL AllReduce 的作用，知道它是数据并行训练的通信底座。
- [ ] 我知道 NCCL 官方仅支持 Linux，Windows 上为 CPU 模拟版本。
- [ ] 我能实现带前向 + 反向的 PyTorch 自定义 CUDA 算子并与 PyTorch 交叉验证。
- [ ] 我能把扩展打包为可 `pip install` 的形式。

**通过标准**：能独立交付一个带前向/反向、经数值校验的 PyTorch CUDA 扩展。

---

# 进阶阶段（AI Infra 专精 · Day 41-90）

> 以下阶段对应 [CUDA_进阶大纲_Day41-90.md](CUDA_进阶大纲_Day41-90.md)。已落地：第九、第十阶段。

## 第九阶段：现代算子编程 — Triton (Day 41-47)

- [ ] 我能说清 Triton 的 program/BLOCK/mask 与 CUDA 的 thread/block/if 边界的对应关系。
- [ ] 我能用 Triton 写向量加法、Softmax，并与 PyTorch 交叉验证通过。
- [ ] 我理解 `@triton.autotune` 如何自动搜索 block 配置，并能对比 Triton GEMM 与 cuBLAS。
- [ ] 我能用 Triton 写融合算子（Fused Softmax / LayerNorm / GELU+Linear）并说清融合省了几次访存。
- [ ] 我能用 Triton 实现 Flash Attention（tiled + online softmax），并对照 Day31 CUDA 版体会简洁性。
- [ ] 我能把 Triton kernel 封装成 `torch.autograd.Function`（含 backward）接入 PyTorch。

**通过标准**：能独立用 Triton 写一个融合算子、通过梯度校验并接入 PyTorch。

---

## 第十阶段：LLM 推理优化 (Day 48-58)

- [ ] 我能从零手写一个 Transformer Decoder 层（MHA + FFN + causal mask + residual + layernorm）。
- [ ] 我理解 KV Cache 为何能把自回归生成从 O(T²) 降到 O(T)，并能实现它。
- [ ] 我理解 PagedAttention 如何用分页管理解决 KV Cache 显存碎片问题。
- [ ] 我能说清 static batching 与 continuous batching 在吞吐/延迟上的差异。
- [ ] 我能实现 greedy / temperature / top-k / top-p 采样。
- [ ] 我理解推理量化（INT8/FP16）流程与精度对齐方法（per-tensor vs per-channel）。
- [ ] 我理解投机解码（draft + target 验证）的加速原理与分布不变性。
- [ ] 我能整合 KV Cache + 采样 + 批处理，搭一个端到端的迷你 LLM 推理引擎。

**通过标准**：能独立搭出带 KV Cache + 采样的迷你推理引擎并产出 tokens/s 性能报告。

---

## 第十一阶段：模型压缩与部署 (Day 59-66)

- [ ] 我能手写对称/非对称量化，并说清 per-tensor 与 per-channel 在离群值下的差异。
- [ ] 我理解 PTQ 校准（min/max、百分位、KL 熵校准）如何确定量化范围。
- [ ] 我理解 QAT 的伪量化节点与 STE 直通估计器，知道它为何比 PTQ 精度高。
- [ ] 我了解 GPTQ / AWQ / SmoothQuant 处理大模型激活离群值的核心思想。
- [ ] 我能做结构化与非结构化剪枝，并通过微调恢复精度。
- [ ] 我能用温度 + KL 散度实现知识蒸馏，让 student 学到 teacher 的软标签。
- [ ] 我理解 PyTorch→ONNX→TensorRT Engine→Plugin 的完整部署链路。
- [ ] 我能对比 FP32/FP16/INT8 的模型大小、推理耗时与精度并做验证。

**通过标准**：能把一个模型走完「训练→压缩→导出→部署」链路并产出精度/大小/耗时对比报告。

---

## 第十二阶段：分布式训练 (Day 67-80)

- [ ] 我理解 rank/world_size/进程组，能说清 allreduce/broadcast/allgather/reduce_scatter 的语义。
- [ ] 我理解数据并行 DDP 的梯度 allreduce，知道它与单卡大 batch 数学等价。
- [ ] 我能说清 ZeRO-1/2/3 分别切分优化器状态/梯度/参数如何降低单卡显存。
- [ ] 我能实现 Column/Row Parallel Linear（张量并行），并验证与单卡结果一致。
- [ ] 我理解流水线并行的 micro-batch 调度（GPipe / 1F1B）与气泡开销。
- [ ] 我了解序列并行、MoE 专家并行、3D 混合并行的划分思路。
- [ ] 我理解梯度检查点（激活重计算）的显存-算力权衡。
- [ ] 我了解 Megatron-LM 的 TP 切分方案与 FSDP 的参数分片，能对比 DDP/FSDP 显存。
- [ ] 我理解通信与计算重叠（梯度分桶）如何提升扩展效率，并能搭训练 profiling。

**通过标准**：能用单进程模拟多 rank 的 DDP 端到端训练一个小模型并产出扩展性报告。

---

## 第十三阶段：通信·存储·工程化 (Day 81-90)

- [ ] 我理解 Ring AllReduce 的两阶段（reduce-scatter + all-gather）与 2(N-1)/N 带宽最优性。
- [ ] 我了解 PCIe/NVLink/NVSwitch 拓扑与 RDMA/GPUDirect 零拷贝原理。
- [ ] 我理解 CUTLASS 的分层 GEMM（ThreadBlock/Warp/Thread tile）模板思想。
- [ ] 我能实现同步/异步/分片/增量 Checkpoint，并说清异步如何减少训练阻塞。
- [ ] 我理解并行文件系统、小文件问题与数据加载流水线（prefetch/多 worker）优化。
- [ ] 我理解 GPUDirect Storage 如何绕过 CPU bounce buffer。
- [ ] 我能写一个迷你 autograd 引擎，理解 PyTorch 的 C10 与 dispatcher 机制。
- [ ] 我掌握 GPU 问题排查（OOM/非法访存/NaN/hang）与 compute-sanitizer/cuda-gdb 用法。
- [ ] 我了解国产 AI 芯片（昇腾/寒武纪/壁仞）的编程模型与 CUDA 迁移路径。

**通过标准**：能读懂通信/存储瓶颈并提出优化，具备 AI Infra 工程化 troubleshooting 能力。

---

*配合 CUDA 90 天课程使用（40 天基础 + 50 天 AI Infra 进阶）· 每阶段自评，稳扎稳打*
