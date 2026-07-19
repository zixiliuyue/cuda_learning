# Day 47: 阶段小结 — Triton vs CUDA vs cuBLAS

> 🧩 **第九阶段（现代算子编程）收官**。Day41-46 用 Triton 重写了向量加法、Softmax、GEMM、Fused LayerNorm、Flash Attention、Fused MLP，并接入了 PyTorch 自动微分。本天做系统对比：对同一操作用 torch(cuBLAS/cuDNN) 与 Triton 分别计时，并总结三种算子实现方式的取舍。

## 学习目标

1. **对比** 手写 CUDA、厂商库（cuBLAS/cuDNN）、Triton 三种实现方式的开发效率与性能
2. **实测** 对 GEMM 与 Softmax 分别用 torch 与 Triton 计时，打印对比表
3. **掌握** 正确的 GPU 计时方法（warmup + `cuda.Event` + 同步）
4. **总结** 第九阶段（Day41-47）的 Triton 核心要点，形成方法论

---

## 前置依赖

- **Day 41-46**：本天是这几天的综合小结
- **Day 13 cuBLAS**：torch 的 GEMM 底层即 cuBLAS
- **Day 42 Triton GEMM / autotune**：Triton 侧的矩阵乘实现

---

## 核心概念

### 三种实现方式的取舍

```
  方式         代表          开发效率  性能        融合能力  适用场景
  ----------------------------------------------------------------------
  手写 CUDA    .cu kernel    低        极高(可调)  强(全手动) 极致优化/特殊算子
  厂商库       cuBLAS/cuDNN  高(调API) 极高        无(黑盒)   标准算子(GEMM/卷积)
  Triton       @triton.jit   中高      高(近手写)  强(易写)   自定义融合算子
```

| 维度 | 手写 CUDA | cuBLAS/cuDNN | Triton |
|------|-----------|--------------|--------|
| 语言 | C++ CUDA | C API | Python |
| 上手难度 | 高 | 低 | 中 |
| 标准 GEMM 性能 | 高（难超库） | 极高 | 高 |
| 自定义融合 | 强但费力 | 不支持 | 强且简洁 |
| 自动调优 | 手动 | 内置 | `@triton.autotune` |
| 可微/训练接入 | 手写 | 框架封装 | `autograd.Function` |

**经验法则**：
- 标准 GEMM/卷积 → 用 cuBLAS/cuDNN，几乎无人能手写超过；
- 需要融合的自定义算子（Fused LayerNorm / Flash Attention）→ Triton 首选；
- 追求极致或库/Triton 都覆盖不了的模式 → 手写 CUDA。

### 正确的 GPU 计时

```python
for _ in range(warmup): fn()          # 预热：触发 JIT 编译、稳定时钟
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters): fn()
end.record()
torch.cuda.synchronize()              # 必须同步，GPU 是异步的
ms = start.elapsed_time(end) / iters
```

> 忘记 warmup 或同步是 GPU 计时最常见的坑，会把 kernel 启动/编译时间也算进去。

---

## 第九阶段（Day41-47）回顾表

| Day | 主题 | 核心要点 |
|-----|------|----------|
| 41 | Triton 入门 | program/BLOCK/mask ↔ CUDA thread/block/if 边界 |
| 42 | GEMM + autotune | 分块矩阵乘，`@triton.autotune` 搜索 block 配置 |
| 43 | Fused Softmax/LayerNorm | 一次读入、片上完成多步、只写回一次 |
| 44 | Flash Attention | tiling + online softmax，比 CUDA 手写简洁 |
| 45 | Fused MLP | GEMM→+bias→GELU 融合，省中间张量落地 |
| 46 | 接入 PyTorch | autograd.Function forward/backward + 梯度校验 + nn.Module |
| 47 | 阶段小结 | Triton vs CUDA vs cuBLAS 取舍与方法论（本天） |

**一句话总结**：Triton 让你用 Python 写「以块为单位」的 kernel，编译器负责向量化、共享内存与调度；它在「自定义融合算子」这个 CUDA 最痛、cuBLAS 覆盖不到的地带，取得开发效率与性能的最佳平衡，是当今大模型算子的主流写法。

---

## 文件结构

```
day47_triton_summary/
├── python/triton_summary.py   # torch vs Triton 计时对比 + 阶段回顾
├── build.sh                   # Linux/WSL2 运行脚本
├── build.bat                  # Windows 运行脚本（提示用 WSL2）
└── README.md                  # 本文件
```

---

## 编译与运行

### Linux / WSL2（推荐，有 NVIDIA GPU）

```bash
cd day47_triton_summary
pip install triton torch
python python/triton_summary.py
```

### Windows

```powershell
cd day47_triton_summary
.\build.bat
```

> 脚本会检测环境：有 GPU+Triton 则做真实 cuBLAS vs Triton 对比；否则用 CPU/NumPy 计时演示方法论（不伪造 GPU 数据）。

---

## 知识点总结

1. **三方取舍**：cuBLAS 强于标准算子，Triton 强于自定义融合，CUDA 强于极致特殊需求。
2. **GPU 计时三要素**：warmup、`cuda.Event`、`synchronize`。
3. **加速比解读**：GEMM 上 cuBLAS 常领先；访存受限的 Softmax/融合算子 Triton 常反超。
4. **正确性优先**：对比性能前先做误差校验，快而错没有意义。
5. **方法论沉淀**：新算子先问「库有没有？要不要融合？」再决定用哪种方式。

---

## 验收标准

- 程序打印三种实现方式对照表与第九阶段回顾表，你能讲清各自适用场景。
- 有 GPU 时：GEMM/Softmax 误差校验通过，并打印 torch vs Triton 的耗时与加速比。
- 无 GPU 时：用 CPU/NumPy 计时演示方法论，你能说出为何这不代表 GPU 性能。
- 你能复述正确 GPU 计时的三要素，并解释忘记同步的后果。

---

## 动手挑战

1. **加维度扫描**：把 GEMM 尺寸从 512 扫到 4096，画出 torch vs Triton 的耗时曲线，找到 Triton 反超/落后的拐点。
2. **加 TFLOPS**：把 GEMM 耗时换算成 TFLOPS（`2*M*N*K / t`），对比两者的算力利用率。
3. **融合对比**：把 Day45 的 Fused MLP 与「torch 分步」也加入计时表，验证融合收益。
4. **autotune**：给 Triton GEMM 加 `@triton.autotune`，看能否进一步逼近 cuBLAS。

---

## 常见问题

- **Q：为什么我的 Triton GEMM 比 cuBLAS 慢很多？**
  A：本例用固定 block 配置且未 autotune；cuBLAS 是高度优化的闭源库。加 `@triton.autotune` 会明显改善，但标准 GEMM 想超越 cuBLAS 仍很难。

- **Q：CPU 计时能代表 GPU 吗？**
  A：不能。CPU 计时仅用于无 GPU 环境演示方法论，绝对/相对数值都与 GPU 不同。

- **Q：学完第九阶段，我该用哪个？**
  A：标准算子直接调库；要融合/自定义就上 Triton；只有在两者都不满足时才手写 CUDA。

---

## 延伸阅读

- [Triton 官方教程合集](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [PyTorch 性能基准与计时最佳实践](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [FlashAttention 论文（融合算子的经典范例）](https://arxiv.org/abs/2205.14135)
