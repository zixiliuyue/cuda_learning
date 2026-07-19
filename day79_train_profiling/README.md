# Day 79: 训练性能监控与 profiling 体系

> 🌐 **第十二阶段（分布式训练）**。训练慢别瞎猜——先建 profiling 体系。本天把一个训练 step 拆成 **前向 / 反向 / 通信 / 优化器** 四个阶段分别计时，看谁占大头，据此判定「计算受限 / 通信受限 / 显存受限」并对症下药。

## 学习目标

1. **搭建** 分阶段计时的训练 profiling（forward/backward/comm/optimizer）
2. **读懂** 各阶段耗时占比报告与条形图
3. **判定** 三类瓶颈（compute-bound / comm-bound / memory-bound）
4. **对应** 每类瓶颈到本阶段学过的具体优化手段

---

## 前置依赖

- **Day 35 Nsight Profiling**：GPU 算子级 profiling 工具
- **Day 26 AMP / Day 43 融合 / Day 78 重叠 / Day 75 检查点 / Day 77 FSDP**：本天判定出瓶颈后要用到的各类对策

---

## 核心概念

### 一个训练 step 的四阶段

```
forward → backward → [通信 all-reduce 梯度] → optimizer.step
```

### 三类瓶颈与对策

| 瓶颈类型 | 表现 | 对策 |
|----------|------|------|
| 计算受限 compute-bound | 前向/反向占大头 | AMP(Day26)、算子融合(Day43)、更优 kernel/Triton |
| 通信受限 comm-bound | 通信占大头 | 重叠(Day78)、分桶、更快互联(NVLink)、梯度压缩 |
| 显存受限 memory-bound | 被迫小 batch、OOM | 梯度检查点(Day75)、FSDP/ZeRO(Day77/69) |

### 计时要点

GPU 上计时必须先 `torch.cuda.synchronize()`，否则 kernel 异步执行会导致计时不准。GPU 首选 `torch.profiler` / Nsight Systems；无 GPU 用 `time.perf_counter` 手动计时，定位思路一致。

---

## 文件结构

```
day79_train_profiling/
├── python/train_profiling.py   # 分阶段计时 + 占比报告 + 瓶颈判定
├── build.sh                    # Linux/WSL2 运行脚本
├── build.bat                   # Windows 运行脚本
└── README.md                   # 本文件
```

> 有 torch：跑真实小模型训练 step，用 CUDA 同步精确计时；无 torch：numpy 模拟 MLP 训练 step。两条路径都输出占比报告与瓶颈判定，任何环境稳定退出。

---

## 编译与运行

### Linux / WSL2

```bash
cd day79_train_profiling
python python/train_profiling.py
```

### Windows

```powershell
cd day79_train_profiling
.\build.bat
```

### 用 torch.profiler 看算子级火焰图（进阶，需 GPU）

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train_one_step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 知识点总结

1. **分阶段计时**：把 step 拆成 4 段，先定位再优化。
2. **CUDA 同步**：GPU 计时前后必须 synchronize。
3. **占比优先**：看百分比而非绝对值，抓主要矛盾。
4. **三类瓶颈**：计算/通信/显存受限，各有专属对策。
5. **预热**：正式计时前跑几步预热，排除 lazy init / autotune 干扰。

---

## 验收标准

- 报告打印四阶段的总耗时、每 step 耗时、占比与条形图，占比合计 100%。
- 程序给出明确的瓶颈判定与对应的优化建议。
- 你能解释为什么单卡演示里「通信」阶段占比不能代表真实多卡。
- 你能说出计算受限与通信受限各自的典型对策。

---

## 动手挑战

1. **放大模型**：把 hidden 调到 4096，观察前向/反向占比上升，逼近计算受限。
2. **接入 torch.profiler**：用上面的代码片段导出 `key_averages` 表，找出耗时最高的算子。
3. **模拟通信受限**：在 `fake_comm` 里加入 `time.sleep` 放大通信耗时，看瓶颈判定切换到 comm-bound。

---

## 常见问题

- **Q：为什么不能直接用 time.time() 给 GPU 计时？**
  A：GPU kernel 是异步下发的，不 synchronize 会测到「下发时间」而非「执行时间」。

- **Q：profiling 会拖慢训练吗？**
  A：手动计时几乎无开销；torch.profiler 有一定开销，建议只在少数 step 上开启（用 schedule 控制）。

---

## 延伸阅读

- [PyTorch Profiler 官方教程](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
