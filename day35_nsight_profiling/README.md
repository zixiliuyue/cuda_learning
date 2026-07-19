# Day 35: Nsight 深度性能分析

## 学习目标

1. 理解内存带宽、计算吞吐、占用率等关键性能指标
2. 掌握判断 Kernel 是 **memory-bound** 还是 **compute-bound** 的方法
3. 理解 **Roofline 模型** 及其“拐点强度”的含义
4. 使用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 计算占用率
5. 学会使用 **Nsight Compute (ncu)** 与 **Nsight Systems (nsys)** 分析程序

## 四类 Kernel 及其特征

| Kernel | 特征 | 瓶颈 | 优化方向 |
|--------|------|------|----------|
| 内存带宽受限 | 向量加法，算术强度极低 | 显存带宽 | 减少访存、提升合并访问 |
| 计算受限 | 每元素大量 FMA | 计算吞吐 | 减少运算、使用更快指令 |
| 低占用率 | 大量寄存器，并行度低 | 延迟隐藏不足 | 降低寄存器 / 调整 Block |
| 高占用率 | 轻量寄存器 | 通常较优 | 作为对照基准 |

## 核心概念

### Roofline 模型

```
性能(GFLOPS)
   ^
   |            ______________ 计算屋顶（峰值算力）
   |          /
   |        /  <- 斜线区：memory-bound（受带宽限制）
   |      /
   |    /
   +--/-----------------------> 算术强度 (FLOP/Byte)
      ^拐点强度 = 峰值算力 / 峰值带宽
```

- **算术强度** = 浮点运算数 / 访存字节数（FLOP/Byte）
- 低于拐点强度 → **memory-bound**（性能受带宽约束）
- 高于拐点强度 → **compute-bound**（性能受算力约束）

程序会自动打印当前 GPU 的拐点强度，可据此判断每个 Kernel 的类型。

### 占用率（Occupancy）

占用率 = 每 SM 实际活跃 Warp 数 / 每 SM 最大可容纳 Warp 数。

本项目使用：

```c
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, kernel, block_size, 0);
```

结合 `maxThreadsPerMultiProcessor` / `warpSize` 计算得到理论占用率。
占用率越高越有利于隐藏访存与指令延迟，但**并非越高越好**——
过度追求占用率可能导致寄存器溢出（register spilling）。

## 编译与运行

### Windows (nvcc + sm_120)

```powershell
cd day35_nsight_profiling
.\build.bat
.\build\nsight_profiling.exe
```

### Linux / macOS (CMake)

```bash
cd day35_nsight_profiling
bash build.sh
./build/nsight_profiling
```

### Python 版本

```bash
python python/nsight_profiling.py
```

## Nsight 工具安装与使用

### 安装

- **Nsight Compute (ncu)**：Kernel 级别的深度性能分析器，随 CUDA Toolkit 一起安装
  （也可从 NVIDIA 官网单独下载）。
- **Nsight Systems (nsys)**：系统级时间线分析器，观察 CPU/GPU/内存传输的整体时序。

安装后确认在 PATH 中：

```powershell
ncu --version
nsys --version
```

### Nsight Compute 命令行

```powershell
REM 采集完整指标集（最详细，耗时较长）
ncu --set full .\build\nsight_profiling.exe

REM 只看某个 Kernel（按名称过滤）
ncu -k kernel_memory_bound .\build\nsight_profiling.exe

REM 导出报告文件，再用 GUI 打开
ncu --set full -o report .\build\nsight_profiling.exe
```

### Nsight Systems 命令行

```powershell
REM 生成时间线报告
nsys profile -o timeline .\build\nsight_profiling.exe

REM 打开 timeline.nsys-rep 用 Nsight Systems GUI 查看
```

## 本节课 Kernel 分析结果预期

- **kernel_memory_bound**：带宽利用率高（接近理论带宽），Compute Throughput 低。
  在 ncu 中 "Memory Throughput" 会明显高于 "Compute (SM) Throughput"。
- **kernel_compute_bound**：SM 利用率高，DRAM 利用率低。
  在 ncu 中 "Compute (SM) Throughput" 接近上限。
- **kernel_low_occupancy**：ncu 报告的 "Achieved Occupancy" 偏低，
  "Registers Per Thread" 偏高，可能出现 "Occupancy Limited by Registers"。
- **kernel_high_occupancy**：占用率高，通常延迟隐藏更好。

## 如何解读 Nsight 报告的关键指标

| 指标 | 含义 | 关注点 |
|------|------|--------|
| SM Throughput | 计算单元利用率 | 高 → compute-bound |
| Memory Throughput / DRAM | 显存带宽利用率 | 高 → memory-bound |
| Achieved Occupancy | 实际占用率 | 与理论占用率对比 |
| Registers Per Thread | 每线程寄存器数 | 过高会限制占用率 |
| Warp Stall Reasons | Warp 停顿原因 | 定位延迟瓶颈 |
| L1/L2 Hit Rate | 缓存命中率 | 低命中说明访存模式差 |

## 关键教学点

1. 先用 Roofline 判断瓶颈类型，再有针对性地优化
2. memory-bound 不要盲目减少计算；compute-bound 不要盲目优化访存
3. 占用率是手段而非目的，最终以实测吞吐 / 耗时为准
4. `-lineinfo` 编译选项可让 ncu 将指标映射回具体源码行

---

## 验收标准

完成本天学习后，运行 `nsight_profiling.exe`，你应当能够观察到并解释以下现象，才算真正掌握：

1. **理论上限被正确打印**：程序开头输出「理论显存带宽 (GB/s)」与「理论计算吞吐(FP32) (GFLOPS)」，最后的 Roofline 提示给出「拐点强度 = 理论算力 / 理论带宽」的具体 FLOP/Byte 值，你应能据此判断某个算术强度落在带宽区还是计算区。
2. **两类 Kernel 的利用率呈现互补特征**：内存受限 Kernel 的「带宽利用率」应远高于其计算利用率、接近理论带宽；计算受限 Kernel 的「计算利用率」应远高于其带宽占用——这正是 memory-bound 与 compute-bound 的直接证据。
3. **占用率分析表数值合理**：占用率表中四个 Kernel 的「活跃 Warp / 占用率」被打印出来，其中「低占用率」Kernel 的占用率应明显低于「高占用率」Kernel，验证了寄存器用量对并行度的影响。
4. **能用 Nsight 交叉验证**：运行 `ncu --set full .\build\nsight_profiling.exe` 后，ncu 报告中 `kernel_memory_bound` 的 Memory Throughput 高、`kernel_compute_bound` 的 Compute (SM) Throughput 高、`kernel_low_occupancy` 的 Achieved Occupancy 偏低，与程序自算结果一致，说明你能把两者对上号。

## 动手挑战

以下任务基于本天 `src/main.cu` 的真实实现，循序渐进地加深理解：

1. **改变算术强度观察瓶颈迁移**：把 `COMPUTE_ITERS`（当前 512）逐步调小到几十，重新运行，观察 `kernel_compute_bound` 的实测 GFLOPS 与计算利用率如何下降、逐渐向 memory-bound 靠拢，亲手验证 Roofline 的拐点效应。
2. **用 -lineinfo + ncu 定位热点行**：在编译命令中加入 `-lineinfo`，用 `ncu -k kernel_low_occupancy --set full .\build\nsight_profiling.exe` 采集，查看报告里的 Registers Per Thread 与 “Occupancy Limited by Registers” 提示，把占用率瓶颈定位到 `kernel_low_occupancy` 里那组独立累加器。
3. **调 Block 大小做占用率扫描**：把 `BLOCK_SIZE` 依次改为 128 / 256 / 512，对每种配置调用 `analyze_occupancy` 打印理论占用率并实测四个 Kernel 耗时，画出占用率与耗时的关系，体会“占用率并非越高越好”。
