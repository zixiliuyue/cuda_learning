# NVIDIA Profiler 性能分析工具使用指南

## 目录
1. [概述](#概述)
2. [nvprof 命令行使用](#nvprof-命令行使用)
3. [Nsight Systems 使用](#nsight-systems-使用)
4. [Nsight Compute 使用](#nsight-compute-使用)
5. [常用性能指标解读](#常用性能指标解读)
6. [实战示例](#实战示例)

## 概述

NVIDIA 提供了三套主要的性能分析工具：

| 工具 | 用途 | 分析级别 |
|------|------|----------|
| **nvprof** | 命令行分析工具（已废弃，推荐使用 Nsight 系列） | 应用程序级 |
| **Nsight Systems** | 系统级性能分析，查看整体执行流程 | 系统级 |
| **Nsight Compute** | 核函数级性能分析，深入分析指令执行 | 核函数级 |

## nvprof 命令行使用

### 基本使用

```bash
# 基本性能分析
nvprof ./simple_kernel

# 输出核函数详细信息
nvprof --print-gpu-trace ./simple_kernel

# 输出内存操作详细信息
nvprof --print-api-trace ./simple_kernel
```

### 常用命令选项

```bash
# 指定输出文件
nvprof -o profile_result.nvprof ./simple_kernel

# 分析特定事件
nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict ./matrix_multiply

# 分析性能指标
nvprof --metrics achieved_occupancy,inst_per_warp,gld_efficiency ./matrix_multiply

# 显示所有可用事件
nvprof --query-events

# 显示所有可用性能指标
nvprof --query-metrics
```

### 输出示例解读

```
==12345== NVPROF is profiling process 12345, command: ./simple_kernel
==12345== Profiling application: ./simple_kernel
==12345== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   40.2%  10.03ms         1  10.03ms  10.03ms  10.03ms  naive_matmul
                   35.1%  8.765ms         1  8.765ms  8.765ms  8.765ms  tiled_matmul
                   24.7%  6.165ms         1  6.165ms  6.165ms  6.165ms  tiled_matmul_bank_conflict
      API calls:   60.1%  30.52ms         3  10.17ms  10.03ms  10.32ms  cudaEventSynchronize
                   25.3%  12.85ms         4  3.212ms  1.234ms  5.678ms  cudaMalloc
                   10.2%  5.176ms         2  2.588ms  2.345ms  2.831ms  cudaMemcpy
```

## Nsight Systems 使用

### 1. 安装与启动

```bash
# Linux
nsys-ui

# Windows (从开始菜单启动)
# Nsight Systems 202x.x.x
```

### 2. 命令行采集数据

```bash
# 基本采集
nsys profile -o report_name ./simple_kernel

# 采集 CUDA 和 CPU 信息
nsys profile --trace=cuda,cuda_api,osrt ./simple_kernel

# 带采样分析
nsys profile --sample=process --trace=cuda,cuda_api ./matrix_multiply
```

### 3. GUI 分析步骤

1. **启动 Nsight Systems**
   - 选择 "New Project"
   - 选择 "Collect Data"
   - 填写应用程序路径和工作目录

2. **关键分析视图**

   | 视图 | 用途 |
   |------|------|
   | **Timeline View** | 查看 CPU/GPU 执行时间线 |
   | **CUDA Hardware** | 查看 GPU 硬件利用率 |
   | **CUDA API** | 查看 CUDA API 调用耗时 |
   | **Threads** | 查看线程执行情况 |

3. **常见问题定位**

   - **GPU 空闲时间长**：检查数据传输是否成为瓶颈
   - **API 调用耗时**：检查 cudaMalloc/cudaMemcpy 频率
   - **核函数重叠**：检查流（Stream）使用情况

## Nsight Compute 使用

### 1. 启动 Nsight Compute

```bash
# Linux
ncu-ui

# Windows (从开始菜单启动)
# Nsight Compute 202x.x.x
```

### 2. 命令行采集

```bash
# 基本分析
ncu -o kernel_profile ./simple_kernel

# 分析特定核函数
ncu -k naive_matmul ./matrix_multiply

# 收集完整性能指标
ncu --set full -o full_profile ./matrix_multiply
```

### 3. 关键性能指标分析

#### 内存访问模式
```bash
# 检查全局内存访问效率
ncu --metrics gld_efficiency,gst_efficiency,gld_transactions,gst_transactions ./simple_kernel
```

#### 指令吞吐量
```bash
# 检查指令执行效率
ncu --metrics ipc,inst_per_warp,issued_ipc,active_warps_per_cycle ./simple_kernel
```

#### 占用率分析
```bash
# 检查计算占用率
ncu --metrics achieved_occupancy,occupancy,warps_per_multiprocessor ./simple_kernel
```

### 4. Nsight Compute GUI 操作步骤

1. **连接并选择应用程序**
   - 点击 "Connect"
   - 选择 "Launch"
   - 配置应用程序路径和参数

2. **配置分析会话**
   - 选择分析配置（Baseline、Full、Custom）
   - 选择要分析的核函数
   - 设置采样频率

3. **分析结果**
   - **Source Page**：源码级性能分析
   - **Assembly Page**：汇编指令级分析
   - **Memory Workload Analysis**：内存工作负载
   - **Compute Workload Analysis**：计算工作负载
   - **Scheduler Statistics**：调度器统计

## 常用性能指标解读

### 1. 内存相关指标

| 指标 | 描述 | 理想值 | 常见问题 |
|------|------|--------|----------|
| **gld_efficiency** | 全局内存加载效率 | > 80% | 低表示内存访问模式差 |
| **gst_efficiency** | 全局内存存储效率 | > 80% | 低表示存储模式差 |
| **shared_efficiency** | 共享内存访问效率 | > 90% | 低表示 bank 冲突 |
| **dram_read_throughput** | DRAM 读吞吐量 | 接近峰值 | 低表示内存带宽未充分利用 |
| **dram_write_throughput** | DRAM 写吞吐量 | 接近峰值 | 低表示内存带宽未充分利用 |

### 2. 计算相关指标

| 指标 | 描述 | 理想值 | 常见问题 |
|------|------|--------|----------|
| **achieved_occupancy** | 实际占用率 | > 50% | 低表示资源限制或并行度不足 |
| **ipc** | 每周期指令数 | 越高越好 | 低表示有停顿 |
| **inst_per_warp** | 每 warp 指令数 | - | 高表示指令复杂度高 |
| **sm_efficiency** | SM 效率 | > 70% | 低表示 SM 未充分利用 |

### 3. 分支与分歧

| 指标 | 描述 | 理想值 | 常见问题 |
|------|------|--------|----------|
| **branch_efficiency** | 分支效率 | > 95% | 低表示 warp 内分歧严重 |
| **divergent_branch** | 分歧分支数 | 0 | 有则表示线程执行路径不同 |

### 4. 指令混合

| 指标 | 描述 | 说明 |
|------|------|------|
| **fp32_instr** | FP32 指令比例 | 计算密集型应用应较高 |
| **fp64_instr** | FP64 指令比例 | 双精度计算应用较高 |
| **ldst_instr** | 加载存储指令比例 | 内存密集型应用较高 |

## 实战示例

### 示例 1：分析 simple_kernel.cu

#### 步骤 1：编译程序
```bash
nvcc -O3 -o simple_kernel simple_kernel.cu
```

#### 步骤 2：使用 nvprof 基本分析
```bash
nvprof ./simple_kernel
```

**预期输出分析要点：**
- 查看三个核函数的执行时间
- 比较 saxpy 与 加法+乘法的时间
- 检查内存传输时间

#### 步骤 3：分析内存效率
```bash
nvprof --metrics gld_efficiency,gst_efficiency ./simple_kernel
```

**预期结果：**
- 三个核函数的 gld_efficiency 应该都接近 100%
- 因为都是连续内存访问

#### 步骤 4：使用 Nsight Compute 详细分析
```bash
ncu --set full ./simple_kernel
```

**分析要点：**
- 查看指令吞吐量
- 检查内存访问模式
- 分析占用率

### 示例 2：分析 matrix_multiply.cu

#### 步骤 1：编译程序
```bash
nvcc -O3 -o matrix_multiply matrix_multiply.cu
```

#### 步骤 2：检测共享内存 bank 冲突
```bash
nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict ./matrix_multiply
```

**预期结果：**
- `tiled_matmul` 会显示 bank 冲突
- `tiled_matmul_bank_conflict` 冲突数应为 0

#### 步骤 3：比较三种版本的性能指标
```bash
# 分析计算占用率
nvprof --metrics achieved_occupancy ./matrix_multiply

# 分析指令吞吐量
nvprof --metrics ipc ./matrix_multiply

# 分析内存吞吐量
nvprof --metrics dram_read_throughput,dram_write_throughput ./matrix_multiply
```

#### 步骤 4：使用 Nsight Systems 查看时间线
```bash
nsys profile -o matmul_profile ./matrix_multiply
```

**在 Nsight Systems GUI 中查看：**
- 三个核函数的执行顺序
- 内存传输与计算的重叠
- CPU/GPU 交互

### 示例 3：优化前后对比

#### 优化前（朴素矩阵乘法）
```bash
ncu -k naive_matmul --metrics sm_efficiency,gld_efficiency,achieved_occupancy ./matrix_multiply
```

#### 优化后（分块矩阵乘法）
```bash
ncu -k tiled_matmul_bank_conflict --metrics sm_efficiency,gld_efficiency,achieved_occupancy ./matrix_multiply
```

#### 对比分析表格
| 指标 | 朴素版本 | 分块版本 | 改进 |
|------|----------|----------|------|
| 执行时间 | ~10ms | ~6ms | 1.67x |
| SM 效率 | ~30% | ~70% | 2.33x |
| 内存效率 | ~15% | ~80% | 5.33x |
| 占用率 | ~40% | ~60% | 1.5x |

## 常见问题排查

### 问题 1：低内存访问效率

**症状：** gld_efficiency < 50%

**可能原因：**
- 非合并内存访问
- 内存访问步长过大
- 随机内存访问

**解决方法：**
- 调整数据布局
- 使用共享内存
- 重新组织线程访问模式

### 问题 2：低占用率

**症状：** achieved_occupancy < 30%

**可能原因：**
- 每个 block 使用的寄存器太多
- 共享内存使用量过大
- block 大小设置不合理

**解决方法：**
- 使用 `launch_bounds` 限制寄存器使用
- 调整 block 大小（如 64, 128, 256, 512）
- 减少共享内存使用

### 问题 3：共享内存 bank 冲突

**症状：** shared_ld_bank_conflict > 0

**解决方法：**
- 使用填充（padding）增加一列
- 调整数据访问顺序
- 使用不同的 tile 大小

### 问题 4：Warp 分歧

**症状：** branch_efficiency < 90%

**解决方法：**
- 重构条件语句
- 使用谓词指令
- 重新设计算法减少分支

## 总结

性能分析流程建议：

1. **系统级分析（Nsight Systems）**
   - 找出性能瓶颈：计算 / 内存 / 数据传输
   - 查看整体时间线

2. **核函数级分析（Nsight Compute）**
   - 深入分析具体核函数
   - 检查内存访问、指令执行、占用率

3. **迭代优化**
   - 根据分析结果优化代码
   - 对比优化前后性能
   - 重复分析直到达到目标

记住：**先测量，再优化**，不要盲目猜测性能瓶颈！
