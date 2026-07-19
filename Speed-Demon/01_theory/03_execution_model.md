# CUDA 执行模型 ⚡

## 1. SIMT 执行模型 🎯

### 1.1 什么是 SIMT？

**SIMT (Single Instruction, Multiple Threads)** - 单指令多线程

SIMT 是 CUDA 的核心执行模型，它将 32 个线程组成一个 Warp，同步执行相同的指令。

### 1.2 SIMD vs SIMT 对比

| 特性 | SIMD (单指令多数据) | SIMT (单指令多线程) |
|------|-------------------|-------------------|
| 执行单位 | 向量寄存器 | 线程 (Warp) |
| 控制流 | 无分支能力 | 有分支发散处理 |
| 内存访问 | 固定对齐 | 灵活但需合并 |
| 编程模型 | 显式向量操作 | 隐式线程并行 |
| 例子 | SSE, AVX | CUDA Warp |

### 1.3 SIMT 工作原理

```
┌─────────────────────────────────────────────────────┐
│                     Warp (32个线程)                  │
│                                                     │
│  指令发射: [ADD R1, R2]                             │
│         ↓                                            │
│  T0: R1 = 1, R2 = 2 → 结果 = 3                      │
│  T1: R1 = 2, R2 = 3 → 结果 = 5                      │
│  T2: R1 = 3, R2 = 4 → 结果 = 7                      │
│  ...                                                 │
│  T31: R1 = 32, R2 = 33 → 结果 = 65                  │
│                                                     │
│  同一指令，不同数据！                                │
└─────────────────────────────────────────────────────┘
```

### 1.4 关键特性

| 特性 | 说明 |
|------|------|
| **同步执行** | Warp 内所有线程同步执行相同指令 |
| **独立线程状态** | 每个线程有自己的 PC、寄存器、栈 |
| **分支处理** | 通过掩码实现条件执行 |
| **零开销调度** | Warp 间切换无上下文切换开销 |

---

## 2. Warp 调度 🚀

### 2.1 Warp 调度器

每个 SM 包含多个 Warp 调度器，负责:
- 管理就绪的 Warp
- 选择下一个要执行的 Warp
- 发射指令到执行单元

```
┌─────────────────────────────────────────────────┐
│              Streaming Multiprocessor            │
│                                                 │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ Warp调度器 1 │  │ Warp调度器 2 │  ...       │
│  └──────────────┘  └──────────────┘            │
│         │                │                      │
│         ▼                ▼                      │
│  ┌───────────────────────────────────┐          │
│  │         执行单元 (ALU, FPU)        │          │
│  └───────────────────────────────────┘          │
│                                                 │
│  就绪 Warp 队列: [W0, W3, W7, W12, ...]         │
└─────────────────────────────────────────────────┘
```

### 2.2 调度策略

**Round-Robin (轮询调度)**:
- 按顺序循环调度就绪的 Warp
- 简单高效，公平对待所有 Warp

**两级调度**:
- 第一级: 选择哪个 Warp 调度器工作
- 第二级: 调度器选择哪个 Warp 执行

### 2.3 延迟隐藏 (Latency Hiding)

GPU 通过切换到其他 Warp 来隐藏长延迟操作（如内存访问）。

```
时间线:
Warp 0: 发射内存读取指令 ──▶ 等待内存响应 (200-800 cycles)
        ↓ (立即切换)
Warp 1: 执行计算指令  ──▶ 继续执行...
        ↓
Warp 2: 执行计算指令  ──▶ 继续执行...
        ↓
... (其他 Warp 执行)
        ↓
Warp 0: 内存数据就绪 ──▶ 继续执行后续指令
```

**延迟隐藏公式**:

```
需要的 Warp 数量 = 延迟 (cycles) / 每个 Warp 每周期指令数

例如:
- 内存访问延迟: 400 cycles
- 每个 Warp 每周期发射 1 条指令
- 需要 Warp 数量: 400 / 1 = 400 个 Warp
```

### 2.4 隐藏不同类型的延迟

| 操作类型 | 典型延迟 | 需要的 Warp 数量 |
|---------|---------|-----------------|
| 算术运算 | 10-20 cycles | 少量 (10-20) |
| 共享内存访问 | 20-40 cycles | 中等 (20-40) |
| 全局内存访问 | 200-800 cycles | 大量 (200-800) |

---

## 3. 分支发散 (Branch Divergence) 🔀

### 3.1 什么是分支发散？

当 Warp 内的线程执行不同的代码路径时，就会发生分支发散。

### 3.2 分支发散示例

```cpp
__global__ void branch_example(float* data) {
    int idx = threadIdx.x;
    
    // Warp 内所有线程到达这里
    
    if (idx < 16) {
        // 路径 A: 线程 0-15 执行
        data[idx] = 1.0f;
    } else {
        // 路径 B: 线程 16-31 执行
        data[idx] = 2.0f;
    }
    
    // Warp 重新汇合
}
```

**执行时序**:

```
周期 1-10: 执行路径 A (线程 0-15 激活，16-31 禁用)
周期 11-20: 执行路径 B (线程 16-31 激活，0-15 禁用)

总耗时: 2 倍! ⚠️
```

### 3.3 分支发散的性能影响

| 发散模式 | 路径数 | 性能损失 |
|---------|--------|---------|
| 无发散 | 1 | ✅ 无损失 |
| 2-way | 2 | ~50% |
| 4-way | 4 | ~75% |
| N-way | N | ~(N-1)/N |
| 完全随机 | 32 | ~97% ❌ |

### 3.4 避免分支发散的技巧 🎯

#### 技巧 1: 确保 Warp 内线程走同一路径

```cpp
// ❌ 坏: Warp 内可能发散
if (threadIdx.x % 2 == 0) { ... }

// ✅ 好: Warp 内走同一路径 (以 32 为边界)
if ((threadIdx.x / 32) % 2 == 0) { ... }
```

#### 技巧 2: 使用谓词执行 (Predication)

对于简短的分支，编译器会自动使用谓词执行:

```cpp
// 编译器可能转换为谓词执行，避免分支
float val = (idx < 16) ? 1.0f : 2.0f;
data[idx] = val;
```

#### 技巧 3: 重新组织数据/算法

```cpp
// ❌ 基于值的分支 (可能发散)
if (data[idx] > threshold) {
    process_A(data[idx]);
} else {
    process_B(data[idx]);
}

// ✅ 先排序/分组，让同 Warp 处理相同类型
// (Host 端预先排序数据)
```

#### 技巧 4: 使用洗牌指令 (Shuffle) 避免分支

```cpp
__device__ float reduce_sum(float val) {
    unsigned int idx = threadIdx.x;
    __shared__ float sh[32];
    
    sh[idx] = val;
    __syncthreads();
    
    // 无分支归约
    for (int i = 16; i > 0; i >>= 1) {
        if (idx < i) {
            sh[idx] += sh[idx + i];
        }
        __syncthreads();
    }
    
    return sh[0];
}
```

### 3.5 分支发散检测工具

| 工具 | 用途 |
|------|------|
| `nvprof` | 查看分支效率指标 |
| `Nsight Compute` | 详细分析每个内核的分支发散 |
| `--ptxas-options=-v` | 编译时查看分支统计 |
| CUDA 采样计数器 | `branch_efficiency` 指标 |

---

## 4. 占用率 (Occupancy) 计算 📊

### 4.1 什么是占用率？

**占用率** 是指每个 SM 上活跃的 Warp 数量与最大可容纳 Warp 数量的比值。

```
占用率 = 活跃 Warp 数 / SM 最大 Warp 数 × 100%
```

### 4.2 各架构的硬件限制

| 架构 | SM 最大 Warp | SM 最大 Block | 每 Block 最大 Thread | 每 SM 最大 Register |
|------|-------------|--------------|---------------------|---------------------|
| Kepler | 64 | 16 | 1024 | 64K |
| Maxwell | 64 | 32 | 1024 | 64K |
| Pascal | 64 | 32 | 1024 | 64K |
| Volta | 64 | 32 | 1024 | 64K |
| Turing | 64 | 32 | 1024 | 64K |
| Ampere | 64 | 32 | 1024 | 64K / 128K |

### 4.3 占用率的三个限制因素

占用率由三个因素共同决定，取最小值:

#### 因素 1: 线程数限制

```
每个 Block 的线程数 (block_size)
├── 每个 Block 的 Warp 数 = ceil(block_size / 32)
└── 每个 SM 的 Block 数 = SM 最大 Warp / 每 Block Warp 数

由此产生的占用率:
occupancy_threads = (Block 数 × 每 Block Warp 数) / SM 最大 Warp 数
```

#### 因素 2: 寄存器限制

```
每个线程使用的寄存器数 (reg_per_thread)
├── 每个 Block 使用的寄存器 = block_size × reg_per_thread
└── 每个 SM 的 Block 数 = SM 最大寄存器 / 每 Block 寄存器

由此产生的占用率:
occupancy_regs = (Block 数 × 每 Block Warp 数) / SM 最大 Warp 数
```

#### 因素 3: 共享内存限制

```
每个 Block 使用的共享内存 (shmem_per_block)
└── 每个 SM 的 Block 数 = SM 最大共享内存 / shmem_per_block

由此产生的占用率:
occupancy_shmem = (Block 数 × 每 Block Warp 数) / SM 最大 Warp 数
```

### 4.4 完整计算示例

**假设配置**:
- 架构: Ampere (SM 最大 Warp = 64)
- Block 大小: 256 线程
- 每个线程寄存器: 32 个
- 每个 Block 共享内存: 8 KB
- SM 最大寄存器: 64 KB
- SM 最大共享内存: 164 KB

**计算过程**:

```
步骤 1: 线程数限制
  每 Block Warp 数 = 256 / 32 = 8
  最大 Block 数 = 64 / 8 = 8
  占用率1 = (8 × 8) / 64 = 100%

步骤 2: 寄存器限制
  每 Block 寄存器 = 256 × 32 = 8192 = 8K 寄存器
  最大 Block 数 = 64K / 8K = 8
  占用率2 = (8 × 8) / 64 = 100%

步骤 3: 共享内存限制
  最大 Block 数 = 164KB / 8KB = 20 (向下取整为 8，受 Block 数限制)
  占用率3 = (8 × 8) / 64 = 100%

最终占用率 = min(100%, 100%, 100%) = 100% ✅
```

### 4.5 占用率计算代码

```cpp
#include <cuda_runtime.h>
#include <iostream>

void calculate_occupancy(int block_size, int reg_per_thread, 
                         int shmem_per_block) {
    int max_warps_per_sm = 64;  // Ampere
    int max_regs_per_sm = 65536; // 64K
    int max_shmem_per_sm = 164 * 1024; // 164KB
    
    // 计算每个 Block 的 Warp 数
    int warps_per_block = (block_size + 31) / 32;
    
    // 限制1: 线程数限制
    int blocks_by_threads = max_warps_per_sm / warps_per_block;
    float occ_threads = (blocks_by_threads * warps_per_block) / 
                        (float)max_warps_per_sm * 100;
    
    // 限制2: 寄存器限制
    int regs_per_block = block_size * reg_per_thread;
    int blocks_by_regs = max_regs_per_sm / regs_per_block;
    float occ_regs = (blocks_by_regs * warps_per_block) / 
                     (float)max_warps_per_sm * 100;
    
    // 限制3: 共享内存限制
    int blocks_by_shmem = max_shmem_per_sm / shmem_per_block;
    float occ_shmem = (blocks_by_shmem * warps_per_block) / 
                      (float)max_warps_per_sm * 100;
    
    // 最终占用率 (取最小值)
    float occupancy = std::min({occ_threads, occ_regs, occ_shmem});
    
    std::cout << "线程数限制占用率: " << occ_threads << "%" << std::endl;
    std::cout << "寄存器限制占用率: " << occ_regs << "%" << std::endl;
    std::cout << "共享内存限制占用率: " << occ_shmem << "%" << std::endl;
    std::cout << "最终占用率: " << occupancy << "%" << std::endl;
}
```

### 4.6 占用率与性能的关系

| 占用率范围 | 性能影响 | 建议 |
|-----------|---------|------|
| < 25% | ❌ 很差，延迟无法隐藏 | 必须优化 |
| 25-50% | ⚠️ 一般，部分延迟隐藏 | 尽量优化 |
| 50-75% | ✅ 良好，大部分延迟隐藏 | 可以接受 |
| 75-100% | ✅ 优秀，延迟隐藏良好 | 理想状态 |

**注意**: 占用率不是越高越好！达到 50%+ 后，继续提高占用率的收益递减。

### 4.7 提高占用率的方法

#### 方法 1: 减少寄存器使用

```cpp
// 编译时限制寄存器数量
nvcc -maxrregcount 32 kernel.cu

// 或使用内联汇编限制
__launch_bounds__(256, 8) __global__ void kernel(...) {
    // 编译器将尝试在 256 线程/Block 时达到 8 Block/SM
}
```

#### 方法 2: 调整 Block 大小

```cpp
// ❌ 可能限制占用率
dim3 block(64);  // 小 Block 可能导致 Block 数量限制

// ✅ 选择合适的 Block 大小
dim3 block(128); // 或 256, 512
```

#### 方法 3: 优化共享内存使用

```cpp
// ❌ 过多的共享内存限制占用率
__shared__ float sh[4096];  // 16KB

// ✅ 按需分配，动态共享内存
extern __shared__ float sh[];
```

---

## 5. 执行模型优化总结 📋

### 5.1 SIMT 优化检查清单

- [ ] 避免 Warp 内分支发散
- [ ] 确保 Warp 内内存访问合并
- [ ] Warp 内同步使用 `__syncwarp()` (CUDA 9+)
- [ ] 利用 Warp 级原语 (洗牌、投票)

### 5.2 Warp 调度优化检查清单

- [ ] 提供足够多的 Warp 以隐藏延迟
- [ ] 平衡计算与内存访问
- [ ] 避免长依赖链
- [ ] 使用流水线重叠计算与内存访问

### 5.3 分支发散优化检查清单

- [ ] 以 Warp 为边界设计分支条件
- [ ] 短分支使用谓词执行替代
- [ ] 数据排序减少发散
- [ ] 检查编译器生成的分支效率

### 5.4 占用率优化检查清单

- [ ] 目标占用率 > 50%
- [ ] 平衡三个限制因素
- [ ] 使用 `__launch_bounds__` 指导编译器
- [ ] 使用 CUDA Occupancy Calculator 分析

---

## 总结 📝

| 概念 | 核心要点 |
|------|---------|
| **SIMT** | 32 线程一组同步执行，有独立线程状态 |
| **Warp 调度** | 轮询调度，切换到就绪 Warp 隐藏延迟 |
| **延迟隐藏** | 需要足够多的活跃 Warp (200-800) |
| **分支发散** | Warp 内不同路径串行执行，严重影响性能 |
| **占用率** | 活跃 Warp / 最大 Warp，目标 > 50% |
| **三个限制** | 线程数、寄存器、共享内存取最小值 |

理解执行模型是编写高性能 CUDA 程序的关键！🎯
