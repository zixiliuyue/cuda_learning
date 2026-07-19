# 诊断决策树：低占用率 (Low Occupancy)

---

## 快速导航

```
发现低占用率
  │
  ├─▶ 步骤1: 确认实际占用率
  │       └─▶ 使用 Nsight Compute / nvprof
  │
  ├─▶ 步骤2: 识别限制因素
  │       ├─▶ 线程数限制 → 分支A
  │       ├─▶ 寄存器限制 → 分支B
  │       └─▶ 共享内存限制 → 分支C
  │
  ├─▶ 步骤3: 评估性能影响分析
  │       ├─▶ 需要提高占用率？
  │       └─▶ 还是需要降低占用率换取其他？
  │
  └─▶ 步骤4: 优化并验证
```

---

## 诊断流程详解

### 阶段一：占用率确认

#### 问题1: 什么是占用率？

```
占用率 = SM上活跃的warp数 / SM支持的最大warp数

  │
  ├─▶ 典型的最大warp数: 32 / 48 / 64 (依架构而定
  │   ├─▶ Kepler: 64 warps/SM
  │   ├─▶ Maxwell/Pascal: 64 warps/SM
  │   ├─▶ Volta/Turing/Ampere: 64 warps/SM
  │   └─▶ Ada Lovelace: 48 warps/SM
  │
  └─▶ 重要说明
      ├─▶ > 70%: 非常好，通常不需要优化
      ├─▶ 50-70%: 良好，可能还有提升空间
      ├─▶ 30-50%: 中等，可能影响延迟隐藏
      └─▶ < 30%: 低，需要关注
```

---

#### 步骤1: 测量实际占用率

```bash
# 使用 nvprof
nvprof --metrics achieved_occupancy ./your_app

# 使用 Nsight Compute (推荐
ncu --metrics achieved_occupancy,occupancy ./your_app
```

```
查看 achieved_occupancy 指标
  │
  ├─▶ achieved_occupancy > 70%
  │   └─▶ ✅ 占用率良好
  │       ├─▶ 如果性能仍有问题，瓶颈不在占用率
  │       └─▶ 检查其他因素（内存、指令等）
  │
  ├─▶ achieved_occupancy 30-70%
  │   └─▶ ⚠️  中等占用率
  │       ├─▶ 检查是否有内存等待
  │       ├─▶ 如果有，尝试提高占用率
  │       └─▶ 没有，可能占用率不是瓶颈
  │
  └─▶ achieved_occupancy < 30%
      └─▶ ❌ 低占用率，需要优化
          └─▶ 继续诊断具体限制因素
```

---

### 阶段二: 识别限制因素

#### 使用Occupancy Calculator分析

```
获取内核信息（--ptxas-options=-v 编译输出）
  │
  ├─▶ 每个线程的寄存器数量
  │
  ├─▶ 每个Block的共享内存大小
  │
  └─▶ Block大小（线程数/Block）
```

**示例输出：
```
ptxas info    : Used 64 registers, 1024 bytes smem, 328 bytes cmem[0]
                ↑                 ↑
           每个线程64寄存器   每个Block 1KB共享内存
```

---

#### 分支A: 线程数/Block大小限制

```
诊断A1: 检查Block大小
  │
  ├─▶ Block大小 < 128线程?
  │   └─▶ ❌ Block太小
  │       ├─▶ 每个Block只占一个warp，调度开销大
  │       └─▶ 建议增大到 128-512
  │
  ├─▶ Block大小不是32的倍数?
  │   └─▶ ❌ 非warp对齐
  │       ├─▶ 最后一个warp不满，浪费资源
  │       └─▶ 建议调整到32的倍数
  │
  ├─▶ Block大小 > 1024?
  │   └─▶ ⚠️  最大Block大小
  │       ├─▶ 寄存器/共享内存可能成为新瓶颈
  │       └─▶ 每个SM只能放1-2个Block
  │
  └─▶ ✅ 128 ≤ Block大小 ≤ 512，且是32倍数
      └─▶ 线程数限制不是问题
          └─▶ 继续检查其他因素
```

**代码示例：调整Block大小**

```cpp
// ❌ Block太小：32线程
dim3 bad_block(32);

// ✅ 推荐：256线程
dim3 good_block(256);

// ❌ 不是32的倍数
dim3 bad_dim(100);

// ✅ 2D Block也要注意：
dim3 good_2d(16, 16);  // 256线程，好
```

---

#### 分支B: 寄存器限制

```
诊断B1: 检查寄存器使用量
  │
  ├─▶ 每个线程使用 > 64 寄存器?
  │   └─▶ ❌ 寄存器压力大
  │       ├─▶ 计算：256线程/Block × 64寄存器/线程
  │       │   = 16384 寄存器/Block
  │       │   每个SM 65536寄存器 → 只能跑4个Block
  │       └─▶ 解决方案：
  │           ├─▶ 方案B1: 使用 __launch_bounds__
  │           ├─▶ 方案B2: 减少活跃变量
  │           └─▶ 方案B3: 减小Block大小
  │
  ├─▶ 每个线程使用 32-64 寄存器?
  │   └─▶ ⚠️  中等寄存器使用
  │       ├─▶ 可以尝试优化
  │       └─▶ 观察性能变化
  │
  └─▶ 每个线程使用 < 32 寄存器?
      └─▶ ✅ 寄存器压力小
          └─▶ 限制因素不是寄存器
```

---

**方案B1: 使用 `__launch_bounds__`**

```cpp
// ✅ 显式限制寄存器使用
// 语法: __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)

// 例1: 256线程/Block，期望每个SM至少跑6个Block
// 编译器会限制寄存器使用
__launch_bounds__(256, 6)
__global__ void optimized_kernel(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        // 编译器会尝试限制寄存器
        // 以满足每个SM至少跑6个Block
    }
}

// 例2: 只指定最大线程数
__launch_bounds__(128)
__global__ void kernel_128(float* data, int n) {
    // ...
}
```

**原理说明**：
- `__launch_bounds__(256, 6)` 告诉编译器：
  - 这个内核最多用256线程/Block
  - 期望每个SM至少跑6个Block
- 编译器会自动限制每个线程的寄存器使用
- 可能导致溢出到本地内存（local memory），需要权衡

---

**方案B2: 减少活跃变量**

```cpp
// ❌ 大量活跃变量，寄存器压力大
__global__ void heavy_registers(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float a0, a1, a2, a3, a4, a5, a6, a7;
        float b0, b1, b2, b3, b4, b5, b6, b7;
        // 16个同时活跃的变量
        a0 = data[idx + 0 * 256];
        a1 = data[idx + 1 * 256];
        // ...
        compute(a0, a1, a2, a3, a4, a5, a6, a7);
        compute(b0, b1, b2, b3, b4, b5, b6, b7);
    }
}

// ✅ 重用来减少寄存器压力
__global__ void light_registers(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float temp;  // 只需要一个临时变量
        
        // 计算a系列，用完即释放寄存器
        temp = data[idx + 0 * 256];
        compute(temp);
        temp = data[idx + 1 * 256];
        compute(temp);
        // ...
        
        // 计算b系列，重用同一个寄存器
        temp = data[idx + 8 * 256];
        compute(temp);
        // ...
    }
}
```

---

**方案B3: 调整Block大小**

```cpp
// ❌ Block太大，每个SM只能跑很少Block
dim3 big_block(1024);
// 1024线程 × 64寄存器/线程 = 65536寄存器
// 每个SM最多只能跑1个Block → 占用率可能只有 1024/2048 = 50%

// ✅ 改用较小的Block
dim3 medium_block(512);
// 512线程 × 64寄存器/线程 = 32768寄存器
// 每个SM可以跑2个Block → 占用率 1024/2048 = 50% （一样）
// 但更多Block数增加了调度灵活性，有利于隐藏延迟
```

---

#### 分支C: 共享内存限制

```
诊断C1: 检查共享内存使用量
  │
  ├─▶ 每个Block使用 > 32KB 共享内存?
  │   └─▶ ❌ 共享内存压力大
  │       ├─▶ 例：每个SM 48KB共享内存
  │       │   每个Block用32KB → 只能跑1个Block
  │       └─▶ 解决方案：
  │           ├─▶ 方案C1: 减少共享内存使用
  │           ├─▶ 方案C2: 分阶段使用共享内存
  │           └─▶ 方案C3: 改用L1缓存
  │
  ├─▶ 每个Block使用 16-32KB 共享内存?
  │   └─▶ ⚠️ 中等使用量
  │       ├─▶ 可能影响占用率
  │       └─▶ 可以尝试优化
  │
  └─▶ 每个Block使用 < 16KB 共享内存?
      └─▶ ✅ 共享内存压力小
          └─▶ 限制因素不是共享内存
```

---

**方案C1: 减少共享内存使用**

```cpp
// ❌ 使用4个独立的共享内存数组
__global__ void too_much_shared(float* data, int n) {
    __shared__ float s0[1024];  // 4KB
    __shared__ float s1[1024];  // 4KB
    __shared__ float s2[1024];  // 4KB
    __shared__ float s3[1024];  // 4KB
    // 总计 16KB/Block
    
    int tid = threadIdx.x;
    // ... 使用所有4个数组
}

// ✅ 只使用1个共享内存数组，分阶段重用
__global__ void reuse_shared(float* data, int n) {
    __shared__ float shared[1024];  // 只用 4KB
    
    int tid = threadIdx.x;
    
    // 阶段1：加载数据
    shared[tid] = data[blockIdx.x * 1024 + tid];
    __syncthreads();
    compute_stage1(shared);
    __syncthreads();
    
    // 阶段2：重用共享内存
    shared[tid] = transform(shared[tid]);
    __syncthreads();
    compute_stage2(shared);
    __syncthreads();
    
    // 阶段3：继续重用
    // ...
}
```

---

**方案C2: 动态共享内存**

```cpp
// ✅ 使用动态共享内存，运行时决定大小
extern __shared__ float dyn_shared[];

__global__ void dynamic_shared_kernel(float* data, int n, int tile_size) {
    int tid = threadIdx.x;
    
    // 动态分配 tile_size × tile_size
    // 启动时指定：kernel<<<grid, block, shared_size>>>()
}

// 可以根据问题规模调整共享内存大小
void run_with_various_sizes(int size) {
    int tile_size = 32;  // 可配置
    int shared_size = tile_size * tile_size * sizeof(float);
    
    dim3 block(tile_size, tile_size);
    dim3 grid(size / tile_size, size / tile_size);
    
    dynamic_shared_kernel<<<grid, block, shared_size>>>(data, size);
}
```

---

### 阶段三: 性能影响分析

#### 问题2: 低占用率真的影响性能吗？

```
不要盲目提高占用率！
  │
  ├─▶ 检查内存相关指标
  │   ├─▶ dram_read_throughput 低?
  │   ├─▶ gld_transactions 少?
  │   └─▶ 有大量的 stall 等待内存?
  │       ├─▶ 是 → 低占用率影响了延迟隐藏 → 需要提高
  │       └─▶ 否 → 低占用率可能不是主要问题
  │
  ├─▶ 检查指令相关指标
  │   ├─▶ sm_efficiency 高?
  │   ├─▶ ipc (instructions per cycle) 高?
  │   └─▶ 计算密集型内核，不需要太多线程隐藏延迟
  │       ├─▶ 是 → 低占用率可能没问题，不影响
  │       └─▶ 否 → 需要提高占用率
  │
  └─▶ 实际测试
      ├─▶ 尝试不同的Block大小
      ├─▶ 尝试 __launch_bounds__
      └─▶ 性能是否提升?
          ├─▶ 是 → 占用率确实是瓶颈
          └─▶ 否 → 占用率不是关键瓶颈
```

**重要原则**：
> 不要为了提高占用率而大幅增加每个线程的指令数
> 也不要为了提高占用率而大幅增加寄存器溢出

---

### 阶段四: 优化验证

```
实施优化后
  │
  ├─▶ 占用率是否提升？
  │   ├─▶ 提升 > 20% → 继续观察性能
  │   ├─▶ 提升 < 10% → 优化效果有限
  │   └─▶ 下降 → 回滚，检查副作用
  │
  ├─▶ 性能（执行时间）是否提升？
  │   ├─▶ 提升 > 10% → 优化有效 ✅
  │   ├─▶ 变化 < 5% → 对性能影响不大
  │   └─▶ 下降 → 有副作用，需要权衡
  │
  └─▶ 检查是否引入新问题？
      ├─▶ 寄存器溢出（spill store/load > 0）?
      ├─▶ 指令数增加太多?
      └─▶ 共享内存Bank冲突加剧?
```

---

## 常见问题快速检查表

| 问题 | 检查项 | 目标值 | 建议 |
|------|--------|--------|-----|
| Block大小 | 每个Block的线程数 | 128-512 | 太小增大，太大减小 |
| 寄存器使用 | 每个线程寄存器数 | < 64 最佳 | 使用 __launch_bounds__ |
| 共享内存 | 每个Block共享内存 | < 16KB | 减少使用，分阶段重用 |
| 寄存器溢出 | spill store/load | 应该为 0 | 减少寄存器压力 |
| 占用率 | achieved_occupancy | > 50% | 根据情况优化 |

---

## 优化案例

**案例：规约内核占用率只有 25%**

问题描述：
- Block大小：1024 线程
- 每个线程：64 寄存器
- 每个 SM 最大：2048 线程 → 只能跑 2 个 Block
- 实际只跑了 1 个 Block → 占用率 50%？不，等等...
- 哦，还有 1024 线程 / Block × 64 寄存器 / 线程 = 65536 寄存器
- 每个 SM 65536 寄存器 → 只能跑 1 个 Block
- 所以 1024 / 2048 = 50% 占用率

诊断：
1. 寄存器是主要限制因素
2. 每个线程用了太多寄存器
3. Block太大了

优化方案1：减小 Block 大小，改用 512 线程
- 512 × 64 = 32768 寄存器 / Block
- 每个 SM 可以跑 2 个 Block → 1024 线程
- 占用率还是 50%...

优化方案2：使用 __launch_bounds__ 减少寄存器
```cpp
__launch_bounds__(512, 4)  // 期望每个SM跑4个Block
__global__ void reduce_kernel(float* data, int n) {
    // 编译器会限制寄存器使用到 ~32/线程
    // 512 × 32 = 16384 寄存器 / Block
    // 每个SM可以跑4个Block → 2048线程 → 100%占用率 ✅
}
```

结果：
- 占用率从 25% 提升到 80%
- 性能提升 ~1.6 倍

---

## 权衡与注意事项

### 1. 占用率 vs ILP（指令级并行）

```
高占用率                    低占用率
  │                           │
  ├─▶ 更多线程               ├─▶ 每个线程更多寄存器
  ├─▶ 更好隐藏内存延迟      ├─▶ 更好的ILP
  └─▶ 适合内存密集型         └─▶ 适合计算密集型

没有绝对的"最佳"占用率
取决于内核的特性
```

### 2. 占用率 vs 寄存器溢出

```
使用 __launch_bounds__ 时要小心
  │
  ├─▶ 检查 spill stores / spill loads
  │   ├─▶ > 0 表示有寄存器溢出到本地内存
  │   └─▶ 溢出会大幅降低性能
  │
  └─▶ 权衡：
      ├─▶ 轻微溢出 + 高占用率 可能比 无溢出 + 低占用率 好
      └─▶ 严重溢出 → 还不如保持低占用率
```

### 3. 不同工作负载的理想占用率

| 工作负载类型 | 理想占用率范围 | 说明 |
|------------|-------------|------|
| 内存密集型 | 50-75% | 需要足够的线程隐藏延迟 |
| 平衡型 | 40-60% | 均衡即可 |
| 计算密集型 | 25-50% | 不需要太高 |
| 高延迟操作（纹理、原子） | 75%+ | 需要更多线程隐藏延迟 |

---

## 分步优化指南

### 第一步：测量基准性能和占用率

```
1. 记录基准性能
2. 记录 achieved_occupancy
3. 记录每个线程的寄存器数
4. 记录每个Block的共享内存大小
```

### 第二步：尝试调整 Block 大小

```
测试不同的 Block 大小：
  - 64, 128, 256, 512, 1024
  - 记录每个配置的性能和占用率
  - 选择性能最佳的配置
```

### 第三步：如果需要，应用 `__launch_bounds__`

```
如果寄存器是主要限制因素
  1. 尝试 __launch_bounds__(block_size, min_blocks)
  2. 从 min_blocks = 4 开始
  3. 检查是否溢出
  4. 检查性能变化
```

### 第四步：优化共享内存使用

```
如果共享内存是主要限制因素
  1. 减少共享内存使用量
  2. 分阶段重用共享内存
  3. 考虑动态共享内存配置
```

### 第五步：最终验证

```
1. 确认性能确实提升
2. 确认没有引入新问题
3. 确保正确性没有问题
4. 记录最终配置
```

---

## 参考资源

- CUDA C++ Programming Guide: Occupancy
- NVIDIA CUDA Occupancy Calculator (Excel)
- NVIDIA Developer Blog: CUDA Occupancy Explained
- GTC Talk: "Occupancy API: Concepts and Strategies
- CUDA Best Practices Guide: Hardware Multiprocessor Occupancy