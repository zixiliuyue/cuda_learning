# 反模式 04: 低占用率 (Low Occupancy)

---

## 1. 问题描述

### 1.1 什么是占用率

**定义**：
占用率是指SM（流式多处理器）上活跃的warp数量与理论最大warp数量的比值。

```
占用率 = (活跃warp数) / (SM最大warp数)
```

**典型硬件参数**（以A100为例）：
- 每个SM最大warp数：64
- 每个SM最大线程数：2048
- 每个SM最大Block数：32
- 每个SM寄存器数：65536个32位寄存器
- 每个SM共享内存：164KB

### 1.2 为什么占用率重要

**延迟隐藏原理**：
当一个warp因为内存访问等原因停顿时，SM可以切换到其他就绪的warp执行。足够高的占用率可以确保SM在任何时候都有可执行的warp，从而隐藏延迟。

```
高占用率 → 更多可切换的warp → 更好的延迟隐藏 → 更高的执行效率
```

### 1.3 性能影响

| 占用率区间 | 性能影响 |
|-----------|---------|
| < 25% | 严重影响，无法隐藏延迟 |
| 25% - 50% | 中等影响，部分场景性能下降 |
| 50% - 75% | 良好，通常足够隐藏延迟 |
| > 75% | 优秀，边际收益递减 |

**注意**：占用率不是越高越好！超过某个阈值后，继续提高占用率的收益很小，甚至可能因为寄存器限制导致性能下降。

---

## 2. 限制占用率的三个因素

### 2.1 因素一：每个Block的线程数

```
每个Block的线程数 → 决定每个SM能放多少个Block
```

**例子**：
- Block大小 = 128线程 → 每个SM最多 2048 / 128 = 16个Block
- Block大小 = 256线程 → 每个SM最多 2048 / 256 = 8个Block
- Block大小 = 1024线程 → 每个SM最多 2048 / 1024 = 2个Block

**问题代码**：
```cpp
// ❌ Block太小，调度开销大
dim3 block(32);  // 每个Block只有1个warp
dim3 grid(1000);

// ❌ Block太大，可能受寄存器/共享内存限制
dim3 block(1024);  // 最大Block大小
dim3 grid(100);
```

### 2.2 因素二：每个线程的寄存器使用

```
每个线程的寄存器数 × Block大小 = 每个Block的总寄存器数
→ 决定受寄存器限制的Blocks/SM
```

**例子**：
- 每个线程使用 64 个寄存器
- Block大小 = 256 线程
- 每个Block需要：64 × 256 = 16384 个寄存器
- 每个SM最多：65536 / 16384 = 4 个Block
- 活跃warp数 = 4 × (256 / 32) = 32
- 占用率 = 32 / 64 = 50%

**问题代码 - 寄存器滥用**：
```cpp
__global__ void heavy_register_kernel(float* data, int n) {
    // ❌ 大量的寄存器变量
    float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
    float arr0[8], arr1[8], arr2[8], arr3[8];  // 大量数组
    
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        // 复杂计算，使用大量临时变量
        // ...
    }
}
```

### 2.3 因素三：每个Block的共享内存使用

```
每个Block的共享内存使用 → 决定受共享内存限制的Blocks/SM
```

**例子**：
- 每个Block使用 32KB 共享内存
- 每个SM有 164KB 共享内存
- 每个SM最多：164 / 32 ≈ 5 个Block
- Block大小 = 256 线程
- 活跃warp数 = 5 × (256 / 32) = 40
- 占用率 = 40 / 64 = 62.5%

**问题代码 - 共享内存过度使用**：
```cpp
__global__ void heavy_shared_memory_kernel(float* data, int n) {
    // ❌ 过大的共享内存数组
    __shared__ float shared0[4096];
    __shared__ float shared1[4096];
    __shared__ float shared2[4096];
    __shared__ float shared3[4096];
    // 总计: 4 × 4096 × 4B = 64KB/Block
    
    int tid = threadIdx.x;
    // ...
}
```

---

## 3. 计算和分析占用率

### 3.1 理论计算方法

**三步骤计算占用率**：

```cpp
// 输入参数
int block_size = 256;           // 每个Block的线程数
int registers_per_thread = 32;  // 每个线程的寄存器数
int shared_per_block = 16384;   // 每个Block的共享内存（字节）

// 硬件参数（A100为例）
const int MAX_WARPS_PER_SM = 64;
const int MAX_REGISTERS_PER_SM = 65536;
const int MAX_SHARED_PER_SM = 164 * 1024;  // 164KB

// 步骤1: 计算受线程数限制的Blocks/SM
int blocks_by_threads = 2048 / block_size;

// 步骤2: 计算受寄存器限制的Blocks/SM
int registers_per_block = registers_per_thread * block_size;
int blocks_by_regs = MAX_REGISTERS_PER_SM / registers_per_block;

// 步骤3: 计算受共享内存限制的Blocks/SM
int blocks_by_shared = MAX_SHARED_PER_SM / shared_per_block;

// 实际Blocks/SM = 三者中的最小值
int actual_blocks = min(min(blocks_by_threads, blocks_by_regs), blocks_by_shared);

// 计算占用率
int active_warps = actual_blocks * (block_size / 32);
float occupancy = (float)active_warps / MAX_WARPS_PER_SM;

std::cout << "理论占用率: " << (occupancy * 100) << "%" << std::endl;
```

### 3.2 使用CUDA Occupancy Calculator

NVIDIA提供了官方的Excel表格，可以输入参数计算占用率。

**在线工具**：
- CUDA Toolkit 自带：`CUDA Toolkit/Samples/1_Utilities/OccupancyCalculator`
- 在线版本：NVIDIA Developer网站

**输入参数**：
- 计算能力（如 7.0, 8.0）
- Block大小（线程数）
- 每个线程的寄存器数
- 每个Block的共享内存（字节）

**输出**：
- 理论占用率
- 各个因素的限制情况

---

## 4. 解决方案

### 4.1 方案一：调整Block大小

**原则**：
- Block大小必须是 32 的倍数（warp大小）
- 推荐范围：128 - 512 线程
- 避免小于 64 或大于 1024

```cpp
// ✅ 推荐的Block大小
dim3 block_128(128);   // 通用，占用率通常较好
dim3 block_256(256);   // 常用，平衡各项因素
dim3 block_512(512);   // 寄存器需求小时使用

// 2D Block推荐
dim3 block2d_16x16(16, 16);  // 256线程
dim3 block2d_32x8(32, 8);    // 256线程，适合某些访问模式

// ❌ 避免
dim3 block_bad_1(50);   // 不是32倍数
dim3 block_bad_2(32);   // 太小，调度开销大
```

---

### 4.2 方案二：限制寄存器使用

**方法1: 使用编译指令 `__launch_bounds__`**

```cpp
// ✅ 明确指定launch bounds，限制寄存器使用
__launch_bounds__(256, 8)  // Block大小256，每个SM至少8个Block
__global__ void optimized_kernel(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        // 编译器会自动限制每个线程的寄存器使用
        // 以满足每个SM至少8个Block的要求
    }
}
```

**方法2: 减少寄存器压力**

```cpp
// ❌ 原始代码：大量临时变量
__global__ void bad_registers(float* data, int n) {
    int idx = ...;
    float t0, t1, t2, t3, t4, t5, t6, t7;
    float t8, t9, t10, t11, t12, t13, t14, t15;
    // ...
}

// ✅ 优化后：重用变量，减少活跃变量
__global__ void good_registers(float* data, int n) {
    int idx = ...;
    float temp;  // 一个变量重复使用
    
    temp = compute_step1(data[idx]);
    data[idx] = temp;
    
    temp = compute_step2(data[idx]);  // 重用temp
    data[idx] = temp;
    // ...
}

// ✅ 优化：把大数组放到共享内存或全局内存
__global__ void array_to_shared(float* data, int n) {
    __shared__ float shared_arr[128];  // 共享内存，不占寄存器
    int tid = threadIdx.x;
    
    shared_arr[tid] = ...;
    __syncthreads();
}
```

---

### 4.3 方案三：优化共享内存使用

**方法1: 减少共享内存使用量**

```cpp
// ❌ 使用4个独立的共享内存数组
__global__ void four_shared_arrays(float* data, int n) {
    __shared__ float s0[512];
    __shared__ float s1[512];
    __shared__ float s2[512];
    __shared__ float s3[512];  // 总计 8KB
    // ...
}

// ✅ 重用同一块共享内存
__global__ void reuse_shared(float* data, int n) {
    __shared__ float shared[512];  // 只需 2KB
    
    // 阶段1: 使用shared
    stage1_compute(shared, ...);
    __syncthreads();
    
    // 阶段2: 继续使用同一个shared（覆盖原有数据）
    stage2_compute(shared, ...);
    __syncthreads();
}
```

**方法2: 使用动态共享内存**

```cpp
// ✅ 动态共享内存，根据需要分配大小
__global__ void dynamic_shared(float* data, int n) {
    extern __shared__ float shared[];  // 动态大小
    
    int tid = threadIdx.x;
    // ...
}

// 启动时指定共享内存大小
kernel<<<grid, block, shared_size>>>(data, n);
```

---

### 4.4 方案四：使用模板和编译期优化

```cpp
// ✅ 使用模板参数控制block大小，便于尝试不同配置
template<int BLOCK_SIZE>
__global__ void template_kernel(float* data, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) {
        // ...
    }
}

// 尝试不同的Block大小
// 编译多个版本，运行时选择最快的
void run_kernel(int version, float* data, int n) {
    dim3 grid((n + 255) / 256);
    
    switch(version) {
        case 128:
            template_kernel<128><<<grid * 2, 128>>>(data, n);
            break;
        case 256:
            template_kernel<256><<<grid, 256>>>(data, n);
            break;
        case 512:
            template_kernel<512><<<grid / 2, 512>>>(data, n);
            break;
    }
}
```

---

## 5. 代码对比：不同占用率的性能差异

### 5.1 性能测试代码

```cpp
#include <iostream>
#include <chrono>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// ========== 低占用率版本 ==========
__launch_bounds__(1024, 1)  // 每个SM只有1个Block，占用率低
__global__ void low_occupancy_kernel(float* data, int n) {
    int idx = blockIdx.x * 1024 + threadIdx.x;
    
    // 使用很多寄存器
    float r0, r1, r2, r3, r4, r5, r6, r7;
    float r8, r9, r10, r11, r12, r13, r14, r15;
    float r16, r17, r18, r19, r20, r21, r22, r23;
    float r24, r25, r26, r27, r28, r29, r30, r31;
    
    if (idx < n) {
        r0 = data[idx];
        r1 = r0 * 2.0f;
        r2 = r1 + 3.0f;
        // ... 使用所有寄存器
        data[idx] = r31;
    }
}

// ========== 高占用率版本 ==========
__launch_bounds__(256, 8)  // 每个SM至少8个Block，占用率高
__global__ void high_occupancy_kernel(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    
    // 较少的寄存器使用
    float temp;
    
    if (idx < n) {
        temp = data[idx];
        temp = temp * 2.0f;
        temp = temp + 3.0f;
        // ...
        data[idx] = temp;
    }
}

int main() {
    const int N = 1 << 24;  // 16M元素
    
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    
    // 测试低占用率版本
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        dim3 grid_low((N + 1023) / 1024);
        dim3 block_low(1024);
        low_occupancy_kernel<<<grid_low, block_low>>>(d_data, N);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double time_low = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    // 测试高占用率版本
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        dim3 grid_high((N + 255) / 256);
        dim3 block_high(256);
        high_occupancy_kernel<<<grid_high, block_high>>>(d_data, N);
    }
    
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double time_high = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    std::cout << "低占用率版本: " << time_low << " ms" << std::endl;
    std::cout << "高占用率版本: " << time_high << " ms" << std::endl;
    std::cout << "加速比: " << time_low / time_high << "x" << std::endl;
    
    CHECK_CUDA(cudaFree(d_data));
    return 0;
}
```

### 5.2 预期结果

| 版本 | 占用率(估计) | 执行时间 | 说明 |
|------|-------------|---------|------|
| 低占用率 | ~25% | ~2.0 ms | 无法有效隐藏延迟 |
| 高占用率 | ~75% | ~0.8 ms | 良好的延迟隐藏 |
| **加速比** | | **~2.5x** | |

---

## 6. 检测与分析方法

### 6.1 使用Nsight Compute分析

```bash
ncu --section Occupancy -o occupancy_profile ./your_app
```

**关键指标**：
- `achieved_occupancy`：实际达到的占用率（最重要！）
- `theoretical_occupancy`：理论最大占用率
- `block_limit`：受Block大小限制的活跃Blocks/SM
- `register_limit`：受寄存器限制的活跃Blocks/SM
- `shared_memory_limit`：受共享内存限制的活跃Blocks/SM

**分析步骤**：
1. 查看 `achieved_occupancy`，判断是否低于 50%
2. 查看哪个因素是主要限制（block/register/shared）
3. 针对性优化

---

### 6.2 使用nvprof查看

```bash
nvprof --metrics achieved_occupancy,register_per_thread,shared_memory ./your_app
```

---

### 6.3 编译时获取寄存器使用信息

```bash
nvcc --ptxas-options=-v -arch=sm_80 your_kernel.cu
```

**输出示例**：
```
ptxas info    : Compiling entry function '_Z15my_kernelPf' for 'sm_80'
ptxas info    : Function properties for _Z15my_kernelPf
    32 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 48 registers, 1024 bytes smem, 328 bytes cmem[0]
                   ↑                 ↑
              每个线程48寄存器  每个Block 1KB共享内存
```

---

## 7. 最佳实践总结

### 7.1 占用率优化的一般流程

1. **测量**：使用Nsight Compute获取实际占用率
2. **诊断**：确定哪个因素是瓶颈（寄存器/共享内存/Block大小）
3. **优化**：针对性优化瓶颈因素
4. **验证**：重新测量，评估性能变化

**注意**：总是以性能测量为准，不要盲目追求高占用率！

---

### 7.2 代码审查检查清单

- [ ] Block大小是否是32的倍数？
- [ ] Block大小是否在128-512范围内？
- [ ] 是否使用 `--ptxas-options=-v` 检查寄存器使用？
- [ ] 每个线程的寄存器数是否 > 64？
- [ ] 共享内存使用是否合理？
- [ ] 是否尝试过不同的Block大小？
- [ ] 是否使用 `__launch_bounds__` 进行控制？

---

### 7.3 常见误区

**误区1: 占用率越高越好**
- 错误！占用率超过50-70%后，边际收益迅速递减
- 有时为了更多的每个线程寄存器，适当降低占用率反而更快

**误区2: 占用率低一定性能差**
- 不一定！如果内核是计算密集型（无内存访问），低占用率也可能很好
- 如果内存访问很少，不需要太高的占用率来隐藏延迟

**误区3: Block越大性能越好**
- 不一定！大Block可能受寄存器限制，占用率反而更低
- 256通常是很好的平衡点

---

## 8. 进阶话题

### 8.1 占用率与指令级并行 (ILP)

**权衡关系**：
- 高占用率 → 更多warp → 更多线程级并行 (TLP)
- 低占用率 → 每个线程更多寄存器 → 更多指令级并行 (ILP)

**选择策略**：
- 内存密集型内核：优先保证足够的占用率（50%+）
- 计算密集型内核：可以接受较低的占用率，追求更高的ILP

---

### 8.2 溢出（Spilling）的影响

当寄存器不够用时，编译器会把数据"溢出"到全局内存（local memory）。溢出会严重影响性能。

**检测溢出**：
```bash
nvcc --ptxas-options=-v ...
# 查看: spill stores / spill loads
```

**解决溢出**：
1. 减少每个线程的寄存器使用
2. 增大Block大小（减少每个SM的Blocks，变相增加每个线程的寄存器配额）
3. 使用 `__launch_bounds__` 显式控制

---

### 8.3 动态并行中的占用率

子内核启动也会占用SM资源。使用动态并行时要注意：
- 避免同时启动太多子内核，超出SM的承载能力
- 子内核的Block大小也需要合理选择
- 考虑父内核和子内核的资源竞争

---

## 9. 相关资源

- CUDA C++ Programming Guide: Occupancy
- NVIDIA Developer Blog: CUDA Occupancy Explained
- CUDA Best Practices Guide: Occupancy Optimization
- GTC Talks: Advanced CUDA Optimization Techniques
- CUDA Occupancy Calculator Excel Spreadsheet
