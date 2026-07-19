# CUDA 占用率分析与优化

## 目录
1. [什么是占用率](#什么是占用率)
2. [占用率的计算方法](#占用率的计算方法)
3. [影响占用率的因素](#影响占用率的因素)
4. [测量占用率的工具](#测量占用率的工具)
5. [占用率优化实战](#占用率优化实战)
6. [占用率与性能的关系](#占用率与性能的关系)

## 什么是占用率

### 1. 基本概念

**占用率（Occupancy）** 是指 GPU 的每个多处理器（SM）上实际活跃的 warp 数量与理论最大可支持的 warp 数量之比。

```
占用率 = (活跃的 warp 数) / (最大可支持的 warp 数)
```

### 2. 为什么重要

- **隐藏延迟**：高占用率可以更好地隐藏内存访问和指令执行延迟
- **SM 利用率**：高占用率意味着 SM 有更多的线程可以调度
- **吞吐量**：适当的占用率可以最大化指令吞吐量

### 3. 注意事项

⚠️ **高占用率 ≠ 最高性能**
- 过高的占用率可能导致每个线程可用资源减少
- 需要在占用率和可用资源之间取得平衡
- 最佳占用率通常在 50% - 75% 之间

## 占用率的计算方法

### 1. 理论最大占用率

#### 第一步：确定 GPU 硬件参数

```bash
# 查看 GPU 规格
nvidia-smi -q | grep -E "Multiprocessors|Max Threads per Multiprocessor|Total Register|Shared Memory"
```

或者使用 CUDA API 查询：
```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("SM 数量: %d\n", prop.multiProcessorCount);
printf("每个 SM 最大线程数: %d\n", prop.maxThreadsPerMultiProcessor);
printf("每个 SM 寄存器总数: %d\n", prop.regsPerMultiprocessor);
printf("每个 SM 共享内存大小: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
```

#### 第二步：计算核函数资源使用

```bash
# 编译时查看资源使用
nvcc -Xptxas -v -o kernel kernel.cu
```

输出示例：
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z11my_kernelPfS_i' for 'sm_70'
ptxas info    : Function properties for _Z11my_kernelPfS_i
    32 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, 4096 bytes smem, 344 bytes cmem[0]
```

#### 第三步：计算限制因素

**每个 Block 资源使用：**
- 线程数（block_size）
- 寄存器使用（registers_per_thread × block_size）
- 共享内存使用（shared_memory）

**限制 1：线程数限制**
```
每个 SM 最大 Blocks = max_threads_per_sm / block_size
```

**限制 2：寄存器限制**
```
每个 SM 最大 Blocks = total_registers_per_sm / (registers_per_thread × block_size)
```

**限制 3：共享内存限制**
```
每个 SM 最大 Blocks = total_shared_memory_per_sm / shared_memory_per_block
```

**最终每个 SM 可容纳的 Blocks 数 = min(限制 1, 限制 2, 限制 3)**

**占用率 = (Blocks_per_SM × block_size) / max_threads_per_sm**

### 2. 计算示例

假设 GPU 参数（A100）：
- 每个 SM 最大线程数：2048
- 每个 SM 寄存器总数：65536
- 每个 SM 共享内存：163840 bytes

核函数参数：
- Block 大小：256 线程
- 每个线程寄存器：32
- 每个 Block 共享内存：8192 bytes

**计算：**

1. **线程数限制**：2048 / 256 = 8 Blocks
2. **寄存器限制**：65536 / (32 × 256) = 65536 / 8192 = 8 Blocks
3. **共享内存限制**：163840 / 8192 = 20 Blocks

**每个 SM 可容纳的 Blocks = min(8, 8, 20) = 8 Blocks**

**占用率 = (8 × 256) / 2048 = 2048 / 2048 = 100%**

### 3. CUDA 占用率计算器 API

```cpp
#include <iostream>
#include <cuda_runtime.h>

void calculate_occupancy(int block_size, int registers_per_thread, 
                         size_t shared_memory_per_block, int device = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        (const void*)your_kernel,  // 核函数指针
        block_size,
        shared_memory_per_block
    );
    
    float occupancy = (max_blocks_per_sm * block_size) / 
                      static_cast<float>(prop.maxThreadsPerMultiProcessor);
    
    printf("=== 占用率计算 ===\n");
    printf("Block 大小: %d 线程\n", block_size);
    printf("每个线程寄存器: %d\n", registers_per_thread);
    printf("每个 Block 共享内存: %d bytes\n", (int)shared_memory_per_block);
    printf("每个 SM 最大 Blocks: %d\n", max_blocks_per_sm);
    printf("理论占用率: %.2f%%\n", occupancy * 100);
}
```

## 影响占用率的因素

### 1. 寄存器使用

| 寄存器/线程 | Block 大小 256 | Blocks/SM | 占用率 |
|-------------|----------------|-----------|--------|
| 16 | 256 | 16 | 100% |
| 32 | 256 | 8 | 100% |
| 48 | 256 | 5 | 62.5% |
| 64 | 256 | 4 | 50% |
| 128 | 256 | 2 | 25% |

### 2. 共享内存使用

| 共享内存/Block | Block 大小 256 | Blocks/SM | 占用率 |
|----------------|----------------|-----------|--------|
| 4 KB | 256 | 40 | 100% |
| 8 KB | 256 | 20 | 100% |
| 16 KB | 256 | 10 | 100% |
| 32 KB | 256 | 5 | 62.5% |
| 64 KB | 256 | 2 | 25% |

### 3. Block 大小

| Block 大小 | Blocks/SM (线程限制) | 占用率 |
|-----------|----------------------|--------|
| 32 | 64 | 100% |
| 64 | 32 | 100% |
| 128 | 16 | 100% |
| 256 | 8 | 100% |
| 512 | 4 | 100% |
| 1024 | 2 | 100% |
| 2048 | 1 | 100% |

⚠️ **注意**：Block 大小建议为 32 的倍数（通常 128、256 或 512）

## 测量占用率的工具

### 1. nvprof

```bash
# 测量实际占用率
nvprof --metrics achieved_occupancy,occupancy ./your_app

# 查看更多相关指标
nvprof --metrics achieved_occupancy,warps_per_multiprocessor,active_warps ./your_app
```

输出示例：
```
==12345== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   100.0%  10.53ms         1  10.53ms  10.53ms  10.53ms  my_kernel(float*, float*, int)

==12345== Kernel metrics:
   75.00%  achieved_occupancy  Achieved Occupancy
   64      warps_per_multiprocessor  Warps per Multiprocessor
```

### 2. Nsight Compute

```bash
# 详细的占用率分析
ncu --section Occupancy ./your_app

# 查看特定核函数的占用率
ncu -k my_kernel --section Occupancy ./your_app
```

### 3. Nsight Systems

在 Nsight Systems 中：
1. 运行分析
2. 查看 CUDA 核函数详情
3. 检查 "Occupancy" 列

### 4. CUDA Occupancy Calculator

NVIDIA 提供的 Excel 表格工具：
- 下载地址：CUDA Toolkit 安装目录下 `tools/CUDA_Occupancy_Calculator.xls`
- 输入 GPU 架构和核函数参数
- 自动计算占用率

## 占用率优化实战

### 实战 1：寄存器使用优化

#### 问题：寄存器使用过多导致低占用率

```cpp
// 高寄存器使用版本
__global__ void high_reg_kernel(float* data, int n) {
    float a, b, c, d, e, f, g, h;  // 8 个寄存器变量
    float i, j, k, l, m, n, o, p;  // 又 8 个
    float q, r, s, t, u, v, w, x;  // 又 8 个
    
    int idx = blockIdx.x * 256 + threadIdx.x;
    
    if (idx < n) {
        a = data[idx] * 1.0f;
        b = data[idx] * 2.0f;
        c = data[idx] * 3.0f;
        // ... 更多计算
        data[idx] = a + b + c + d;
    }
}
```

#### 优化方案 1：使用 launch_bounds

```cpp
// 限制每个 Block 的最大线程数，允许编译器减少寄存器使用
__global__ void __launch_bounds__(256, 8) optimized_kernel(float* data, int n) {
    float a, b, c, d;
    
    int idx = blockIdx.x * 256 + threadIdx.x;
    
    if (idx < n) {
        a = data[idx] * 1.0f;
        b = data[idx] * 2.0f;
        c = data[idx] * 3.0f;
        d = data[idx] * 4.0f;
        data[idx] = a + b + c + d;
    }
}
```

#### 优化方案 2：重用变量，减少活动变量

```cpp
__global__ void reuse_vars_kernel(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        float sum = 0.0f;
        
        sum += val * 1.0f;  // 重用 val
        sum += val * 2.0f;
        sum += val * 3.0f;
        sum += val * 4.0f;
        
        data[idx] = sum;
    }
}
```

#### 优化方案 3：使用 maxrregcount 编译选项

```bash
# 限制每个线程最多使用 32 个寄存器
nvcc -maxrregcount 32 -o kernel kernel.cu
```

### 实战 2：共享内存优化

#### 问题：共享内存使用过多导致低占用率

```cpp
__global__ void large_shared_kernel(float* data, int n) {
    __shared__ float s_tile1[32][32];  // 4KB
    __shared__ float s_tile2[32][32];  // 4KB
    __shared__ float s_tile3[32][32];  // 4KB
    __shared__ float s_tile4[32][32];  // 4KB
    // 总共 16KB 共享内存
    
    int idx = blockIdx.x * 256 + threadIdx.x;
    // ... 使用这些 tile
}
```

#### 优化方案：分阶段使用共享内存

```cpp
__global__ void staged_shared_kernel(float* data, int n) {
    __shared__ float s_tile[32][32];  // 只用 4KB，分阶段使用
    
    int idx = blockIdx.x * 256 + threadIdx.x;
    
    // 阶段 1：使用 s_tile 处理第一部分
    // ...
    
    __syncthreads();
    
    // 阶段 2：重用 s_tile 处理第二部分
    // ...
    
    __syncthreads();
    
    // 阶段 3：继续重用
    // ...
}
```

### 实战 3：Block 大小优化

#### 测试不同 Block 大小的性能

```cpp
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

template <int BLOCK_SIZE>
__global__ void test_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        val = val * val + val * 2.0f + 1.0f;
        output[idx] = val;
    }
}

template <int BLOCK_SIZE>
float test_occupancy(int n) {
    float *d_input, *d_output;
    int bytes = n * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes));
    
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 预热
    test_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(d_output, d_input, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    const int iterations = 100;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        test_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(d_output, d_input, n);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    
    return time_ms / iterations;
}

int main() {
    const int n = 1 << 24;  // 16M elements
    
    printf("=== Block 大小性能测试 ===\n");
    printf("数据量: %d elements = %.1f MB\n\n", n, n * sizeof(float) / (1024.0f * 1024.0f));
    
    printf("Block 大小 | 平均时间 (ms) | 带宽 (GB/s)\n");
    printf("-----------|---------------|-------------\n");
    
    float time_64 = test_occupancy<64>(n);
    printf("        64 |    %10.3f   |  %10.2f\n", time_64, 
           (2.0f * n * sizeof(float) / 1e9) / (time_64 / 1000.0f));
    
    float time_128 = test_occupancy<128>(n);
    printf("       128 |    %10.3f   |  %10.2f\n", time_128,
           (2.0f * n * sizeof(float) / 1e9) / (time_128 / 1000.0f));
    
    float time_256 = test_occupancy<256>(n);
    printf("       256 |    %10.3f   |  %10.2f\n", time_256,
           (2.0f * n * sizeof(float) / 1e9) / (time_256 / 1000.0f));
    
    float time_512 = test_occupancy<512>(n);
    printf("       512 |    %10.3f   |  %10.2f\n", time_512,
           (2.0f * n * sizeof(float) / 1e9) / (time_512 / 1000.0f));
    
    float time_1024 = test_occupancy<1024>(n);
    printf("      1024 |    %10.3f   |  %10.2f\n", time_1024,
           (2.0f * n * sizeof(float) / 1e9) / (time_1024 / 1000.0f));
    
    return 0;
}
```

编译运行：
```bash
nvcc -O3 -o block_size_test block_size_test.cu
./block_size_test
```

## 占用率与性能的关系

### 1. 典型的占用率-性能曲线

```
性能 ^
     |                    .
     |                  .
     |                .
     |              .
     |            .
     |          .
     |        .
     |      .
     |    .
     |  .
     +------------------------> 占用率
        0%   25%   50%   75%  100%
```

**特征：**
- 低占用率时，性能随占用率快速提升
- 达到一定阈值（通常 50%-75%）后，性能提升放缓
- 过高的占用率可能导致性能下降（资源争用）

### 2. 不同工作负载的最佳占用率

| 工作负载类型 | 最佳占用率范围 | 原因 |
|-------------|---------------|------|
| **计算密集型** | 25% - 50% | 指令级并行性高，不需要太多 warp 隐藏延迟 |
| **内存密集型** | 50% - 75% | 需要更多 warp 来隐藏内存访问延迟 |
| **延迟敏感型** | 75% - 100% | 高延迟操作需要大量上下文切换 |

### 3. 占用率优化决策树

```
开始
  |
  v
测量实际占用率
  |
  +-- 占用率低 (< 50%)？
  |     |
  |     +-- 是
  |     |   |
  |     |   +-- 是寄存器限制？
  |     |   |     |
  |     |   |     +-- 是 -> 使用 launch_bounds / maxrregcount
  |     |   |     |
  |     |   |     +-- 否
  |     |   |           |
  |     |   |           +-- 是共享内存限制？
  |     |   |                 |
  |     |   |                 +-- 是 -> 优化共享内存使用
  |     |   |                 |
  |     |   |                 +-- 否 -> 调整 Block 大小
  |     |   |
  |     |   +-- 性能是否提升？
  |     |         |
  |     |         +-- 是 -> 完成
  |     |         |
  |     |         +-- 否 -> 尝试其他优化
  |     |
  |     +-- 否
  |         |
  |         +-- 占用率是否过高 (> 90%)？
  |               |
  |               +-- 是 -> 尝试增加每线程工作量
  |               |
  |               +-- 否 -> 占用率合适，检查其他瓶颈
  |
  v
完成
```

## 占用率优化的完整示例

### 步骤 1：识别问题

```bash
# 编译并查看资源使用
nvcc -Xptxas -v -o my_app my_app.cu

# 测量实际占用率
nvprof --metrics achieved_occupancy ./my_app
```

输出显示：
- 每个线程使用 64 个寄存器
- Block 大小：256
- 实际占用率：25%

### 步骤 2：分析限制因素

对于每个 SM：
- 最大线程数：2048
- 总寄存器：65536
- 寄存器限制：65536 / (64 × 256) = 4 Blocks
- 占用率：(4 × 256) / 2048 = 50%（但实际只有 25%，说明还有其他问题）

### 步骤 3：应用优化

**优化 1：使用 launch_bounds**
```cpp
__global__ void __launch_bounds__(256, 8) optimized_kernel(...) {
    // ... 代码
}
```

**优化 2：限制寄存器数量**
```bash
nvcc -maxrregcount 32 -o my_app_optimized my_app.cu
```

**优化 3：调整 Block 大小**
```cpp
// 尝试 512 线程/Block
dim3 block_size(512);
dim3 grid_size((n + 512 - 1) / 512);
```

### 步骤 4：验证优化效果

```bash
# 重新测量
nvprof --metrics achieved_occupancy,ipc,sm_efficiency ./my_app_optimized

# 比较性能
nvprof ./my_app
nvprof ./my_app_optimized
```

## 占用率优化检查清单

- [ ] 使用 `-Xptxas -v` 查看核函数资源使用
- [ ] 测量实际占用率（achieved_occupancy）
- [ ] 分析占用率限制因素（寄存器/共享内存/线程数）
- [ ] 尝试不同的 Block 大小（128, 256, 512, 1024）
- [ ] 必要时使用 `launch_bounds` 限制寄存器使用
- [ ] 优化共享内存使用，避免过度分配
- [ ] 平衡占用率和每线程工作量
- [ ] 验证优化后的实际性能提升
- [ ] 注意：不要为了占用率而牺牲其他优化

## 总结

占用率优化的关键要点：

1. **理解占用率**：占用率是活跃 warp 与最大可支持 warp 的比率
2. **不是越高越好**：最佳占用率通常在 50%-75% 之间
3. **多重限制**：寄存器、共享内存、线程数都会限制占用率
4. **工具测量**：使用 nvprof 和 Nsight Compute 测量实际占用率
5. **综合优化**：结合寄存器、共享内存、Block 大小进行综合优化
6. **实际验证**：始终验证优化后的性能变化

记住：**占用率是手段，不是目的**。最终目标是提升程序性能，而不是单纯追求高占用率！
