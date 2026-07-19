# CUDA 内存性能分析

## 目录
1. [内存层次结构概述](#内存层次结构概述)
2. [内存带宽测试](#内存带宽测试)
3. [全局内存访问模式分析](#全局内存访问模式分析)
4. [共享内存 Bank 冲突检测](#共享内存-bank-冲突检测)
5. [本地内存使用分析](#本地内存使用分析)
6. [内存优化实战](#内存优化实战)

## 内存层次结构概述

CUDA GPU 具有多层内存层次结构，每种内存具有不同的特性：

| 内存类型 | 位置 | 容量 | 延迟 | 带宽 | 作用域 | 是否缓存 |
|----------|------|------|------|------|--------|----------|
| **寄存器** | SM 内 | ~64KB/SM | ~1周期 | 极高 | 线程 | 否 |
| **共享内存** | SM 内 | ~48KB/SM | ~1-2周期 | 高 | Block | 否 |
| **L1 缓存** | SM 内 | ~16-48KB/SM | ~10-20周期 | 高 | SM | 是 |
| **L2 缓存** | 全局 | ~2-6MB | ~50-80周期 | 中 | 全局 | 是 |
| **全局内存** | DRAM | 数 GB | ~200-400周期 | 较低 | 全局 | 可选 |
| **本地内存** | DRAM | 无限制 | ~200-400周期 | 较低 | 线程 | 是 |
| **常量内存** | DRAM | 64KB | ~200-400周期 | 较低 | 全局 | 是 |
| **纹理内存** | DRAM | 无限制 | ~200-400周期 | 较低 | 全局 | 是 |

## 内存带宽测试

### 1. 理论带宽计算

```bash
# 查看 GPU 规格
nvidia-smi -q | grep -E "Product Name|Memory Clock|Memory Bus Width"
```

**理论带宽计算公式：**
```
理论带宽 (GB/s) = (内存频率 MHz × 总线宽度位 × 2(DDR)) / 8 / 1000
```

**示例：**
- 内存频率：5005 MHz
- 总线宽度：384-bit
- 理论带宽 = (5005 × 384 × 2) / 8 / 1000 = ~480 GB/s

### 2. 实测带宽测试代码

```cpp
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void copy_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void copy_kernel_stride(float* dst, const float* src, int n, int stride) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx * stride < n) {
        dst[idx * stride] = src[idx * stride];
    }
}

void test_bandwidth(int size_mb) {
    int n = size_mb * 1024 * 1024 / sizeof(float);
    int bytes = n * sizeof(float);
    
    float *h_src, *h_dst;
    float *d_src, *d_dst;
    
    h_src = (float*)malloc(bytes);
    h_dst = (float*)malloc(bytes);
    
    for (int i = 0; i < n; ++i) {
        h_src[i] = static_cast<float>(i);
    }
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_src, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dst, bytes));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // 预热
    copy_kernel<<<grid_size, block_size>>>(d_dst, d_src, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    printf("=== 内存带宽测试 (数据量: %d MB) ===\n", size_mb);
    
    // 测试设备到设备复制
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    copy_kernel<<<grid_size, block_size>>>(d_dst, d_src, n);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
    
    float bandwidth_gb_s = (2.0f * bytes / (1024.0f * 1024.0f * 1024.0f)) / (time_ms / 1000.0f);
    printf("设备到设备复制带宽: %.2f GB/s (耗时: %.3f ms)\n", bandwidth_gb_s, time_ms);
    
    // 测试不同步长的访问
    printf("\n=== 不同步长访问带宽测试 ===\n");
    int strides[] = {1, 2, 4, 8, 16, 32};
    for (int stride : strides) {
        int stride_n = n / stride;
        int stride_grid = (stride_n + block_size - 1) / block_size;
        
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        copy_kernel_stride<<<stride_grid, block_size>>>(d_dst, d_src, n, stride);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
        float data_gb = (float)bytes / (1024.0f * 1024.0f * 1024.0f) / stride;
        bandwidth_gb_s = (2.0f * data_gb) / (time_ms / 1000.0f);
        printf("步长 %2d: 带宽 %.2f GB/s (耗时: %.3f ms)\n", stride, bandwidth_gb_s, time_ms);
    }
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_src));
    CHECK_CUDA_ERROR(cudaFree(d_dst));
    free(h_src);
    free(h_dst);
}

int main() {
    test_bandwidth(128);  // 128MB 数据
    return 0;
}
```

### 3. 使用性能分析工具测量带宽

```bash
# 编译测试程序
nvcc -O3 -o bandwidth_test bandwidth_test.cu

# 使用 nvprof 测量内存事务
nvprof --metrics gld_transactions,gst_transactions,dram_read_throughput,dram_write_throughput ./bandwidth_test

# 使用 Nsight Compute 详细分析
ncu --metrics l1_cache_global_hit_rate,l2_cache_read_hit_rate,gld_efficiency ./bandwidth_test
```

## 全局内存访问模式分析

### 1. 合并访问 vs 非合并访问

#### 良好的访问模式（合并访问）
```cpp
__global__ void good_access_pattern(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // 连续访问，完美合并
    }
}
```

#### 不良的访问模式（非合并访问）
```cpp
__global__ void bad_access_pattern(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx * 32 < n) {
        data[idx * 32] = data[idx * 32] * 2.0f;  // 大步长，非合并
    }
}
```

### 2. 使用性能指标分析访问模式

| 指标 | 描述 | 合并访问 | 非合并访问 |
|------|------|----------|------------|
| **gld_efficiency** | 全局内存加载效率 | > 90% | < 50% |
| **gld_transactions** | 全局内存加载事务数 | 少 | 多 |
| **gld_transactions_per_request** | 每次请求的事务数 | ~1 | >> 1 |
| **l1_cache_global_hit_rate** | L1 缓存命中率 | 可变 | 通常较低 |

### 3. 性能分析命令

```bash
# 分析内存访问效率
nvprof --metrics gld_efficiency,gst_efficiency,gld_transactions_per_request ./your_app

# 查看内存访问模式详情
ncu --section MemoryWorkloadAnalysis ./your_app
```

### 4. 二维数组访问优化

**问题代码：**
```cpp
__global__ void bad_2d_access(float* mat, int rows, int cols) {
    int row = blockIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    if (row < rows && col < cols) {
        mat[col * rows + row] += 1.0f;  // 按列访问，非合并
    }
}
```

**优化代码：**
```cpp
__global__ void good_2d_access(float* mat, int rows, int cols) {
    int row = blockIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    if (row < rows && col < cols) {
        mat[row * cols + col] += 1.0f;  // 按行访问，合并
    }
}
```

## 共享内存 Bank 冲突检测

### 1. Bank 冲突原理

- Kepler 及以后架构：每个 SM 有 32 个共享内存 Bank
- 每个 Bank 带宽为 4 bytes/周期
- 当 warp 内多个线程访问同一 Bank 的不同地址时，发生 Bank 冲突
- 冲突会导致访问串行化，降低带宽

### 2. Bank 冲突示例代码

```cpp
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void bank_conflict_example(float* output) {
    __shared__ float s_data[32];
    
    int tid = threadIdx.x;
    
    // 无冲突：每个线程访问不同的 bank
    s_data[tid] = static_cast<float>(tid);
    __syncthreads();
    
    // 有冲突：步长为 2，线程 0 和 16 同时访问 bank 0
    float val = s_data[tid * 2 % 32];
    __syncthreads();
    
    output[tid] = val;
}

__global__ void no_bank_conflict_example(float* output) {
    __shared__ float s_data[32 + 1];  // 添加 padding
    
    int tid = threadIdx.x;
    
    s_data[tid] = static_cast<float>(tid);
    __syncthreads();
    
    // 无冲突：使用 padding 避免 bank 冲突
    float val = s_data[(tid * 2) % 33];
    __syncthreads();
    
    output[tid] = val;
}

void test_bank_conflict() {
    float *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, 32 * sizeof(float)));
    
    printf("=== 共享内存 Bank 冲突测试 ===\n");
    
    // 预热
    bank_conflict_example<<<1, 32>>>(d_output);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // 测试有冲突版本
    const int iterations = 10000;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        bank_conflict_example<<<1, 32>>>(d_output);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
    printf("有冲突版本: %.3f ms (%.3f us/次)\n", time_ms, time_ms * 1000 / iterations);
    
    // 测试无冲突版本
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        no_bank_conflict_example<<<1, 32>>>(d_output);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
    printf("无冲突版本: %.3f ms (%.3f us/次)\n", time_ms, time_ms * 1000 / iterations);
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

int main() {
    test_bank_conflict();
    return 0;
}
```

### 3. 检测 Bank 冲突的性能指标

```bash
# 编译测试
nvcc -O3 -o bank_conflict_test bank_conflict_test.cu

# 使用 nvprof 检测 bank 冲突事件
nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict ./bank_conflict_test

# 使用 Nsight Compute 详细分析
ncu --metrics shared_efficiency,shared_ld_bank_conflict,shared_st_bank_conflict ./bank_conflict_test
```

### 4. 常见 Bank 冲突模式及解决方法

| 访问模式 | 是否冲突 | 解决方法 |
|----------|----------|----------|
| `s_data[tid]` | 无 | - |
| `s_data[tid + 1]` | 无 | - |
| `s_data[tid * 2]` | 有 (2-way) | 使用 padding |
| `s_data[tid * 4]` | 有 (4-way) | 使用 padding |
| `s_data[tid * n]` | n 是 2 的幂时有冲突 | 调整步长或使用 padding |
| `s_data[random_idx]` | 可能有冲突 | 重新组织数据 |

## 本地内存使用分析

### 1. 本地内存产生的原因

- 寄存器溢出（寄存器不足）
- 动态索引的数组
- 大于寄存器容量的结构体

### 2. 检测本地内存使用

```bash
# 编译时显示寄存器使用情况
nvcc -Xptxas -v -o your_app your_app.cu

# 使用 nvprof 检测本地内存访问
nvprof --metrics local_load_transactions,local_store_transactions ./your_app

# 使用 Nsight Compute 查看
ncu --metrics local_memory_overhead ./your_app
```

### 3. 减少本地内存使用的方法

**问题代码（寄存器溢出）：**
```cpp
__global__ void kernel_with_spill() {
    float temp[128];  // 大数组，可能溢出到本地内存
    int idx = threadIdx.x;
    
    for (int i = 0; i < 128; ++i) {
        temp[i] = ...;
    }
}
```

**优化代码（使用共享内存）：**
```cpp
__global__ void kernel_with_shared() {
    __shared__ float s_temp[128];  // 使用共享内存
    int idx = threadIdx.x;
    
    for (int i = 0; i < 128; ++i) {
        s_temp[i] = ...;
    }
    __syncthreads();
}
```

## 内存优化实战

### 实战 1：矩阵转置的内存优化

#### 原始版本（非合并访问）
```cpp
__global__ void transpose_naive(float* output, const float* input, int width, int height) {
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];  // 写操作非合并
    }
}
```

#### 优化版本（使用共享内存）
```cpp
__global__ void transpose_optimized(float* output, const float* input, int width, int height) {
    __shared__ float tile[32][32 + 1];  // +1 避免 bank 冲突
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // 合并读入到共享内存
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // 从共享内存写出，合并写
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < height && y < width) {
        output[x * width + y] = tile[threadIdx.x][threadIdx.y];
    }
}
```

#### 性能分析对比
```bash
# 编译两个版本
nvcc -O3 -D NAIVE -o transpose_naive transpose.cu
nvcc -O3 -o transpose_optimized transpose.cu

# 分析内存效率
nvprof --metrics gld_efficiency,gst_efficiency ./transpose_naive
nvprof --metrics gld_efficiency,gst_efficiency ./transpose_optimized

# 分析执行时间
nvprof ./transpose_naive
nvprof ./transpose_optimized
```

### 实战 2：规约操作的内存优化

#### 原始版本
```cpp
__global__ void reduce_naive(float* output, const float* input, int n) {
    __shared__ float s_data[256];
    
    int tid = threadIdx.x;
    s_data[tid] = 0.0f;
    
    for (int i = tid; i < n; i += 256) {
        s_data[tid] += input[i];
    }
    __syncthreads();
    
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];  // 有 bank 冲突
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *output = s_data[0];
    }
}
```

#### 优化版本（无 Bank 冲突）
```cpp
__global__ void reduce_optimized(float* output, const float* input, int n) {
    __shared__ float s_data[256];
    
    int tid = threadIdx.x;
    s_data[tid] = 0.0f;
    
    for (int i = tid; i < n; i += 256) {
        s_data[tid] += input[i];
    }
    __syncthreads();
    
    // 顺序规约，无 bank 冲突
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *output = s_data[0];
    }
}
```

## 内存性能检查清单

### 编译阶段检查
- [ ] 使用 `-Xptxas -v` 查看寄存器和本地内存使用
- [ ] 检查是否有大量本地内存使用

### 运行时性能分析
- [ ] 全局内存加载效率 > 80%
- [ ] 全局内存存储效率 > 80%
- [ ] 共享内存加载效率 > 90%
- [ ] 共享内存 bank 冲突数 = 0
- [ ] 本地内存事务数 = 0
- [ ] L2 缓存命中率合理

### 代码优化检查
- [ ] 数组访问使用连续索引
- [ ] 二维数组按行优先访问
- [ ] 共享内存使用合理的 padding
- [ ] 大数组使用共享内存而非本地内存
- [ ] 内存访问模式考虑 warp 内的合并性

## 总结

内存性能优化的关键原则：

1. **合并访问**：确保 warp 内线程访问连续的内存地址
2. **利用缓存**：合理使用 L1/L2 缓存和共享内存
3. **避免冲突**：消除共享内存 bank 冲突
4. **减少溢出**：避免寄存器溢出到本地内存
5. **数据重排**：调整数据布局以优化访问模式

记住：**内存访问通常是 CUDA 程序的主要性能瓶颈**，优先优化内存访问模式！
