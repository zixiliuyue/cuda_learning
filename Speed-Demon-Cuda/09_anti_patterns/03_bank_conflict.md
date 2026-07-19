# 反模式 03: 共享内存Bank冲突 (Shared Memory Bank Conflict)

---

## 1. 问题描述

### 1.1 什么是Bank

共享内存被划分为32个等宽的内存Bank（对应warp大小32）。每个Bank可以在每个时钟周期提供一个32位（4字节）的数据访问。

**地址映射规则**：
```
Bank编号 = (字节地址 / 4) % 32
```

**例**：
- 地址0 → Bank 0
- 地址4 → Bank 1
- 地址8 → Bank 2
- ...
- 地址124 → Bank 31
- 地址128 → Bank 0

### 1.2 什么是Bank冲突

当同一个warp内的多个线程在同一个时钟周期访问同一个Bank的不同地址时，就会发生Bank冲突。

**冲突类型**：
- **2路冲突**：2个线程访问同一个Bank
- **4路冲突**：4个线程访问同一个Bank
- **N路冲突**：N个线程访问同一个Bank

### 1.3 性能影响

| 冲突类型 | 性能下降 |
|---------|---------|
| 无冲突 | 基准 (1x) |
| 2路冲突 | ~2x |
| 4路冲突 | ~4x |
| 8路冲突 | ~8x |
| 32路冲突 | ~32x |

**特殊情况 - 广播机制**：
当warp内所有线程访问同一个地址时，硬件会执行广播操作，这不是冲突，效率很高！

---

## 2. 问题代码示例

### 2.1 步长访问导致冲突

```cpp
// ❌ 步长为2的访问：2路冲突
__global__ void conflict_stride2(float* data, int n) {
    __shared__ float shared[128];
    int tid = threadIdx.x;
    
    // 步长为2：tid和tid+16访问同一个Bank
    shared[tid * 2] = data[tid];  // Bank = (tid * 2) % 32
    __syncthreads();
}

// ❌ 步长为32的访问：32路冲突（最坏情况！）
__global__ void conflict_stride32(float* data, int n) {
    __shared__ float shared[1024];
    int tid = threadIdx.x;
    
    // 所有线程访问Bank tid，冲突极其严重
    shared[tid * 32] = data[tid];  // Bank = (tid * 32) % 32 = 0
    __syncthreads();
}
```

### 2.2 矩阵访问导致冲突

```cpp
// ❌ 共享内存矩阵转置中的Bank冲突
__global__ void transpose_with_conflict(float* input, float* output, int width, int height) {
    __shared__ float tile[32][32];
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;
    
    if (row < height && col < width) {
        tile[ty][tx] = input[row * width + col];  // 读取无冲突
    }
    __syncthreads();
    
    if (row < height && col < width) {
        // ❌ 写入时：tile[tx][ty] 导致Bank冲突
        // 同一个warp内tx相同，ty不同
        // Bank = (tx * 32 + ty) % 32 = ty % 32 → 32路冲突
        output[col * height + row] = tile[tx][ty];
    }
}
```

### 2.3 结构体布局导致冲突

```cpp
// ❌ 结构体数组可能导致Bank冲突
struct Vec2 {
    float x, y;
};

__global__ void struct_conflict(Vec2* data, int n) {
    __shared__ Vec2 shared[64];
    int tid = threadIdx.x;
    
    shared[tid].x = data[tid].x;  // Bank = (tid * 8 + 0) / 4 % 32 = (tid * 2) % 32
    shared[tid].y = data[tid].y;  // Bank = (tid * 8 + 4) / 4 % 32 = (tid * 2 + 1) % 32
    // 步长为2，2路冲突
    __syncthreads();
}
```

### 2.4 规约算法中的典型冲突

```cpp
// ❌ 朴素规约：严重的Bank冲突
__global__ void reduce_naive(float* data, float* result, int n) {
    __shared__ float shared[512];
    int tid = threadIdx.x;
    int idx = blockIdx.x * 512 + threadIdx.x;
    
    shared[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    // ❌ 从1开始加倍，步长访问导致严重冲突
    for (int s = 1; s < 512; s *= 2) {
        if (tid % (2 * s) == 0) {
            shared[tid] += shared[tid + s];  // Bank冲突！
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        result[blockIdx.x] = shared[0];
    }
}
```

---

## 3. 解决方案

### 3.1 方案一：填充（Padding）

**原理**：在每行末尾添加额外的元素，改变地址映射。

```cpp
// ✅ 使用填充避免Bank冲突
// 原来是 [32][32]，现在是 [32][33]，每行多一个元素
__global__ void transpose_no_conflict(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // +1 填充
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;
    
    if (row < height && col < width) {
        tile[ty][tx] = input[row * width + col];  // 读取无冲突
    }
    __syncthreads();
    
    int out_row = blockIdx.x * 32 + ty;
    int out_col = blockIdx.y * 32 + tx;
    
    if (out_row < width && out_col < height) {
        // ✅ Bank = (ty * 33 + tx) % 32 = (ty + tx) % 32
        // 同一个warp内tx连续，无冲突
        output[out_row * height + out_col] = tile[ty][tx];
    }
}
```

**填充的成本**：
- 额外的共享内存使用（32 * 4 = 128字节）
- 通常可以忽略不计

---

### 3.2 方案二：调整访问顺序

**原理**：改变循环顺序或索引计算方式，使访问连续。

```cpp
// ✅ 调整规约顺序，从大到小，消除Bank冲突
__global__ void reduce_no_conflict(float* data, float* result, int n) {
    __shared__ float shared[512];
    int tid = threadIdx.x;
    int idx = blockIdx.x * 512 + threadIdx.x;
    
    shared[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    // ✅ 从512/2开始减半，连续访问
    for (int s = 256; s > 0; s >>= 1) {
        if (tid < s) {
            // Bank = tid 和 Bank = tid + s
            // 连续访问，无冲突
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        result[blockIdx.x] = shared[0];
    }
}
```

---

### 3.3 方案三：使用向量类型

**原理**：使用float2、float4等向量类型，每个线程一次访问多个Bank。

```cpp
// ✅ 使用float4向量访问
__global__ void vectorized_shared(float* data, int n) {
    __shared__ float4 shared_vec[32];
    int tid = threadIdx.x;
    
    // 每个线程一次访问4个float，分布在4个Bank
    if (tid < 32) {
        float4 vec = *reinterpret_cast<float4*>(data + tid * 4);
        shared_vec[tid] = vec;
    }
    __syncthreads();
    
    // 处理数据
    if (tid < 32) {
        float4 val = shared_vec[tid];
        process(val.x, val.y, val.z, val.w);
    }
}
```

---

### 3.4 方案四：转置数据布局

**原理**：对于某些访问模式，预先转置数据可以消除冲突。

```cpp
// ✅ 数据转置避免访问冲突
// CPU端预处理
void pretranspose_for_gpu(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// GPU端使用转置后的数据
__global__ void use_transposed(float* transposed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float shared[128];
    
    // 现在是连续访问，无Bank冲突
    shared[tid] = transposed[idx];
    __syncthreads();
}
```

---

## 4. 代码对比

### 4.1 有冲突 vs 无冲突的矩阵转置

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

// ========== 有冲突版本 ==========
__global__ void transpose_conflict(float* input, float* output, int width, int height) {
    __shared__ float tile[32][32];
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;
    
    if (row < height && col < width) {
        tile[ty][tx] = input[row * width + col];
    }
    __syncthreads();
    
    if (row < height && col < width) {
        // ❌ Bank冲突：tile[tx][ty]
        output[col * height + row] = tile[tx][ty];
    }
}

// ========== 无冲突版本 ==========
__global__ void transpose_no_conflict(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // +1填充
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;
    
    if (row < height && col < width) {
        tile[ty][tx] = input[row * width + col];
    }
    __syncthreads();
    
    int out_row = blockIdx.x * 32 + ty;
    int out_col = blockIdx.y * 32 + tx;
    
    if (out_row < width && out_col < height) {
        // ✅ 无Bank冲突
        output[out_row * height + out_col] = tile[ty][tx];
    }
}

int main() {
    const int WIDTH = 2048;
    const int HEIGHT = 2048;
    
    float* h_input = new float[WIDTH * HEIGHT];
    float* h_output = new float[WIDTH * HEIGHT];
    
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, WIDTH * HEIGHT * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, WIDTH * HEIGHT * sizeof(float), H2D));
    
    dim3 block(32, 32);
    dim3 grid((WIDTH + 31) / 32, (HEIGHT + 31) / 32);
    
    // 测试有冲突版本
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        transpose_conflict<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double time_conflict = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    // 测试无冲突版本
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        transpose_no_conflict<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double time_no_conflict = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    std::cout << "有Bank冲突: " << time_conflict << " ms" << std::endl;
    std::cout << "无Bank冲突: " << time_no_conflict << " ms" << std::endl;
    std::cout << "加速比: " << time_conflict / time_no_conflict << "x" << std::endl;
    
    delete[] h_input;
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    
    return 0;
}
```

### 4.2 预期结果

| 版本 | 执行时间 | 说明 |
|------|---------|------|
| 有Bank冲突 | ~2.5 ms | 32路冲突，性能下降 |
| 无Bank冲突 | ~0.3 ms | 消除冲突 |
| **加速比** | **~8x** | |

---

## 5. 检测与分析方法

### 5.1 使用Nsight Compute分析

```bash
ncu --section MemoryWorkloadAnalysis -o bank_profile ./your_app
```

**关键指标**：
- `shared_load_transactions`：共享内存加载事务数
- `shared_store_transactions`：共享内存存储事务数
- `shared_efficiency`：共享内存效率（理想100%）
- `shared_load_transactions_per_request`：每个请求的平均事务数

**判断标准**：
- 理想情况：`shared_efficiency` ≈ 100%
- 有冲突情况：`shared_efficiency` < 50%

**查看Bank冲突详情**：
在Nsight Compute UI中查看：
- Source页面 → 选择对应代码行
- 查看"Shared Memory Transactions"列

---

### 5.2 手动计算检查

对于给定的访问模式，可以手动计算Bank编号：

```cpp
// 检查 pattern: shared[base + tid * stride]
void check_bank_conflict(int base, int stride) {
    int banks[32] = {0};
    for (int tid = 0; tid < 32; tid++) {
        int addr = base + tid * stride;
        int bank = (addr / 4) % 32;  // float是4字节
        banks[bank]++;
    }
    
    int max_conflict = 0;
    for (int i = 0; i < 32; i++) {
        max_conflict = max(max_conflict, banks[i]);
    }
    
    std::cout << "最大冲突路数: " << max_conflict << std::endl;
}
```

---

## 6. 最佳实践总结

### 6.1 共享内存设计原则

1. **优先使用连续访问**：`shared[tid]` 是最安全的模式
2. **使用填充消除冲突**：1个元素的填充通常就够了
3. **注意2D数组的访问模式**：行优先 vs 列优先
4. **合理使用向量类型**：float2/float4可以增加带宽

### 6.2 代码审查检查清单

- [ ] 共享内存访问是否是 `shared[tid]` 形式？
- [ ] 是否有 `shared[tid * stride]` 的步长访问？
- [ ] 2D共享内存访问是否交换了行列索引？
- [ ] 结构体数组是否可能导致步长访问？
- [ ] 规约算法是否从大到小迭代？

### 6.3 常见场景的优化策略

| 场景 | 优化策略 |
|------|---------|
| 矩阵转置 | 每行+1填充 |
| 规约求和 | 从大到小迭代 |
| 步长访问 | 添加填充或重新排列数据 |
| 结构体数组 | 使用SoA或调整字段顺序 |
| 1D直方图 | 使用多个中间bin减少冲突 |

---

### 6.4 特殊情况：Volta+架构的独立线程调度

Volta（计算能力7.0）及更高架构引入了独立线程调度，可以在某些情况下减少Bank冲突的影响，但不能完全消除。最佳实践仍然适用。

---

## 7. 进阶话题

### 7.1 不同数据类型的Bank映射

| 类型 | 大小 | Bank映射 | 冲突风险 |
|------|------|---------|---------|
| float/int | 4字节 | Bank = addr % 32 | 标准 |
| double/int64 | 8字节 | Bank = (addr/4) % 32 | 占用2个Bank |
| float2 | 8字节 | Bank = (addr/4) % 32 | 占用2个Bank |
| float4 | 16字节 | Bank = (addr/4) % 32 | 占用4个Bank |

**注意**：double和向量类型会占用多个连续的Bank，更容易导致冲突。

---

### 7.2 Bank冲突与L1缓存的交互

共享内存与L1缓存使用相同的硬件资源。在某些架构上，可以配置共享内存与L1缓存的比例：
- 48KB共享 + 16KB L1
- 32KB共享 + 32KB L1
- 16KB共享 + 48KB L1

根据应用的共享内存使用量选择合适的配置。

---

### 7.3 原子操作与Bank冲突

共享内存的原子操作也会受到Bank冲突的影响。如果多个线程对同一个Bank执行原子操作，性能会显著下降。

**优化**：使用per-thread中间数组，最后再归约。

```cpp
__global__ void histogram_atomic(int* data, int* hist, int n, int bins) {
    __shared__ int shared_hist[128];
    int tid = threadIdx.x;
    
    // 初始化
    if (tid < 128) shared_hist[tid] = 0;
    __syncthreads();
    
    // 每个线程处理多个元素
    for (int i = tid; i < n; i += blockDim.x) {
        int bin = compute_bin(data[i]);
        atomicAdd(&shared_hist[bin], 1);  // 可能有冲突
    }
    __syncthreads();
    
    // 写回全局
    if (tid < 128) {
        atomicAdd(&hist[tid], shared_hist[tid]);
    }
}
```

---

## 8. 相关资源

- CUDA C++ Programming Guide: Shared Memory
- NVIDIA Developer Blog: Using Shared Memory in CUDA
- CUDA Best Practices Guide: Shared Memory Optimizations
- GTC Talks: Advanced Shared Memory Programming
