# 反模式 02: 非合并内存访问 (Uncoalesced Memory Access)

---

## 1. 问题描述

### 1.1 什么是合并访问

合并访问是指warp内的线程在访问全局内存时，访问的地址满足一定的对齐和连续性要求，使得硬件可以将多个内存请求合并为单个或少数几个内存事务。

**历史背景**：
- 计算能力 1.x：非常严格的对齐要求（必须对齐到128字节边界）
- 计算能力 2.0+：更加灵活，支持缓存机制
- 计算能力 7.0+：L1缓存重新启用，合并更智能

### 1.2 性能影响

| 访问模式 | 带宽利用率 | 性能下降 |
|---------|-----------|---------|
| 完全合并 | 80-95% | 基准 |
| 步长为2 | 40-50% | ~2x |
| 步长为32 | 3-5% | ~20-30x |
| 随机访问 | 1-10% | ~10-100x |

非合并访问是CUDA程序中最常见、也是对性能影响最大的瓶颈之一。

---

## 2. 问题代码示例

### 2.1 步长访问 (Strided Access)

```cpp
// ❌ 步长访问：非合并
__global__ void strided_access(float* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程间隔stride个元素访问
    // 当stride > 1时，warp内地址不连续
    if (idx * stride < n) {
        float val = data[idx * stride];  // 严重非合并
        process(val);
    }
}

// 典型场景：AoS (Array of Structs)
struct PointAoS {
    float x, y, z;
};

__global__ void process_points(PointAoS* points, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 每个线程访问的x字段间隔12字节（3个float）
        float x = points[idx].x;  // 步长为3，非合并
        process(x);
    }
}
```

### 2.2 索引重排导致非合并

```cpp
// ❌ 间接索引访问
__global__ void indexed_access(float* data, int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // indices数组中的值可能是任意的，导致完全随机访问
        float val = data[indices[idx]];
        process(val);
    }
}

// ❌ 转置操作中的非合并访问
__global__ void transpose_bad(float* input, float* output, int width, int height) {
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    
    if (row < height && col < width) {
        // 写输出时，同一个warp的列连续但行不连续
        // 导致跨步访问
        output[col * height + row] = input[row * width + col];
    }
}
```

### 2.3 小数据类型的非对齐访问

```cpp
// ❌ 未对齐的结构体访问
struct __align__(8) HalfAligned {
    short a, b;
    float c;
};  // 总大小8字节，但访问c时可能非对齐

__global__ void unaligned_access(HalfAligned* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float c = data[idx].c;  // 地址可能没有4字节对齐
        process(c);
    }
}
```

---

## 3. 解决方案

### 3.1 方案一：数据结构转换 (AoS → SoA)

**原理**：将结构体数组转换为数组的结构体，使相同字段连续存储。

```cpp
// ✅ SoA (Struct of Arrays) 布局
struct PointsSoA {
    float* x;
    float* y;
    float* z;
};

__global__ void process_points_soa(PointsSoA points, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 每个字段都是连续存储，完全合并访问
        float x = points.x[idx];  // 合并！
        float y = points.y[idx];  // 合并！
        float z = points.z[idx];  // 合并！
        process(x, y, z);
    }
}
```

**性能对比**：
| 布局 | 每个字段的内存事务数 | 带宽利用率 |
|------|---------------------|-----------|
| AoS | 32个事务/字段 | ~12% |
| SoA | 1-2个事务/字段 | ~85% |

**加速比**：~5-7倍

---

### 3.2 方案二：使用共享内存中转

**原理**：对于矩阵转置等需要重新排列数据的操作，使用共享内存作为中转，实现全局内存的合并访问。

```cpp
// ✅ 使用共享内存的高效转置
__global__ void transpose_good(float* input, float* output, int width, int height) {
    __shared__ float tile[32][32];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;
    
    // 第一步：合并读取到共享内存
    if (row < height && col < width) {
        tile[ty][tx] = input[row * width + col];  // 合并读取
    }
    __syncthreads();
    
    // 第二步：从共享内存写入全局内存
    // 注意：交换block索引，避免bank冲突
    int out_row = blockIdx.x * 32 + ty;
    int out_col = blockIdx.y * 32 + tx;
    
    if (out_row < width && out_col < height) {
        output[out_row * height + out_col] = tile[tx][ty];  // 合并写入
    }
}
```

**关键点**：
- 读输入和写输出都使用合并访问
- 共享内存块大小32x32避免bank冲突
- 写入时转置索引

---

### 3.3 方案三：批量收集+处理

**原理**：对于间接索引访问，先批量收集索引对应的连续数据块，再处理。

```cpp
// ✅ 间接索引的优化版本
__global__ void optimized_indexed_access(float* data, int* indices, int n) {
    __shared__ float shared_data[128];
    int tid = threadIdx.x;
    int block = blockIdx.x;
    
    // 第一步：当前block要处理的索引范围
    int start = block * 128;
    int end = min(start + 128, n);
    
    // 第二步：每个线程负责一个索引
    if (tid < end - start) {
        int idx = indices[start + tid];
        shared_data[tid] = data[idx];  // 虽然可能非合并，但总事务数减少
    }
    __syncthreads();
    
    // 第三步：使用共享内存处理数据
    if (tid < end - start) {
        float val = shared_data[tid];
        process(val);
    }
}

// ✅ 更好的方案：CPU端重排数据为连续
// CPU预处理
void reorder_for_gpu(float* data, int* indices, int n) {
    float* reordered = new float[n];
    for (int i = 0; i < n; i++) {
        reordered[i] = data[indices[i]];  // CPU预先重排为连续
    }
    // 传输重排后的数据到GPU
    cudaMemcpy(d_data, reordered, n * sizeof(float), H2D);
}
```

---

### 3.4 方案四：使用向量加载指令

**原理**：使用float2、float4等向量类型加载，增加每个内存事务的有效数据量。

```cpp
// ✅ 使用向量加载提高带宽利用率
__global__ void vectorized_load(float* data, int n) {
    int idx = blockIdx.x * 64 + threadIdx.x;
    
    // 每个线程加载4个连续的float
    if (idx * 4 < n) {
        float4 vec = *reinterpret_cast<float4*>(data + idx * 4);
        
        // 处理向量中的每个元素
        process(vec.x);
        process(vec.y);
        process(vec.z);
        process(vec.w);
    }
}

// 注意：确保指针对齐
// 推荐使用cudaMalloc分配的内存（天然对齐到256字节）
```

**向量加载的好处**：
- 减少指令数量
- 提高内存总线利用率
- 编译器更容易优化

---

## 4. 代码对比

### 4.1 AoS vs SoA 性能对比

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

// ========== AoS版本 ==========
struct PointAoS {
    float x, y, z;
};

__global__ void kernel_aos(PointAoS* points, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float x = points[idx].x;
        float y = points[idx].y;
        float z = points[idx].z;
        points[idx].x = x + y + z;
    }
}

// ========== SoA版本 ==========
struct PointsSoA {
    float* x;
    float* y;
    float* z;
};

__global__ void kernel_soa(PointsSoA points, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float x = points.x[idx];
        float y = points.y[idx];
        float z = points.z[idx];
        points.x[idx] = x + y + z;
    }
}

int main() {
    const int N = 1 << 20;  // 100万点
    
    PointAoS* h_aos = new PointAoS[N];
    for (int i = 0; i < N; i++) {
        h_aos[i].x = static_cast<float>(i);
        h_aos[i].y = static_cast<float>(i + 1);
        h_aos[i].z = static_cast<float>(i + 2);
    }
    
    PointsSoA h_soa;
    h_soa.x = new float[N];
    h_soa.y = new float[N];
    h_soa.z = new float[N];
    for (int i = 0; i < N; i++) {
        h_soa.x[i] = static_cast<float>(i);
        h_soa.y[i] = static_cast<float>(i + 1);
        h_soa.z[i] = static_cast<float>(i + 2);
    }
    
    PointAoS* d_aos;
    PointsSoA d_soa;
    
    CHECK_CUDA(cudaMalloc(&d_aos, N * sizeof(PointAoS)));
    CHECK_CUDA(cudaMalloc(&d_soa.x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_soa.y, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_soa.z, N * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_aos, h_aos, N * sizeof(PointAoS), H2D));
    CHECK_CUDA(cudaMemcpy(d_soa.x, h_soa.x, N * sizeof(float), H2D));
    CHECK_CUDA(cudaMemcpy(d_soa.y, h_soa.y, N * sizeof(float), H2D));
    CHECK_CUDA(cudaMemcpy(d_soa.z, h_soa.z, N * sizeof(float), H2D));
    
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    
    // 测试AoS
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        kernel_aos<<<grid, block>>>(d_aos, N);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double time_aos = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    // 测试SoA
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        kernel_soa<<<grid, block>>>(d_soa, N);
    }
    
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double time_soa = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    std::cout << "AoS (非合并): " << time_aos << " ms" << std::endl;
    std::cout << "SoA (合并):   " << time_soa << " ms" << std::endl;
    std::cout << "加速比: " << time_aos / time_soa << "x" << std::endl;
    
    delete[] h_aos;
    delete[] h_soa.x;
    delete[] h_soa.y;
    delete[] h_soa.z;
    
    CHECK_CUDA(cudaFree(d_aos));
    CHECK_CUDA(cudaFree(d_soa.x));
    CHECK_CUDA(cudaFree(d_soa.y));
    CHECK_CUDA(cudaFree(d_soa.z));
    
    return 0;
}
```

### 4.2 预期结果

| 布局 | 执行时间 | 带宽利用率 |
|------|---------|-----------|
| AoS (非合并) | ~0.8 ms | ~15% |
| SoA (合并) | ~0.15 ms | ~85% |
| **加速比** | **~5.3x** | |

---

## 5. 检测与分析方法

### 5.1 使用Nsight Compute分析

```bash
ncu --section MemoryWorkloadAnalysis -o memory_profile ./your_app
```

**关键指标**：
- `gld_transactions`：全局加载事务数
- `gst_transactions`：全局存储事务数
- `gld_transactions_per_request`：每个请求的平均事务数
- `dram_read_throughput`：DRAM读取吞吐量
- `dram_write_throughput`：DRAM写入吞吐量

**判断标准**：
- 理想情况：`gld_transactions_per_request` ≈ 1-2
- 非合并情况：`gld_transactions_per_request` > 4

### 5.2 理论计算验证

计算理想情况下的内存事务数：
```
理想事务数 = 总字节数 / 128字节（每个事务大小）
实际事务数 = gld_transactions
合并效率 = 理想事务数 / 实际事务数
```

---

## 6. 最佳实践总结

### 6.1 数据布局设计原则

1. **优先使用SoA布局**：特别是GPU端数据
2. **避免结构体嵌套**：增加内存访问复杂度
3. **合理使用向量类型**：float2/float4可以提高加载效率
4. **注意对齐**：确保数据对齐到4/8/16字节边界

### 6.2 代码审查检查清单

- [ ] 数组索引是否是 `idx = block * blockDim + thread` 的形式？
- [ ] 是否使用了AoS而不是SoA？
- [ ] 是否有 `data[idx * stride]` 这样的步长访问？
- [ ] 是否有间接索引访问 `data[indices[idx]]`？
- [ ] 矩阵访问是否使用了 `row * width + col` 形式？

### 6.3 常见场景的优化策略

| 场景 | 优化策略 |
|------|---------|
| 结构体访问 | AoS → SoA |
| 矩阵转置 | 使用共享内存中转 |
| 间接索引 | CPU预重排 + 批量收集 |
| 步长访问 | 转置数据或使用向量化 |
| 小数据类型 | 打包为向量类型 |

---

## 7. 进阶话题

### 7.1 L1/L2缓存的影响

计算能力7.0+的GPU有L1缓存，可以缓解非合并访问的影响，但不能完全消除。合并访问仍然是最佳实践。

### 7.2 统一内存的注意事项

统一内存的页迁移机制对于非连续访问模式的效率更低，更需要注意合并访问。

---

## 8. 相关资源

- CUDA C++ Programming Guide: Memory Hierarchy
- NVIDIA Developer Blog: Memory Optimization
- CUDA Best Practices Guide: Coalesced Access
- GTC Talks: CUDA Memory Optimization
