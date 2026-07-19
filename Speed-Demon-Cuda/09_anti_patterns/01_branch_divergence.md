# 反模式 01: 分支发散 (Branch Divergence)

---

## 1. 问题描述

### 1.1 什么是分支发散

分支发散是指同一个warp内的不同线程执行不同的代码路径。由于CUDA的SIMT（单指令多线程）执行模型，warp内的所有线程必须步调一致地执行同一条指令。当发生分支发散时，硬件会串行执行所有可能的路径，对不需要执行该路径的线程进行掩码（mask off）。

### 1.2 性能影响

| 场景 | 性能下降 |
|------|---------|
| warp内一半线程走不同路径 | ~50% |
| warp内每个线程走不同路径 | ~97% (32倍) |
| 循环内条件分支 | 取决于分支频率 |

最坏情况下，一个原本可以全速运行的warp，因为严重的分支发散，性能会下降到原来的1/32。

---

## 2. 问题代码示例

### 2.1 条件判断导致的发散

```cpp
// ❌ 有分支发散的代码
__global__ void bad_branch_example(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 边界检查：通常只有最后一个warp有发散，影响较小
        float val = data[idx];
        
        // 严重发散：warp内线程可能走不同路径
        if (val > 0.0f) {
            val = sqrt(val);      // 路径A
        } else {
            val = log(-val);      // 路径B
        }
        
        data[idx] = val;
    }
}
```

### 2.2 循环迭代次数不同导致的发散

```cpp
// ❌ 循环次数不一致导致发散
__global__ void different_loop_iterations(float* data, int* sizes, int n) {
    int idx = blockIdx.x;
    
    // 每个block的循环次数可能完全不同
    for (int i = 0; i < sizes[idx]; i++) {
        process(data[idx * 1024 + i]);
    }
}
```

---

## 3. 解决方案

### 3.1 方案一：重新组织线程/数据布局

**原理**：将具有相同行为的线程组织在同一个warp内。

```cpp
// ✅ 优化：按条件分组数据
// CPU端预先重排数据，将相同类别的数据放在连续区域
__global__ void grouped_branch(float* data, int positive_count, int total_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 前N个线程处理正数
    if (idx < positive_count) {
        float val = data[idx];
        data[idx] = sqrt(val);
    }
    // 剩余线程处理负数
    else if (idx < total_count) {
        float val = data[idx];
        data[idx] = log(-val);
    }
}
```

### 3.2 方案二：使用内建函数消除分支

**原理**：利用CUDA的内建指令或谓词执行，将条件分支转化为数据选择。

```cpp
// ✅ 使用内建函数消除分支
__device__ float process_value(float val) {
    float sqrt_val = sqrt(max(val, 1e-10f));
    float log_val = log(max(-val, 1e-10f));
    
    // 两个路径都计算，最后选择结果
    // 编译器会使用谓词指令，没有分支发散
    return val > 0.0f ? sqrt_val : log_val;
}

__global__ void branch_free_example(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = process_value(data[idx]);
    }
}
```

### 3.3 方案三：使用warp内建函数

**原理**：对于规约、扫描等操作，使用warp内建函数避免分支。

```cpp
// ✅ warp内规约，无分支发散
__device__ float warp_reduce_sum(float val) {
    unsigned int mask = 0xffffffff;
    
    // 使用shfl指令，无明显分支
    val += __shfl_xor_sync(mask, val, 16);
    val += __shfl_xor_sync(mask, val, 8);
    val += __shfl_xor_sync(mask, val, 4);
    val += __shfl_xor_sync(mask, val, 2);
    val += __shfl_xor_sync(mask, val, 1);
    
    return val;
}
```

### 3.4 方案四：模板元编程

**原理**：使用模板参数在编译期确定代码路径。

```cpp
// ✅ 编译期分支选择
template<bool IS_POSITIVE>
__device__ float process_template(float val) {
    if (IS_POSITIVE) {
        return sqrt(val);  // 编译时确定，无运行时分支
    } else {
        return log(-val);
    }
}

// 两个内核分别处理不同类别的数据
__global__ void process_positive(float* data, int n) {
    int idx = ...;
    if (idx < n) data[idx] = process_template<true>(data[idx]);
}

__global__ void process_negative(float* data, int n) {
    int idx = ...;
    if (idx < n) data[idx] = process_template<false>(data[idx]);
}
```

---

## 4. 代码对比

### 4.1 对比测试

```cpp
#include <iostream>
#include <chrono>
#include <random>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// ❌ 有分支发散的版本
__global__ void kernel_with_branch(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        if (val > 0.0f) {
            data[idx] = sqrt(val);
        } else {
            data[idx] = log(-val + 1e-10f);
        }
    }
}

// ✅ 无分支的版本
__global__ void kernel_without_branch(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        float sqrt_val = sqrt(max(val, 1e-10f));
        float log_val = log(max(-val, 1e-10f));
        data[idx] = val > 0.0f ? sqrt_val : log_val;
    }
}

int main() {
    const int N = 1 << 20;  // 100万元素
    float *h_data, *d_data;
    
    h_data = new float[N];
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        h_data[i] = dist(gen);  // 正负各半，最坏情况
    }
    
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    
    // 测试有分支的版本
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        kernel_with_branch<<<grid, block>>>(d_data, N);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double time_with_branch = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    // 测试无分支的版本
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        kernel_without_branch<<<grid, block>>>(d_data, N);
    }
    
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double time_without_branch = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    std::cout << "有分支发散: " << time_with_branch << " ms" << std::endl;
    std::cout << "无分支发散: " << time_without_branch << " ms" << std::endl;
    std::cout << "加速比: " << time_with_branch / time_without_branch << "x" << std::endl;
    
    delete[] h_data;
    CHECK_CUDA(cudaFree(d_data));
    
    return 0;
}
```

### 4.2 预期结果

| 版本 | 执行时间 | 说明 |
|------|---------|------|
| 有分支发散 | ~2.0 ms | 50%的分支发散概率 |
| 无分支发散 | ~1.2 ms | 统一执行路径 |
| **加速比** | **~1.7x** | |

---

## 5. 检测与分析方法

### 5.1 使用Nsight Compute分析

```bash
ncu --set full -o branch_analysis ./branch_test
```

**关键指标查看**：
- `Branch Efficiency`：分支效率（理想100%）
- `Divergent Branch`：发散分支数量
- `Threads Active`：活跃线程比例

### 5.2 编译时检测

```bash
nvcc --ptxas-options=-v branch_test.cu
```

查看生成的PTX代码中是否有分支指令。

---

## 6. 最佳实践总结

### 6.1 什么时候不用担心分支发散

1. **边界检查**：`if (idx < n)` 通常只有最后一个warp发散
2. **warp内所有线程走相同路径**
3. **小内核中分支开销可以忽略**

### 6.2 优化优先级顺序

1. **首先**：重新组织数据/线程，让warp内走相同路径
2. **其次**：使用内建函数/谓词执行消除分支
3. **再次**：两个路径都计算然后选择
4. **最后**：使用模板/动态并行分离工作负载

### 6.3 经验法则

- **分支效率 > 90%**：通常不需要优化
- **分支效率 70-90%**：可以考虑优化
- **分支效率 < 70%**：强烈建议优化

---

## 7. 相关资源

- CUDA C++ Programming Guide: SIMT Architecture
- NVIDIA Developer Blog: Branch Predication
- CUDA Best Practices Guide: Branch Divergence
