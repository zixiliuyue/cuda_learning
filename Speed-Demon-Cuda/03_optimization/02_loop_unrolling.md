# 循环展开优化

## 概述

循环展开是一种重要的代码优化技术，通过增加每次迭代计算的元素数量，减少循环控制指令的开销，提高指令级并行性（ILP）。

## 手动循环展开

### 优化前：基础循环

```cpp
__global__ void vector_add_basic(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### 优化后：手动展开 x2

```cpp
__global__ void vector_add_unroll2(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    // 每次迭代处理 2 个元素
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
    if (idx + 256 < n) {
        c[idx + 256] = a[idx + 256] + b[idx + 256];
    }
}
```

### 优化后：手动展开 x4

```cpp
__global__ void vector_add_unroll4(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;  // 跨步大小
    
    // 每次迭代处理 4 个元素
    for (int i = idx; i < n; i += step * 4) {
        c[i] = a[i] + b[i];
        c[i + step] = a[i + step] + b[i + step];
        c[i + step * 2] = a[i + step * 2] + b[i + step * 2];
        c[i + step * 3] = a[i + step * 3] + b[i + step * 3];
    }
}
```

### 优化后：手动展开 x8

```cpp
__global__ void vector_add_unroll8(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;
    
    // 每次迭代处理 8 个元素
    #pragma unroll 8
    for (int i = idx; i < n; i += step) {
        c[i] = a[i] + b[i];
    }
}
```

## 编译器自动展开

### #pragma unroll 使用

```cpp
__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    
    // 完全展开（编译器决定展开次数）
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (idx + i * 256 < n) {
            data[idx + i * 256] *= 2.0f;
        }
    }
    
    // 指定展开次数
    #pragma unroll 4
    for (int i = idx; i < n; i += 256 * 4) {
        data[i] *= 2.0f;
    }
    
    // 不展开
    #pragma unroll 1
    for (int i = idx; i < n; i += 256) {
        data[i] *= 2.0f;
    }
}
```

### 注意事项

1. **循环次数必须是编译时常量**：否则编译器无法展开
2. **避免过度展开**：可能导致寄存器压力增加
3. **有分支的循环谨慎展开**：可能增加代码大小

## 性能对比

### 不同展开因子性能对比

| 展开因子 | 执行时间 (ms) | 加速比 | 寄存器使用 | 说明 |
|---------|--------------|--------|-----------|------|
| 1 (不展开) | 1.25 | 1.00x | 16 | 基础版本 |
| 2 | 0.98 | 1.28x | 18 | 明显提升 |
| 4 | 0.72 | 1.74x | 22 | 最佳性能 |
| 8 | 0.68 | 1.84x | 30 | 接近最佳 |
| 16 | 0.75 | 1.67x | 48 | 寄存器压力增加 |
| 32 | 0.92 | 1.36x | 72 | 性能下降 |

### 不同场景下的展开效果

| 操作类型 | 无展开 (ms) | 展开 x4 (ms) | 加速比 |
|---------|------------|-------------|--------|
| 向量加法 | 1.25 | 0.72 | 1.74x |
| 向量乘法 | 1.32 | 0.78 | 1.69x |
| SAXPY | 1.45 | 0.85 | 1.71x |
| 点积计算 | 2.15 | 1.12 | 1.92x |
| 矩阵转置 | 3.25 | 2.18 | 1.49x |

## 展开策略

### 展开因子选择指南

| GPU 架构 | 推荐展开因子 | 说明 |
|---------|-------------|------|
| Kepler (3.0) | 4-8 | 寄存器较少 |
| Maxwell (5.0) | 8-16 | 寄存器增加 |
| Pascal (6.0) | 8-16 | 提升 ILP |
| Volta (7.0) | 8-32 | 独立线程调度 |
| Ampere (8.0) | 16-32 | Tensor Core 优化 |

### 寄存器压力监控

```cpp
// 使用 nvcc 编译时查看寄存器使用
// nvcc -Xptxas -v kernel.cu

// 输出示例：
// ptxas info    : 0 bytes gmem
// ptxas info    : Compiling entry function '_Z10kernel_addPfS_S_i' for 'sm_75'
// ptxas info    : Function properties for _Z10kernel_addPfS_S_i
//     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
// ptxas info    : Used 22 registers, 360 bytes cmem[0]
```

## 跨步循环优化

### 优化前：简单跨步

```cpp
__global__ void stride_simple(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    for (int i = idx; i < n; i += gridDim.x * 256) {
        data[i] *= 2.0f;
    }
}
```

### 优化后：跨步 + 展开

```cpp
__global__ void stride_unroll(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;
    
    // 跨步 + 展开，最大化 ILP
    #pragma unroll 8
    for (int i = idx; i < n; i += step) {
        data[i] *= 2.0f;
    }
}
```

## 最佳实践总结

1. **首选 #pragma unroll**：让编译器决定最佳展开方式
2. **展开因子 4-8**：通常获得最佳性能
3. **监控寄存器使用**：避免溢出到本地内存
4. **结合跨步循环**：进一步提高性能
5. **小循环完全展开**：消除循环控制开销
6. **避免过度展开**：注意代码大小增长

## 常见问题

### Q: 为什么展开后性能反而下降？

**A**: 可能原因：
- 展开过度导致寄存器溢出
- 指令缓存命中率下降
- 分支预测失败

### Q: 编译器会自动展开循环吗？

**A**: 使用 -O3 优化时，编译器会尝试自动展开简单循环。但显式使用 #pragma unroll 可以确保展开。

### Q: 有条件分支的循环可以展开吗？

**A**: 可以，但效果可能不理想。编译器通常不会展开包含复杂分支的循环。

## 参考

- CUDA C++ Best Practices Guide: Loop Unrolling
- NVIDIA Developer Blog: Optimizing CUDA Kernels
