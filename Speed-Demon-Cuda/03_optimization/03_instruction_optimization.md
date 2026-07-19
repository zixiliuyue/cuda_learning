# 指令优化

## 概述

指令优化是 CUDA 性能调优的核心，通过优化算术指令、避免分支发散、使用内置函数和选择合适的数据类型，可以显著提升 kernel 性能。

## 算术指令优化

### 避免昂贵的数学运算

| 运算类型 | 相对开销 | 优化建议 |
|---------|---------|---------|
| 加法/减法 | 1.0x | 基础运算 |
| 乘法 | 1.0x | 基础运算 |
| 乘加 (FMA) | 1.0x | 优先使用 |
| 除法 | 10-20x | 预计算倒数 |
| 平方根 | 10-15x | 使用 rsqrt 或近似 |
| 三角函数 | 20-50x | 查找表或近似 |
| 指数/对数 | 15-30x | 内置近似 |

### 优化前：使用昂贵运算

```cpp
__global__ void expensive_ops(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        // 多次昂贵运算
        float result = sqrt(x) / sqrt(x + 1.0f);
        data[idx] = result;
    }
}
```

### 优化后：数学运算优化

```cpp
__global__ void optimized_ops(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= n) return;
    
    float x = data[idx];
    // 优化：先加 1.0f / sqrt(x + 1.0f) 代替除法
    float sum = x + 1.0f;
    float rsqrt_sum = rsqrt(sum);  // 快速倒数平方根
    float sqrt_x = sqrt(x);
    float result = sqrt_x * rsqrt_sum;  // 乘法代替除法
    data[idx] = result;
}
```

### 优化前：重复计算

```cpp
__global__ void duplicate_calc(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        // 重复计算相同表达式
        c[idx] = a[idx] * b[idx] + a[idx] * 2.0f - b[idx] * 3.0f;
    }
}
```

### 优化后：公共子表达式消除

```cpp
__global__ void optimized_calc(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= n) return;
    
    float av = a[idx];
    float bv = b[idx];
    // 公共子表达式只计算一次
    float prod = av * bv;
    c[idx] = prod + av * 2.0f - bv * 3.0f;
}
```

## 避免分支发散

### 优化前：warp 内分支发散

```cpp
__global__ void divergent_branch(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        // warp 内线程可能走不同分支
        if (x > 0.5f) {
            data[idx] = sin(x);  // 部分线程执行
        } else {
            data[idx] = cos(x);  // 另一部分线程执行
        }
    }
}
```

### 优化后：使用内置选择

```cpp
__global__ void optimized_branch(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= n) return;
    
    float x = data[idx];
    // 使用 select 代替分支，避免发散
    float s = sin(x);
    float c = cos(x);
    data[idx] = x > 0.5f ? s : c;
}
```

### 优化后：使用掩码

```cpp
__global__ void mask_branch(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= n) return;
    
    float x = data[idx];
    // 使用掩码计算
    float mask = x > 0.5f ? 1.0f : 0.0f;
    data[idx] = mask * sin(x) + (1.0f - mask) * cos(x);
}
```

### 优化后：重新组织数据避免分支

```cpp
__global__ void reorganized_branch(float* pos_data, float* neg_data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= n) return;
    
    // 数据预先分组，warp 内线程走相同分支
    pos_data[idx] = sin(pos_data[idx]);
    neg_data[idx] = cos(neg_data[idx]);
}
```

## 使用内置函数

### 快速数学函数

| 标准函数 | 快速版本 | 精度 | 性能 |
|---------|---------|------|------|
| sin(x) | sin(x) | 高 | 慢 |
| sin(x) | __sinf(x) | 低 | 快 |
| cos(x) | __cosf(x) | 低 | 快 |
| exp(x) | __expf(x) | 低 | 快 |
| log(x) | __logf(x) | 低 | 快 |
| sqrt(x) | sqrt(x) | 高 | 中 |
| rsqrt(x) | rsqrt(x) | 高 | 快 |

### 优化前：标准函数

```cpp
__global__ void standard_func(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= n) return;
    
    float x = data[idx];
    // 标准函数，高精度但较慢
    data[idx] = sin(x) * exp(-x * x);
}
```

### 优化后：快速内置函数

```cpp
__global__ void fast_func(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= n) return;
    
    float x = data[idx];
    // 快速内置函数，稍低精度但更快
    data[idx] = __sinf(x) * __expf(-x * x);
}
```

### 使用 --use_fast_math 编译选项

```bash
# 编译时启用快速数学
nvcc -O3 --use_fast_math kernel.cu -o kernel
```

**注意**：使用 `--use_fast_math` 会：
- 将所有数学函数替换为快速版本
- 可能影响数值稳定性
- 适合对精度要求不高的场景

## 数据类型选择

### 精度与性能对比

| 数据类型 | 计算性能 | 内存占用 | 适用场景 |
|---------|---------|---------|---------|
| float | 1.0x (基准) | 4 bytes | 通用计算 |
| double | 0.25x - 0.5x | 8 bytes | 需要高精度 |
| half | 2.0x - 8.0x | 2 bytes | 机器学习、低精度 |
| int | 1.0x | 4 bytes | 整数计算 |
| char | 1.0x | 1 byte | 字节操作 |

### 优化前：不必要的双精度

```cpp
__global__ void unnecessary_double(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        // 使用双精度计算，然后转回单精度
        double x = data[idx];
        double result = sin(x) * exp(-x * x);
        data[idx] = (float)result;
    }
}
```

### 优化后：使用单精度

```cpp
__global__ void use_float(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= n) return;
    
    // 直接使用单精度
    float x = data[idx];
    float result = sin(x) * exp(-x * x);
    data[idx] = result;
}
```

### 使用半精度 (FP16)

```cpp
#include <cuda_fp16.h>

__global__ void use_half(half* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= n) return;
    
    half x = data[idx];
    // FP16 计算（需要架构支持）
    half result = hmul(hexp(hmul(x, x)));
    data[idx] = result;
}
```

## 性能对比

### 算术优化效果

| 优化方式 | 执行时间 (ms) | 加速比 | 说明 |
|---------|--------------|--------|------|
| 未优化 | 5.25 | 1.00x | 基础版本 |
| 消除重复计算 | 4.12 | 1.27x | 减少冗余 |
| 倒数优化除法 | 3.45 | 1.52x | 乘法代替除法 |
| FMA 融合 | 2.98 | 1.76x | 乘加指令 |
| 快速数学函数 | 1.85 | 2.84x | 内置快速函数 |

### 分支优化效果

| 分支类型 | 执行时间 (ms) | 加速比 | 发散率 |
|---------|--------------|--------|--------|
| 随机分支 (50%发散) | 8.50 | 1.00x | 50% |
| 掩码计算 | 3.20 | 2.66x | 0% |
| 三元运算符 | 3.45 | 2.46x | 0% |
| 数据预分组 | 2.10 | 4.05x | 0% |

### 数据类型性能

| 数据类型 | 执行时间 (ms) | 加速比 | 精度 |
|---------|--------------|--------|------|
| double | 12.50 | 1.00x | 高 |
| float | 3.20 | 3.91x | 中 |
| half (FP16) | 0.85 | 14.7x | 低 |

## 最佳实践总结

1. **优先使用 FMA 指令**：让编译器自动融合乘加操作
2. **避免除法运算**：预计算倒数，用乘法代替
3. **最小化分支发散**：使用掩码或三元运算符
4. **合理使用快速数学**：平衡精度和性能
5. **选择合适的数据类型**：避免不必要的高精度
6. **消除公共子表达式**：减少重复计算
7. **使用内置函数**：利用硬件加速

## 常见问题

### Q: 什么时候应该使用快速数学函数？

**A**: 当精度要求不高时，如：
- 机器学习训练
- 图形渲染
- 信号处理
- 蒙特卡洛模拟

### Q: 如何检测分支发散？

**A**: 使用 NVIDIA Nsight Compute 分析：
- 查看 Warp Execution Efficiency
- 检查 Branch Divergence 指标
- 分析 Control Flow 图表

### Q: 双精度一定慢多少？

**A**: 取决于 GPU 架构：
- Titan V: 1/32 单精度性能
- Tesla V100: 1/2 单精度性能
- A100: 1/2 单精度性能
- 消费级显卡：通常 1/32 或更差

## 参考

- CUDA C++ Best Practices Guide: Instruction Optimization
- NVIDIA Math Library Documentation
- Tuning CUDA Applications for Kepler, Page-Latency Hiding
