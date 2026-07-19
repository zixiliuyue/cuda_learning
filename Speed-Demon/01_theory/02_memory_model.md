# CUDA 内存模型 🧠

## 1. 内存类型详解 📋

CUDA 提供多种内存类型，各有不同的特性、作用域和使用场景。

### 1.1 全局内存 (Global Memory) 🌍

**位置**: GPU 板载显存 (DRAM)

#### 特性

| 属性 | 值 |
|------|-----|
| 容量 | GB 级 (4GB - 80GB+) |
| 速度 | ~200-800 cycles |
| 带宽 | ~1TB/s (A100) |
| 作用域 | 所有线程 + Host |
| 生命周期 | 应用程序级 |
| 关键字 | `__global__` / `cudaMalloc()` |

#### 分配方式

```cpp
// Host 端分配
float* dev_ptr;
cudaMalloc(&dev_ptr, size);          // 分配
cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice);  // 拷贝
cudaFree(dev_ptr);                   // 释放

// 内核中声明
__global__ void kernel() {
    // 动态全局内存 (需要编译时指定 -rdc=true)
    __device__ float global_var;
}
```

---

### 1.2 共享内存 (Shared Memory) 🤝

**位置**: SM 片上内存

#### 特性

| 属性 | 值 |
|------|-----|
| 容量 | 16KB - 164KB per SM |
| 速度 | ~1-32 cycles |
| 带宽 | ~10-20 TB/s |
| 作用域 | Block 内线程共享 |
| 生命周期 | 内核执行期间 |
| 关键字 | `__shared__` |

#### 两种声明方式

```cpp
__global__ void kernel() {
    // 1. 静态共享内存 - 编译时确定大小
    __shared__ float static_share[256];
    
    // 2. 动态共享内存 - 启动时指定大小
    extern __shared__ float dynamic_share[];
    // 启动时: kernel<<<grid, block, shared_size>>>(...);
}
```

#### Bank 冲突 ⚠️

共享内存被分为 32 个 Bank，每个 Bank 宽度为 4 字节。

```
Bank 0  Bank 1  Bank 2  ...  Bank 31
  │       │       │            │
  ▼       ▼       ▼            ▼
┌─────┐ ┌─────┐ ┌─────┐     ┌─────┐
│addr0│ │addr1│ │addr2│     │addr31│
├─────┤ ├─────┤ ├─────┤     ├─────┤
│addr32││addr33││addr34│     │addr63│
└─────┘ └─────┘ └─────┘     └─────┘
```

**Bank 冲突类型**:

| 访问模式 | 说明 | 性能 |
|---------|------|------|
| 无冲突 | 每个线程访问不同 Bank | ✅ 全速 |
| 广播访问 | 多个线程读同一地址 | ✅ 广播优化 |
| 2-way 冲突 | 2个线程访问同一Bank | ❌ 2倍延迟 |
| N-way 冲突 | N个线程访问同一Bank | ❌ N倍延迟 |

**避免 Bank 冲突的技巧**:

```cpp
// ❌ 可能冲突: stride = 32
__shared__ float sh[1024];
float val = sh[threadIdx.x];

// ✅ 填充避免冲突: stride = 33 (增加一个padding)
__shared__ float sh[1024 + 32];
float val = sh[threadIdx.x + threadIdx.x / 32];
```

---

### 1.3 常量内存 (Constant Memory) 📏

**位置**: GPU 板载显存 + 专用缓存

#### 特性

| 属性 | 值 |
|------|-----|
| 容量 | 最多 64KB |
| 速度 | ~1-30 cycles (命中缓存) |
| 作用域 | 所有线程可读 |
| 生命周期 | 应用程序级 |
| 关键字 | `__constant__` |
| 更新方式 | Host 端更新 |

#### 使用方式

```cpp
// 声明常量内存 (文件作用域)
__constant__ float const_data[256];

// Host 端初始化
float host_data[256] = {...};
cudaMemcpyToSymbol(const_data, host_data, sizeof(host_data));

// 内核中使用 (只读)
__global__ void kernel(float* output) {
    int idx = threadIdx.x;
    output[idx] = const_data[idx] * 2.0f;
}
```

**最佳适用场景**:
- 所有线程访问相同数据
- 只读数据
- 小尺寸数据 (<64KB)

---

### 1.4 纹理内存 (Texture Memory) 🎨

**位置**: GPU 板载显存 + 纹理缓存

#### 特性

| 属性 | 值 |
|------|-----|
| 容量 | 可达 GB 级 |
| 速度 | ~10-100 cycles (空间局部性好时快) |
| 作用域 | 所有线程 |
| 访问模式 | 优化空间局部性访问 |
| 特性 | 自动插值、边界处理 |

#### 使用场景

- 图像处理
- 空间局部性好的访问模式
- 需要插值或边界检查

---

### 1.5 本地内存 (Local Memory) 📍

**位置**: 全局内存 (不是片上！)

#### 特性

| 属性 | 值 |
|------|-----|
| 容量 | 每个线程最多 512KB |
| 速度 | 与全局内存相同 (~200-800 cycles) |
| 作用域 | 单个线程私有 |
| 触发条件 | 寄存器溢出 |

**什么时候使用本地内存**:
- 大数组无法放入寄存器时
- 编译时大小未知的数组
- 寄存器不足时的溢出

**避免本地内存的方法**:
- 减少每个线程的寄存器使用
- 使用共享内存替代
- 调整 Block 大小提高占用率

```cpp
__global__ void kernel() {
    // 这个数组可能放在本地内存
    float local_arr[100];  // 太大，寄存器放不下
    
    // 这个变量优先放在寄存器
    float reg_var = 1.0f;
}
```

---

## 2. 内存访问模式与合并访问 🔗

### 2.1 什么是合并访问？

**合并访问 (Coalesced Memory Access)** 是指一个 Warp 内的 32 个线程访问全局内存时，能够合并成较少的内存事务。

### 2.2 内存事务大小

| 架构 | 缓存行大小 | 事务粒度 |
|------|-----------|----------|
| Kepler | 128 bytes | 32, 64, 128 bytes |
| Maxwell/Pascal | 128 bytes | 32 bytes |
| Volta/Ampere | 128 bytes | 32 bytes |

### 2.3 访问模式对比

#### ✅ 理想的合并访问

```
Warp 32个线程访问连续对齐的地址:

Thread:  0   1   2   ...   31
        ↓   ↓   ↓         ↓
Addr:  0x0 0x4 0x8 ... 0x7C  (连续的 128 bytes)

┌─────────────────────────────────┐
│         单个 128-byte 事务      │
└─────────────────────────────────┘
```

#### ⚠️ 有间隔的访问

```
每个线程间隔访问 (stride = 2):

Thread:  0   1   2   ...   31
        ↓   ↓   ↓         ↓
Addr:  0x0 0x8 0x10 ... 0xF8

需要 2 个事务，带宽利用率 50%
```

#### ❌ 完全随机访问

```
每个线程访问随机地址:

Thread:  0    1    2    ...    31
        ↓    ↓    ↓           ↓
Addr:  乱序、分散的地址

需要 32 个事务，带宽利用率 1/32
```

### 2.4 合并访问条件表

| 数据类型 | 对齐要求 | 理想访问模式 |
|---------|---------|-------------|
| char (1B) | 1 byte | threadIdx.x |
| short (2B) | 2 bytes | threadIdx.x |
| int/float (4B) | 4 bytes | threadIdx.x |
| long/double (8B) | 8 bytes | threadIdx.x |
| float4 (16B) | 16 bytes | threadIdx.x |

### 2.5 代码示例对比

```cpp
// ✅ 好: 合并访问
__global__ void good_access(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // Warp 内连续访问
    }
}

// ❌ 坏: 非合并访问 (间隔访问)
__global__ void bad_access(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x * 2;  // stride=2
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// ✅ 修复: 调整索引模式
__global__ void fixed_access(float* data, int n) {
    int idx = blockIdx.x * 512 + threadIdx.x;
    if (idx < n) {
        data[idx * 2] = data[idx * 2] * 2.0f;
        data[idx * 2 + 1] = data[idx * 2 + 1] * 2.0f;
    }
}
```

---

## 3. 内存带宽计算 📐

### 3.1 理论带宽公式

```
理论带宽 (GB/s) = 内存频率 (GHz) × 总线宽度 (bit) × 倍率 / 8

其中:
- 倍率: DDR=2, GDDR5=4, GDDR5X=8, HBM2=2
```

### 3.2 常见 GPU 理论带宽

| GPU | 显存类型 | 总线宽度 | 频率 | 理论带宽 |
|-----|---------|---------|------|---------|
| V100 | HBM2 | 4096 bit | 0.877 GHz | ~900 GB/s |
| A100 | HBM2 | 5120 bit | 1.215 GHz | ~1555 GB/s |
| RTX 3090 | GDDR6X | 384 bit | 1.215 GHz | ~936 GB/s |
| RTX 4090 | GDDR6X | 384 bit | 1.313 GHz | ~1008 GB/s |
| A10 | GDDR6 | 320 bit | 1.25 GHz | ~600 GB/s |

### 3.3 实际带宽计算公式

```
实际带宽 (GB/s) = (读取字节数 + 写入字节数) / 时间 / 1e9

读取字节数 = 元素数量 × 元素大小
写入字节数 = 元素数量 × 元素大小
```

**示例代码 - 带宽测试**:

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void copy_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

int main() {
    int n = 1 << 26;  // 64M 元素
    size_t bytes = n * sizeof(float);
    
    float *d_src, *d_dst;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);
    
    // 预热
    copy_kernel<<<n/256, 256>>>(d_dst, d_src, n);
    cudaDeviceSynchronize();
    
    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    copy_kernel<<<n/256, 256>>>(d_dst, d_src, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    // 计算带宽: 读 + 写
    float gb = (bytes * 2.0f) / (1 << 30);  // 转换为 GB
    float gbps = gb / (ms / 1000.0f);       // GB/s
    
    std::cout << "带宽: " << gbps << " GB/s" << std::endl;
    
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
```

### 3.4 带宽利用率

```
带宽利用率 (%) = 实际带宽 / 理论带宽 × 100
```

**目标**: 达到理论带宽的 70%+ 就算优秀 🎯

### 3.5 影响带宽的因素

| 因素 | 影响 | 优化方向 |
|------|------|---------|
| 合并访问 | 影响最大 | 确保 Warp 内连续访问 |
| 内存对齐 | 显著影响 | 使用 cudaMalloc (自动对齐) |
| 数据类型 | 粒度影响 | 使用向量类型 (float2, float4) |
| 缓存使用 | L2 缓存影响 | 利用数据局部性 |
| 访问模式 | 读写比例 | 平衡读写，避免随机访问 |

---

## 4. 内存优化总结 📋

### 4.1 内存类型选择指南

| 访问模式 | 推荐内存类型 |
|---------|-------------|
| 线程私有临时变量 | 寄存器 |
| Block 内共享数据 | 共享内存 |
| 全局只读小数据 | 常量内存 |
| 空间局部性好的访问 | 纹理内存 |
| 大尺寸数据读写 | 全局内存 |

### 4.2 优化优先级

1. **合并访问** - 最关键的优化
2. **减少全局内存访问** - 使用共享/寄存器
3. **避免 Bank 冲突** - 共享内存优化
4. **利用缓存层次** - L1/L2 缓存
5. **数据预取** - 重叠计算与内存访问

### 4.3 常见问题排查表

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 带宽利用率低 | 非合并访问 | 调整索引模式 |
| 共享内存性能差 | Bank 冲突 | 添加 padding 或调整访问模式 |
| 寄存器溢出 | 每个线程寄存器太多 | 减少寄存器使用，调整 Block 大小 |
| 常量内存性能差 | Warp 内访问不同地址 | 确保 Warp 内访问相同地址 |

---

## 总结 📝

| 关键概念 | 核心要点 |
|---------|---------|
| 全局内存 | 容量大但慢，必须合并访问 |
| 共享内存 | 速度快但小，注意 Bank 冲突 |
| 常量内存 | 只读小数据，所有线程同地址时最优 |
| 本地内存 | 本质是全局内存，尽量避免 |
| 合并访问 | Warp 内连续对齐访问，性能关键 |
| 带宽计算 | 理论 vs 实际，衡量优化效果 |

掌握内存模型是 CUDA 性能优化的核心！🎯
