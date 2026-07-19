# 📝 CUDA 速查表

## 🔧 编译命令

| 命令 | 说明 |
|------|------|
| `nvcc file.cu -o program` | 编译 CUDA 文件 |
| `nvcc -arch=sm_75 file.cu` | 指定 GPU 架构编译 |
| `nvcc -O3 file.cu` | 优化编译 |
| `nvcc -g -G file.cu` | 调试编译（CPU+GPU） |
| `nvcc --ptx file.cu` | 生成 PTX 汇编代码 |

---

## 🧠 CUDA 内存类型

| 内存类型 | 位置 | 访问速度 | 作用域 | 大小限制 |
|----------|------|----------|--------|----------|
| 寄存器 Register | 片上 | 最快 | 线程 | 有限 |
| 共享内存 Shared | 片上 | 很快 | Thread Block | 几十KB |
| 常量内存 Constant | 全局 | 较慢 | 全局 | 64KB |
| 纹理内存 Texture | 全局 | 较慢 | 全局 | 较大 |
| 全局内存 Global | 全局 | 慢 | 全局 | 全部显存 |

---

## 📊 线程层次结构

```
Grid (网格)
├── Block 0 (线程块)
│   ├── Thread (0,0)
│   ├── Thread (1,0)
│   └── ...
├── Block 1
│   ├── Thread (0,0)
│   └── ...
└── ...
```

### 内置变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `threadIdx.x/y/z` | uint3 | 线程在块内的索引 |
| `blockIdx.x/y/z` | uint3 | 块在网格内的索引 |
| `blockDim.x/y/z` | dim3 | 每个块的线程数 |
| `gridDim.x/y/z` | dim3 | 网格的块数 |
| `warpSize` | int | Warp 大小（通常32） |

---

## 🔄 内存操作函数

| 函数 | 说明 |
|------|------|
| `cudaMalloc(&devPtr, size)` | 分配设备内存 |
| `cudaFree(devPtr)` | 释放设备内存 |
| `cudaMemcpy(dst, src, size, kind)` | 内存拷贝 |
| `cudaMemcpyAsync(...)` | 异步内存拷贝 |
| `cudaMemset(devPtr, val, size)` | 设置设备内存 |

### cudaMemcpyKind

| 类型 | 说明 |
|------|------|
| `cudaMemcpyHostToDevice` | CPU -> GPU |
| `cudaMemcpyDeviceToHost` | GPU -> CPU |
| `cudaMemcpyDeviceToDevice` | GPU -> GPU |
| `cudaMemcpyHostToHost` | CPU -> CPU |

---

## ⚡ 内核启动

```cpp
// 基本语法
kernel_name<<<grid_size, block_size>>>(args);

// 带共享内存和流
kernel_name<<<grid_size, block_size, shared_size, stream>>>(args);
```

### 示例

```cpp
// 1维
dim3 block(256);
dim3 grid((N + block.x - 1) / block.x);

// 2维
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);
```

---

## 🛡️ 错误检查

```cpp
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// 使用方法
CUDA_CHECK(cudaMalloc(&devPtr, size));

// 内核启动后检查
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

---

## ⏱️ 性能测量

### CPU 计时
```cpp
#include <chrono>
auto start = std::chrono::high_resolution_clock::now();
// ... 代码 ...
auto end = std::chrono::high_resolution_clock::now();
auto ms = std::chrono::duration<float, std::milli>(end - start);
```

### CUDA Event 计时
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds;
cudaEventElapsedTime(&milliseconds, start, stop);
```

---

## 🚀 优化技巧速览

### 1. 内存访问模式
- ✅ 合并访问 (Coalescing)
- ✅ 避免内存冲突 (Bank Conflict)
- ✅ 使用共享内存缓存数据

### 2. 线程组织
- ✅ Block 大小应为 32 的倍数
- ✅ 建议 block 大小: 128, 256, 512
- ✅ 充分利用 SM（流式多处理器）

### 3. 指令优化
- ✅ 减少分支 divergence
- ✅ 使用内置函数 (__syncthreads, atomicAdd 等)

---

## 🔧 常用 CUDA 工具

| 工具 | 说明 |
|------|------|
| `nvprof` | NVIDIA 性能分析工具 |
| `nvidia-smi` | GPU 状态监控 |
| `cuda-gdb` | CUDA 调试器 |
| `nsys profile` | Nsight Systems 分析 |
| `nvdisasm` | 反汇编工具 |

### nvidia-smi 常用命令

```bash
nvidia-smi                    # 显示GPU状态
nvidia-smi dmon -s p         # 实时监控性能
nvidia-smi --query-gpu=memory.used --format=csv
```

---

## 📚 数学函数

| 函数 | 说明 |
|------|------|
| `atomicAdd(&addr, val)` | 原子加法 |
| `atomicSub(&addr, val)` | 原子减法 |
| `atomicExch(&addr, val)` | 原子交换 |
| `atomicMin(&addr, val)` | 原子最小值 |
| `atomicMax(&addr, val)` | 原子最大值 |
| `atomicInc(&addr, val)` | 原子递增 |
| `atomicDec(&addr, val)` | 原子递减 |
| `atomicCAS(&addr, comp, val)` | 原子比较交换 |

---

## 🔄 同步函数

| 函数 | 作用域 | 说明 |
|------|--------|------|
| `__syncthreads()` | Block | 块内线程同步 |
| `__threadfence()` | Grid | 内存访问栅栏 |
| `__threadfence_block()` | Block | 块内内存栅栏 |
| `cudaDeviceSynchronize()` | Device | 设备同步 |

---

## 💡 常见计算公式

### 网格/块大小计算
```cpp
int block_size = 256;
int grid_size = (N + block_size - 1) / block_size;
```

### 全局索引计算
```cpp
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;
```

### 带宽计算
```
带宽 (GB/s) = (数据字节数 × 2) / (时间秒 × 1e9)
// ×2 因为通常读写各一次
```

---

## ⚠️ 常见错误代码

| 错误码 | 说明 |
|--------|------|
| `cudaSuccess` | 成功 |
| `cudaErrorInvalidValue` | 无效参数 |
| `cudaErrorMemoryAllocation` | 内存分配失败 |
| `cudaErrorLaunchFailure` | 内核启动失败 |
| `cudaErrorLaunchOutOfResources` | 资源不足 |
| `cudaErrorInvalidDeviceFunction` | 无效设备函数 |

---

**🎯 保存此文件，随时查阅！**
