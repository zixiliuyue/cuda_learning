# CUDA API 速查表 (CHEATSHEET)

> 一页速查常用 CUDA C++ API。配合本项目 40 天课程使用。
> 目标环境：CUDA 12.8~13.x / RTX 5060 Ti (sm_120)

---

## 一、Kernel 定义与启动

### 函数限定符
| 限定符 | 执行位置 | 调用位置 | 说明 |
|--------|---------|---------|------|
| `__global__` | 设备(GPU) | 主机/设备 | Kernel 入口，返回 void，异步执行 |
| `__device__` | 设备(GPU) | 设备 | 只能被 Kernel 或其他 device 函数调用 |
| `__host__` | 主机(CPU) | 主机 | 普通 CPU 函数（默认） |
| `__host__ __device__` | 两者 | 两者 | 同一份代码编译两个版本 |

### Kernel 启动语法
```cpp
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
//        ↑网格维度  ↑块维度   ↑动态共享内存    ↑流(可选)
```

### 内建变量（Kernel 内可用）
```cpp
threadIdx.x/y/z    // 线程在块内的索引
blockIdx.x/y/z     // 块在网格内的索引
blockDim.x/y/z     // 每个块的线程数
gridDim.x/y/z      // 网格的块数
// 全局线程 ID（一维）：
int tid = blockIdx.x * blockDim.x + threadIdx.x;
// 网格跨步循环（处理超大数组）：
for (int i = tid; i < n; i += blockDim.x * gridDim.x) { ... }
```

---

## 二、内存管理

### 分配与释放
```cpp
cudaMalloc((void**)&d_ptr, bytes);          // 设备全局内存
cudaMallocHost((void**)&h_ptr, bytes);      // 页锁定主机内存(异步传输必需)
cudaMallocManaged((void**)&ptr, bytes);     // 统一内存(CPU/GPU 自动迁移)
cudaFree(d_ptr);                            // 释放设备内存
cudaFreeHost(h_ptr);                        // 释放页锁定内存
```

### 数据传输
```cpp
cudaMemcpy(dst, src, bytes, kind);          // 同步拷贝
cudaMemcpyAsync(dst, src, bytes, kind, stream);  // 异步拷贝(需页锁定内存)
// kind 取值：
//   cudaMemcpyHostToDevice   H2D
//   cudaMemcpyDeviceToHost   D2H
//   cudaMemcpyDeviceToDevice D2D
cudaMemset(d_ptr, value, bytes);            // 按字节填充
```

---

## 三、内存类型（Kernel 内声明）

| 声明 | 作用域 | 速度 | 用途 |
|------|--------|------|------|
| `__shared__ float s[N];` | Block 内共享 | 很快 | Tiling、块内规约 |
| `__constant__ float c[N];` | 全局只读 | 快(有缓存) | 滤波核、常量参数 |
| 局部变量 | 线程私有 | 最快(寄存器) | 临时计算 |
| `extern __shared__ float s[];` | 动态共享内存 | 很快 | 大小运行时决定 |

```cpp
// 动态共享内存：启动时第 3 个参数指定大小
kernel<<<grid, block, N * sizeof(float)>>>(...);
```

---

## 四、同步

```cpp
__syncthreads();               // Block 内所有线程同步(Kernel 内)
__syncwarp();                  // Warp 内线程同步
cudaDeviceSynchronize();       // 主机等待所有 GPU 任务完成
cudaStreamSynchronize(stream); // 主机等待指定流完成
cudaEventSynchronize(event);   // 主机等待事件完成
```
⚠️ `__syncthreads()` 必须被 Block 内**所有**线程执行，放在 if 分支内会死锁。

---

## 五、错误检查（务必使用）

```cpp
// 检查 API 返回值
#define CUDA_SAFE_CALL(call) do {                        \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        printf("CUDA错误 %s:%d: %s\n", __FILE__, __LINE__,\
               cudaGetErrorString(err));                 \
        exit(1);                                         \
    }                                                    \
} while(0)

// 检查 Kernel 启动错误（Kernel 无返回值）
kernel<<<g, b>>>(...);
CUDA_SAFE_CALL(cudaGetLastError());        // 检查启动配置错误
CUDA_SAFE_CALL(cudaDeviceSynchronize());   // 检查执行期错误
```
> 本项目封装见 `common/include/cuda_utils.h` 的 `CUDA_SAFE_CALL` / `CHECK_KERNEL_ERROR`。

---

## 六、流（Stream）与并发

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaMemcpyAsync(..., stream);        // 绑定到流
kernel<<<g, b, 0, stream>>>(...);    // 绑定到流
cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
// 多流可实现"拷贝-计算"重叠，提升吞吐
```

---

## 七、事件（Event）与计时

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
kernel<<<g, b>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);   // GPU 精确计时(毫秒)
cudaEventDestroy(start); cudaEventDestroy(stop);
```
> 本项目封装见 `common/include/timer.h` 的 `GpuTimer`。

---

## 八、Warp 级原语（sm_70+）

```cpp
// Warp 内规约(求和示例)
for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
// 其他：
__shfl_sync(mask, var, srcLane);       // 广播
__shfl_up_sync(mask, var, delta);      // 向上偏移
__shfl_xor_sync(mask, var, laneMask);  // 蝶形交换
__ballot_sync(mask, predicate);        // 投票
__all_sync / __any_sync                // 全部/任一为真
```

---

## 九、常用原子操作

```cpp
atomicAdd(&addr, val);     // 原子加(支持 int/float/double)
atomicSub / atomicMax / atomicMin
atomicExch(&addr, val);    // 原子交换
atomicCAS(&addr, cmp, val);// 比较并交换(实现自定义原子操作)
```

---

## 十、数学函数（设备端）

```cpp
// 快速近似(硬件加速，精度略低)
__fdividef(a, b);   rsqrtf(x);   __expf(x);   __logf(x);   __sinf(x);
// 标准精度
sqrtf / expf / logf / powf / tanhf / fabsf / fminf / fmaxf
// 融合乘加(1 条指令，精度更高)
fmaf(a, b, c);      // a*b+c
```

---

## 十一、常用启动配置计算

```cpp
int blockSize = 256;                              // 每块线程数(32 的倍数)
int gridSize = (n + blockSize - 1) / blockSize;   // 向上取整
// 2D 配置(图像处理)：
dim3 block(16, 16);
dim3 grid((width + 15) / 16, (height + 15) / 16);
// 查询最优 blockSize：
int minGrid, bs;
cudaOccupancyMaxPotentialBlockSize(&minGrid, &bs, kernel, 0, 0);
```

---

## 十二、编译命令速查

```bash
# 基本编译
nvcc -arch=sm_120 main.cu -o app

# 常用选项
-arch=sm_120          # 目标架构(RTX 50 系 Blackwell)
-I path               # 头文件目录
-lcublas -lcufft      # 链接库(cuBLAS/cuFFT/cusparse)
-rdc=true             # 可重定位设备代码(动态并行需要)
-Xcompiler /utf-8     # (Windows)中文源码编码
-lineinfo             # 源码级性能分析信息(Nsight)
-O3 / -O1             # 优化等级
-G                    # 调试信息(cuda-gdb)

# 性能分析
ncu --set full ./app         # Nsight Compute(Kernel 级)
nsys profile ./app           # Nsight Systems(时间线)
compute-sanitizer ./app      # 内存越界检测(原 cuda-memcheck)
```

---

## 十三、Compute Capability 架构对照

| CC | 架构 | 代表 GPU |
|----|------|---------|
| 7.0 | Volta | V100 |
| 7.5 | Turing | RTX 20 系 / T4 |
| 8.0 / 8.6 | Ampere | A100 / RTX 30 系 |
| 8.9 | Ada Lovelace | RTX 40 系 |
| 9.0 | Hopper | H100 |
| 10.0~12.x | Blackwell | RTX 50 系 / B200 |

> ⚠️ CUDA 13 最低支持 sm_75，`memoryClockRate` 等字段已移除，改用 `cudaDeviceGetAttribute()`。

---

*配合 40 天课程速查使用 · CUDA 40-Day Learning Project*
