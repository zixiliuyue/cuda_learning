# CUDA 性能反模式

---

## 一、内存访问反模式

### 1. 非合并全局内存访问

**问题描述：**
warp内线程访问全局内存的地址不连续、不对齐，导致硬件无法合并内存事务，大幅降低内存带宽利用率。

**示例代码：**
```cpp
__global__ void uncoalesced_access(float* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ❌ 步长访问，地址不连续
    float val = data[idx * stride];
    
    // ❌ 随机访问，完全无法合并
    float val2 = data[random_index[idx]];
    
    // ❌ 结构体数组，单字段访问可能非合并
    struct Point { float x, y, z; };
    Point* points = (Point*)data;
    float x = points[idx].x;  // 每个线程间隔12字节访问
}
```

**性能影响：**
- 内存带宽利用率可能只有10-30%
- 最坏情况（步长32）性能下降32倍

**解决方法：**
```cpp
// ✅ 连续对齐访问
float val = data[idx];

// ✅ AoS转SoA
struct PointsSoA {
    float* x;
    float* y;
    float* z;
};
float x = points.x[idx];  // 连续4字节访问

// ✅ 使用向量类型加载
float4 vec = *reinterpret_cast<float4*>(data + idx * 4);
```

---

### 2. 共享内存Bank冲突

**问题描述：**
同一个warp内的多个线程访问共享内存的同一个Bank，导致内存访问串行化。

**示例代码：**
```cpp
__global__ void bank_conflict_example(float* data) {
    __shared__ float sdata[128];
    int tid = threadIdx.x;
    
    // ❌ 2路冲突：tid和tid+64访问同一个Bank
    sdata[tid * 2] = data[tid];
    
    // ❌ 严重冲突：所有线程访问Bank 0
    sdata[0] = data[tid];
    
    // ❌ 步长访问导致冲突
    for (int i = 0; i < 128; i++) {
        data[tid * 128 + i] = sdata[i * 128 + tid];  // 步长128
    }
}
```

**性能影响：**
- N路冲突性能下降N倍
- 严重情况可能慢10倍以上

**解决方法：**
```cpp
// ✅ 方法1: 填充padding
__shared__ float sdata[128 + 1];  // 多一个元素
sdata[tid * 2 + (tid / 32)] = data[tid];

// ✅ 方法2: 调整访问顺序
for (int i = 0; i < 128; i++) {
    data[tid * 128 + i] = sdata[tid * 128 + i];  // 无冲突
}

// ✅ 方法3: 使用向量类型
__shared__ float4 svec[32];
svec[tid] = ...;
```

---

### 3. 全局内存访问粒度不足

**问题描述：**
每个线程只访问少量数据，导致内存请求过多，无法充分利用总线宽度。

**示例代码：**
```cpp
__global__ void small_access(float* data, int n) {
    int idx = blockIdx.x * 64 + threadIdx.x;
    
    // ❌ 每个线程只访问一个float（4字节）
    float val = data[idx];
    compute(val);
}
```

**解决方法：**
```cpp
__global__ void large_access(float* data, int n) {
    int idx = blockIdx.x * 64 + threadIdx.x;
    
    // ✅ 每个线程访问多个元素
    float vals[8];
    for (int i = 0; i < 8; i++) {
        vals[i] = data[idx + i * 32];
    }
    
    for (int i = 0; i < 8; i++) {
        compute(vals[i]);
    }
}
```

---

## 二、线程调度反模式

### 4. 分支发散严重

**问题描述：**
同一个warp内线程执行不同的代码路径，导致SIMT硬件串行执行所有路径。

**示例代码：**
```cpp
__global__ void bad_branch(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ❌ 每个线程可能走不同路径
    if (data[idx] > threshold) {
        path_A(data[idx]);
    } else {
        path_B(data[idx]);
    }
    
    // ❌ 循环次数不同
    for (int i = 0; i < data[idx]; i++) {  // warp内循环次数不同
        compute(i);
    }
}
```

**性能影响：**
- 最坏情况性能下降32倍
- 分支效率低于50%时需要优化

**解决方法：**
```cpp
// ✅ 方法1: 重新组织线程
__global__ void grouped_branch(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // warp内前半走A，后半走B
    if (tid < 16) {
        path_A(data[idx]);
    } else {
        path_B(data[idx]);
    }
}

// ✅ 方法2: 预计算，两个路径都执行
float a = compute_path_A(data[idx]);
float b = compute_path_B(data[idx]);
data[idx] = condition ? a : b;

// ✅ 方法3: 使用内建指令（如__shfl_*系列）
```

---

### 5. Block大小不合理

**问题描述：**
- Block太小：无法充分利用SM资源，调度开销大
- Block太大：占用率低，无法隐藏延迟
- Block大小不是32的倍数：warp不完整

**示例代码：**
```cpp
// ❌ Block太小
kernel<<<10000, 16>>>(data);

// ❌ Block不是32倍数
kernel<<<1000, 50>>>(data);
```

**建议：**
```cpp
// ✅ 推荐的Block大小
dim3 block_32(32);    // 最小合理值
dim3 block_64(64);    // 适合小内核
dim3 block_128(128);  // 通用推荐
dim3 block_256(256);  // 常用
dim3 block_512(512);  // 寄存器需求小
dim3 block_1024(1024); // 最大（注意寄存器限制）

// 2D/3D Block也要保证总线程数是32的倍数
dim3 block2d(16, 16);  // 256线程 ✅
dim3 block3d(8, 8, 4); // 256线程 ✅
```

---

### 6. 低占用率（Occupancy）

**问题描述：**
每个SM上活跃的warp数量太少，无法有效隐藏内存和指令延迟。

**常见原因：**
- 每个线程使用寄存器太多
- 每个Block使用共享内存太多
- Block大小不合理

**检测与优化：**
```bash
# 使用Nsight Compute查看占用率
ncu --metrics achieved_occupancy ./app
```

**解决方法：**
```cpp
// ✅ 限制寄存器使用
__launch_bounds__(256, 8)  // Block大小256，每个SM至少8个Block
__global__ void kernel(float* data) {
    // 编译器会限制寄存器使用
}

// ✅ 减少共享内存使用
// 使用更多全局内存访问或寄存器重用

// ✅ 调整Block大小
```

---

## 三、同步与通信反模式

### 7. 过度同步

**问题描述：**
不必要的`__syncthreads()`调用，或者同步频率过高。

**示例代码：**
```cpp
__global__ void over_sync(float* data) {
    __shared__ float sdata[128];
    int tid = threadIdx.x;
    
    sdata[tid] = data[tid];
    __syncthreads();  // 需要
    
    float val1 = sdata[tid] * 2;
    __syncthreads();  // ❌ 不需要，没有新的写入
    
    float val2 = val1 + sdata[127 - tid];
    __syncthreads();  // ❌ 不需要
    
    data[tid] = val2;
}
```

**性能影响：**
- 同步操作有开销（约几十个周期）
- 同步是Barrier，所有线程必须等待
- 过度同步降低有效计算时间

---

### 8. 分支内同步

**问题描述：**
在条件分支内调用`__syncthreads()`，可能导致死锁。

**示例代码：**
```cpp
__global__ void branch_sync_bug(float* data) {
    __shared__ float sdata[128];
    int tid = threadIdx.x;
    
    // ❌ 危险！只有部分线程到达同步点
    if (tid < 64) {
        sdata[tid] = data[tid];
        __syncthreads();  // 另一半线程永远不会到达这里
        data[tid] = sdata[tid + 64];
    }
}
```

**正确做法：**
```cpp
__global__ void branch_sync_fixed(float* data) {
    __shared__ float sdata[128];
    int tid = threadIdx.x;
    
    // 所有线程写入
    sdata[tid] = data[tid];
    
    // ✅ 同步在分支外，所有线程都到达
    __syncthreads();
    
    if (tid < 64) {
        data[tid] = sdata[tid + 64];
    }
}
```

---

### 9. 原子操作滥用

**问题描述：**
不必要地使用原子操作，或者原子操作冲突率太高。

**示例代码：**
```cpp
__global__ void atomic_abuse(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ❌ 所有线程竞争同一个原子变量
    atomicAdd(counter, 1);
    
    // ❌ 高频原子操作
    for (int i = 0; i < 1000; i++) {
        atomicAdd(&histogram[data[idx + i]], 1);
    }
}
```

**解决方法：**
```cpp
// ✅ 方法1: Block内归约，最后一次原子
__global__ void reduced_atomic(int* counter, int n) {
    __shared__ int block_counter;
    int tid = threadIdx.x;
    
    // Block内计数
    int local_count = 0;
    for (int i = tid; i < n / gridDim.x; i += blockDim.x) {
        local_count += condition(i);
    }
    
    // Block内归约
    __shared__ int reduce[128];
    reduce[tid] = local_count;
    __syncthreads();
    
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    
    // 只在tid=0时做一次原子操作
    if (tid == 0) {
        atomicAdd(counter, reduce[0]);
    }
}

// ✅ 方法2: 使用per-thread计数器数组
__global__ void per_thread_atomic(int* counters, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程只更新自己的计数器
    counters[idx] = compute_count(idx);
}
// CPU端最后汇总所有计数器
```

---

## 四、算法与架构反模式

### 10. 全局内存通信模式低效

**问题描述：**
并行算法的通信模式导致大量全局内存访问或全局同步。

**示例：**
```cpp
// ❌ 频繁的全局内存读写通信
__global__ void step1(float* data) { ... }
__global__ void step2(float* data) { ... }
__global__ void step3(float* data) { ... }

// 多次内核启动，中间数据都写回全局内存
step1<<<grid, block>>>(data);
step2<<<grid, block>>>(data);
step3<<<grid, block>>>(data);
```

**解决方法：**
```cpp
// ✅ 融合内核，中间数据保存在寄存器/共享内存
__global__ void fused_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = data[idx];
    
    val = step1_compute(val);  // 寄存器中完成
    val = step2_compute(val);  // 不需要写回全局
    val = step3_compute(val);
    
    data[idx] = val;
}
```

---

### 11. 小内核过多

**问题描述：**
大量小内核（执行时间<1ms）导致启动开销占比过高。

**性能影响：**
- 内核启动开销约5-50µs
- 如果内核只执行100µs，开销占比达50%

**解决方法：**
```cpp
// ❌ 多个小内核
small_kernel1<<<grid, block>>>(data);
small_kernel2<<<grid, block>>>(data);
small_kernel3<<<grid, block>>>(data);

// ✅ 融合成一个大内核
big_fused_kernel<<<grid, block>>>(data);

// ✅ 或者使用CUDA Graph减少启动开销
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);
// 向图中添加多个内核节点
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
// 一次启动执行所有内核
cudaGraphLaunch(exec, stream);
```

---

### 12. 负载不均衡

**问题描述：**
不同线程/Block的工作量差异很大，导致部分SM空闲等待。

**示例：**
```cpp
__global__ void unbalanced_work(float* data, int* sizes) {
    int idx = blockIdx.x;
    
    // ❌ 每个Block的工作量差异很大
    for (int i = 0; i < sizes[idx]; i++) {
        compute(data[idx * 1024 + i]);
    }
}
```

**解决方法：**
```cpp
// ✅ 方法1: 动态任务分配
__global__ void dynamic_schedule(float* data, int* tasks, int num_tasks) {
    __shared__ int shared_task_idx;
    if (threadIdx.x == 0) shared_task_idx = 0;
    __syncthreads();
    
    while (true) {
        // 原子获取任务
        int task = atomicAdd(&shared_task_idx, 1);
        if (task >= num_tasks) break;
        
        // 执行任务
        process_task(data, tasks[task]);
    }
}

// ✅ 方法2: 任务重排，均衡分配
// CPU端预先对任务排序，将大任务和小任务混合分配
```

---

## 五、流与数据传输反模式

### 13. 数据传输未重叠

**问题描述：**
数据传输和内核执行串行进行，GPU空闲等待数据。

**示例代码：**
```cpp
// ❌ 完全串行
for (int i = 0; i < batches; i++) {
    cudaMemcpy(d_data, h_data[i], size, H2D);  // CPU等待
    kernel<<<grid, block>>>(d_data);             // GPU计算，CPU空闲
    cudaMemcpy(h_result[i], d_result, size, D2H);
    cudaDeviceSynchronize();
}
```

**解决方法：**
```cpp
// ✅ 使用多流重叠
const int NUM_STREAMS = 4;
cudaStream_t streams[NUM_STREAMS];
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
}

for (int i = 0; i < batches + NUM_STREAMS - 1; i++) {
    for (int s = 0; s < NUM_STREAMS; s++) {
        int batch = i - s;
        if (batch >= 0 && batch < batches) {
            if (s == 0) {
                cudaMemcpyAsync(d_data[s], h_data[batch], size, H2D, streams[s]);
            } else if (s == 1) {
                kernel<<<grid, block, 0, streams[s]>>>(d_data[s-1], d_result[s]);
            } else {
                cudaMemcpyAsync(h_result[batch], d_result[s-1], size, D2H, streams[s]);
            }
        }
    }
}
```

---

### 14. 过多小数据传输

**问题描述：**
频繁传输小块数据，PCIe协议开销占比过高。

**性能影响：**
- PCIe延迟约1-5µs
- 传输1KB和64KB的延迟差异很小
- 小数据传输效率很低

**解决方法：**
```cpp
// ✅ 批量传输
// 将多次小传输合并成一次大传输

// ✅ 使用固定内存（Pinned Memory）
cudaMallocHost(&h_data, size);  // 减少传输时间约30%

// ✅ 使用零拷贝（需要权衡）
cudaHostAlloc(&h_data, size, cudaHostAllocMapped);
// GPU直接访问主机内存，省去显式拷贝
// 但访问延迟较高，适合只读一次的数据
```

---

## 六、性能优化检查清单

### 优化优先级排序

1. **首先检查**（性能影响最大）
   - [ ] 全局内存访问是否合并？
   - [ ] 共享内存是否有Bank冲突？
   - [ ] 分支发散是否严重？
   - [ ] 是否有不必要的全局内存读写？

2. **其次检查**
   - [ ] 占用率是否合理（>30%）？
   - [ ] Block大小是否合适（128-512）？
   - [ ] 内核是否太小（启动开销占比高）？
   - [ ] 同步是否过多？

3. **最后检查**（收益较小）
   - [ ] 指令级并行是否充分？
   - [ ] 是否可以使用向量指令？
   - [ ] 循环是否可以展开？
   - [ ] 原子操作是否可以优化？

---

### 性能分析流程

1. **Nsight Systems 整体分析**
   - GPU利用率是否足够？
   - 数据传输和计算是否重叠？
   - 哪个内核耗时最长？

2. **Nsight Compute 内核分析**
   - 内存带宽利用率（理想>70%）
   - 分支效率（理想>90%）
   - 占用率（理想30-70%）
   - 指令吞吐量

3. **针对性优化**
   - 根据瓶颈选择优化方案
   - 优化后重新验证
   - 记录性能变化

---

## 七、常见优化误区

### 误区1: 过早优化
- 先保证正确性，再优化性能
- 使用工具定位瓶颈，不要凭感觉优化

### 误区2: 过度优化
- 每个优化都有代价（代码复杂度、可读性）
- 优化到满足需求即可，不需要追求极致

### 误区3: 只看Flops
- 大多数CUDA程序是内存瓶颈，不是计算瓶颈
- 优化内存访问通常比优化计算收益更大

### 误区4: 忽略启动开销
- 小内核要注意启动开销
- 使用CUDA Graph或内核融合减少开销

---

## 总结

避免CUDA性能反模式的核心原则：
1. **内存优先**：优化内存访问是最重要的
2. **减少同步**：同步越少越好，越晚越好
3. **均衡负载**：让所有SM都忙碌起来
4. **重叠操作**：计算和数据传输尽量重叠
5. **数据局部**：重用寄存器和共享内存
6. **测量验证**：用工具而不是猜测，每次优化后验证
