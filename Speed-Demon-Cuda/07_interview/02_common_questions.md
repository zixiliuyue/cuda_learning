# CUDA 常见面试题

---

## 一、基础概念

### Q1: 请解释CUDA的线程层次模型

**答案：**

CUDA的线程层次模型分为三级：

1. **Thread（线程）**：最基本的执行单元，每个线程有自己的寄存器和栈空间

2. **Block（线程块）**：
   - 由多个线程组成（通常32、64、128、256、512、1024个线程）
   - 同一个Block内的线程可以：
     - 通过共享内存（Shared Memory）通信
     - 通过`__syncthreads()`进行同步
   - 有自己的`blockIdx`和`blockDim`

3. **Grid（网格）**：
   - 由多个Block组成
   - 有自己的`gridDim`
   - 同一个Grid内的线程共享全局内存和常量内存

**线程索引计算：**
```cpp
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;
```

**设计原因：**
- 硬件层面：SM（流式多处理器）以warp为单位调度线程
- 编程层面：提供灵活的并行化抽象，适应不同的问题维度

---

### Q2: 什么是warp？warp大小是多少？

**答案：**

**warp的定义：**
- warp是GPU硬件调度的基本执行单元
- 一个warp包含32个线程（固定大小，所有NVIDIA GPU都是32）
- 同一个warp内的线程以SIMD（单指令多数据）方式执行同一条指令

**warp的重要特性：**

1. **SIMT执行模型**：
   - warp内的线程执行相同的指令
   - 但每个线程有自己的程序计数器和寄存器
   - 线程可以有不同的执行路径（分支发散）

2. **warp调度**：
   - SM以warp为单位进行调度
   - 当一个warp停顿（如等待内存）时，SM可以切换到其他就绪的warp
   - 这就是为什么高占用率有助于隐藏延迟

3. **内存访问合并**：
   - warp内的线程同时发起内存请求
   - 如果访问地址连续且对齐，硬件会合并为一个或几个内存事务

**面试关键点：**
- warp大小固定为32（重要！）
- warp是硬件调度单元
- 理解warp对理解分支发散、合并访问等概念很关键

---

### Q3: 解释全局内存、共享内存、常量内存的区别

**答案：**

| 内存类型 | 位置 | 大小 | 访问速度 | 作用域 | 特点 |
|---------|------|------|---------|--------|------|
| 全局内存 | 板载显存 | 大（GB级） | 慢（~400-800周期） | Grid | 所有线程可读写，容量大，延迟高 |
| 共享内存 | SM片上 | 小（~48KB/SM） | 快（~1-2周期） | Block | Block内线程共享，需显式管理，有Bank冲突问题 |
| 常量内存 | 板载显存 | 小（64KB） | 快（缓存后） | Grid | 只读，有专用缓存，适合所有线程读相同数据 |
| 寄存器 | SM片上 | 很小（KB级） | 最快（0周期） | Thread | 线程私有，编译器自动分配 |

**详细说明：**

**全局内存（Global Memory）：**
- 主要的存储区域，存储输入输出数据
- 延迟高，需要通过合并访问和缓存来提高效率
- 通过`cudaMalloc()`分配，`cudaMemcpy()`传输

**共享内存（Shared Memory）：**
- `__shared__`关键字声明
- 程序员显式控制的高速缓存
- 用于Block内线程间的数据共享
- 需要注意Bank冲突问题

**常量内存（Constant Memory）：**
- `__constant__`关键字声明
- 只读，有专门的常量缓存
- 当warp内所有线程访问同一个地址时效率最高
- 适合存储常量参数、查找表等

**使用场景选择：**
```cpp
__constant__ float kernel[MASK_SIZE];  // 卷积核，所有线程读相同

__global__ void conv(float* input, float* output) {
    __shared__ float tile[TILE_SIZE];   // Block内共享的输入块
    
    int idx = threadIdx.x;
    float reg_val = input[idx];         // 寄存器存储
    
    // ...
}
```

---

## 二、内存优化

### Q4: 什么是合并访问（Coalesced Access）？

**答案：**

**定义：**
合并访问是指warp内的线程在访问全局内存时，所有线程访问的地址满足一定的连续性和对齐条件，使得硬件可以将多个内存请求合并为少数几个内存事务，从而提高内存带宽利用率。

**为什么需要合并访问：**
- 全局内存延迟很高（~400-800时钟周期）
- 如果不合并，32个线程会产生32个内存事务
- 合并后可能只需要1-2个内存事务
- 性能差异可达10倍以上

**合并访问的条件（计算能力2.0+）：**

1. **对齐要求**：内存地址对齐到缓存行边界（通常128字节）
2. **连续要求**：warp内线程访问的地址连续
3. **类型要求**：访问的是相同的数据类型

**好的访问模式：**
```cpp
__global__ void good_access(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // thread 0: data[0]
        // thread 1: data[1]
        // ...
        // thread 31: data[31]
        // 地址连续，完全合并 ✅
        float val = data[idx];
        // ...
    }
}
```

**坏的访问模式：**
```cpp
__global__ void bad_access1(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 步长为2，地址不连续 ❌
        float val = data[idx * 2];
    }
}

__global__ void bad_access2(float* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 随机步长，地址完全不连续 ❌
        float val = data[idx * stride];
    }
}
```

**特殊情况处理：**
- 转置操作：需要使用共享内存做中转
- 结构体访问：确保结构体大小是32/64/128字节的倍数
- 间接访问：先收集索引，再批量访问

**性能影响量化：**
- 完全合并：可达 80-90% 理论带宽
- 严重非合并：可能只有 5-10% 理论带宽

---

### Q5: 什么是Bank冲突？如何避免？

**答案：**

**定义：**
共享内存被分成32个等大小的内存Bank（对应warp的32个线程）。当同一个warp内的多个线程访问同一个Bank的不同地址时，就会发生Bank冲突。Bank冲突会导致内存访问串行化，降低性能。

**Bank组织方式：**
- 32个Bank，编号0-31
- 每个Bank宽度通常为4字节（32位）
- 地址映射：`BankID = (Address / 4) % 32`

**Bank冲突类型：**

1. **2路冲突**：2个线程访问同一个Bank
2. **4路冲突**：4个线程访问同一个Bank
3. **N路冲突**：N个线程访问同一个Bank（性能降到1/N）

**冲突示例：**
```cpp
__global__ void bank_conflict_example() {
    __shared__ float shared[128];
    
    int tid = threadIdx.x;
    
    // ✅ 无冲突：tid, tid+32, tid+64... 在不同Bank
    shared[tid] = 1.0f;
    
    // ❌ 严重冲突：所有线程访问Bank 0
    shared[0] = 1.0f;
    
    // ❌ 2路冲突：tid和tid+16访问同一个Bank
    // 当tid=0和tid=16时，都访问Bank 0
    shared[tid * 2] = 1.0f;
}
```

**避免Bank冲突的方法：**

**方法1：填充（Padding）**
```cpp
// ❌ 有冲突
__shared__ float bad[128];
bad[tid * 4] = ...;  // 4路冲突

// ✅ 无冲突
__shared__ float good[128 + 1];  // 多一个元素填充
good[tid * 4 + (tid / 32)] = ...;
```

**方法2：调整访问模式**
```cpp
// ❌ 有冲突：行优先访问128x128的共享内存
for (int i = 0; i < 128; i++) {
    val += shared[i * 128 + tid];  // 步长128 = 4*32，冲突
}

// ✅ 无冲突：转置后列优先
for (int i = 0; i < 128; i++) {
    val += shared[tid * 128 + i];
}
```

**方法3：使用向量类型**
```cpp
// 使用float4等向量类型可以减少冲突概率
__shared__ float4 shared_vec[32];
shared_vec[tid] = ...;
```

**特殊情况 - 广播机制：**
当warp内所有线程访问同一个地址时，硬件会进行广播，这不是冲突，反而效率很高：
```cpp
__shared__ float shared[128];
// ✅ 广播，无性能损失
float val = shared[16];  // 所有线程读同一个地址
```

**Bank冲突检测：**
- 使用Nsight Compute的"Shared Memory Statistics"
- 查看"Bank Conflicts"指标
- 理想值：0

---

### Q6: 如何减少全局内存访问？

**答案：**

减少全局内存访问是CUDA优化的核心。以下是主要方法：

**1. 使用共享内存重用数据**
```cpp
// ❌ 每个元素被读多次
__global__ void matmul_bad(float* A, float* B, float* C, int N) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];  // N次全局内存读
    }
    C[row * N + col] = sum;
}

// ✅ 使用共享内存分块，减少重复读取
__global__ void matmul_good(float* A, float* B, float* C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;
    float sum = 0.0f;
    
    for (int m = 0; m < N / BLOCK_SIZE; m++) {
        As[ty][tx] = A[row * N + m * BLOCK_SIZE + tx];
        Bs[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];  // 共享内存访问
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

**2. 利用寄存器重用**
```cpp
__global__ void reuse_registers(float* data, int n) {
    int idx = blockIdx.x * 128 + threadIdx.x;
    
    // ✅ 把数据加载到寄存器，多次使用
    float reg[4];
    for (int i = 0; i < 4; i++) {
        reg[i] = data[idx + i * 32];  // 只加载一次
    }
    
    // 多次使用寄存器中的数据
    for (int i = 0; i < 100; i++) {
        compute(reg[0], reg[1], reg[2], reg[3]);
    }
}
```

**3. 合并内核，减少中间数据传输**
```cpp
// ❌ 多个内核，中间数据写回全局内存
kernel1<<<grid, block>>>(input, temp1);
kernel2<<<grid, block>>>(temp1, temp2);
kernel3<<<grid, block>>>(temp2, output);

// ✅ 融合成一个内核，中间数据保存在寄存器/共享内存
__global__ void fused_kernel(float* input, float* output) {
    int idx = ...;
    float val = input[idx];
    val = op1(val);  // 寄存器中完成
    val = op2(val);  // 不需要写回全局内存
    val = op3(val);
    output[idx] = val;
}
```

**4. 使用纹理内存/常量内存缓存**
- 纹理内存：适合二维数据访问、有滤波需求
- 常量内存：适合只读的小数据（<64KB）

**5. 数据重排优化内存局部性**
```cpp
// 例如：AoS（数组的结构体）转 SoA（结构体的数组）
// AoS - 不好
struct PointAoS { float x, y, z; };
PointAoS points[N];  // 访问x时y,z也被加载

// SoA - 好
struct PointSoA { float* x, * y, * z; };
PointSoA points;      // 只加载需要的数组
```

**6. 计算与数据传输重叠**
```cpp
// 使用多个流实现流水线
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 第1批：传输+计算+传输
cudaMemcpyAsync(d_data1, h_data1, size, H2D, stream1);
kernel<<<grid, block, 0, stream1>>>(d_data1);
cudaMemcpyAsync(h_result1, d_result1, size, D2H, stream1);

// 第2批：与第1批重叠执行
cudaMemcpyAsync(d_data2, h_data2, size, H2D, stream2);
kernel<<<grid, block, 0, stream2>>>(d_data2);
cudaMemcpyAsync(h_result2, d_result2, size, D2H, stream2);
```

---

## 三、性能分析与调优

### Q7: 解释占用率（Occupancy）的概念

**答案：**

**定义：**
占用率是指SM（流式多处理器）上活跃的warp数量与理论上最多可容纳的warp数量的比值。

**计算公式：**
```
占用率 = (活跃warp数) / (SM最大warp数)
```

**影响占用率的因素：**

1. **每个Block的寄存器使用量**
   - 每个SM的寄存器数量有限（例如A100有65536个32位寄存器）
   - 每个线程使用的寄存器越多，能同时驻留的线程就越少

2. **每个Block的共享内存使用量**
   - 每个SM的共享内存数量有限（例如A100有164KB/SM）
   - Block使用的共享内存越多，能同时驻留的Block就越少

3. **Block的大小（线程数）**
   - 必须是warp大小的倍数（32）
   - 太小会浪费调度资源，太大可能导致占用率低

**占用率的重要性：**

**为什么需要高占用率？**
- 隐藏内存延迟：当一个warp等待内存时，SM可以调度其他warp执行
- 提高计算资源利用率：更多活跃warp意味着更多指令级并行

**占用率不是越高越好：**
- 太高的占用率可能意味着每个线程的寄存器太少，导致溢出到本地内存
- 太低的占用率无法有效隐藏延迟
- 通常理想范围：30-70%（视具体应用而定）

**计算示例：**
假设GPU：
- 每个SM最多64个warp（2048线程）
- 每个SM有65536个寄存器
- 每个SM有48KB共享内存

如果内核：
- Block大小 = 128线程
- 每个线程使用32个寄存器
- 每个Block使用8KB共享内存

计算：
```
每个Block的寄存器 = 128 × 32 = 4096
寄存器限制的Blocks/SM = 65536 / 4096 = 16

共享内存限制的Blocks/SM = 48KB / 8KB = 6

实际最大Blocks/SM = min(16, 6) = 6
活跃warp数 = 6 × (128/32) = 24
占用率 = 24 / 64 = 37.5%
```

**调整占用率的方法：**

1. **调整Block大小**
```cpp
// 尝试不同的Block大小
dim3 block1(64);   // 小Block，可能更高占用率
dim3 block2(256);  // 大Block，可能更好的指令吞吐量
dim3 block3(1024); // 最大Block
```

2. **限制寄存器使用**
```cpp
// 使用编译指令限制每个线程的寄存器数量
__launch_bounds__(256, 8)  // Block大小256，每个SM至少8个Block
__global__ void kernel(...) {
    // ...
}
```

3. **调整共享内存使用**
- 减少共享内存使用量
- 或者使用更多共享内存换取更少的全局内存访问

**查看占用率的工具：**
- Nsight Compute：查看"Occupancy"指标
- CUDA Occupancy Calculator：官方提供的Excel表格
- nvprof：`--metrics achieved_occupancy`

**面试关键点：**
- 占用率是活跃warp与最大warp的比值
- 主要受寄存器和共享内存限制
- 高占用率有助于隐藏延迟但不是唯一目标
- 需要平衡占用率和其他优化手段

---

### Q8: 什么是分支发散（Branch Divergence）？如何处理？

**答案：**

**定义：**
分支发散是指同一个warp内的不同线程执行不同的代码路径。由于SIMT执行模型，warp内的线程必须执行相同的指令，当发生分支发散时，硬件会串行执行所有路径，对不需要执行的线程进行掩码（mask off），导致性能下降。

**分支发散的性能影响：**
```
执行时间 = Σ(每个分支路径的执行时间)
```
最坏情况：32个线程每个走不同路径 → 性能降到1/32

**示例：**

**有分支发散的代码：**
```cpp
__global__ void divergent_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ❌ 分支发散：同一个warp内线程走不同路径
    if (idx % 2 == 0) {
        data[idx] = compute1(data[idx]);
    } else {
        data[idx] = compute2(data[idx]);
    }
}
```

**无分支发散的代码：**
```cpp
__global__ void uniform_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // ✅ warp内线程执行相同路径
    if (tid < 16) {  // 前16个线程执行compute1
        data[idx] = compute1(data[idx]);
    } else {         // 后16个线程执行compute2
        data[idx] = compute2(data[idx]);
    }
}
```

**处理分支发散的方法：**

**方法1：重新组织数据/线程映射**
```cpp
// ❌ 按条件分散线程
if (condition[idx]) { pathA(); }
else                { pathB(); }

// ✅ 按条件分组线程
int mask = __ballot_sync(0xffffffff, condition[idx]);
int laneA = __popc(mask & ((1 << threadIdx.x) - 1));
int laneB = threadIdx.x - __popc(mask & ((1 << threadIdx.x) - 1));

if (threadIdx.x < __popc(mask)) {
    int original_idx = find_nth_set_bit(mask, threadIdx.x);
    pathA(original_idx);
} else {
    int original_idx = find_nth_clear_bit(mask, threadIdx.x - __popc(mask));
    pathB(original_idx);
}
```

**方法2：使用分支内建函数**
```cpp
// 使用predicate指令减少发散影响
float val = data[idx];
float resultA = compute1(val);
float resultB = compute2(val);
data[idx] = condition ? resultA : resultB;  // 两个都算，最后选择
```

**方法3：利用warp内建函数**
```cpp
// warp内归约，避免分支
__global__ void warp_reduce(float* data, int n) {
    int idx = blockIdx.x * 64 + threadIdx.x;
    float val = 0.0f;
    
    if (idx < n) {
        val = data[idx];
        
        // warp内归约，无分支发散
        for (int i = 16; i > 0; i >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, i);
        }
    }
    
    if (threadIdx.x == 0) {
        data[blockIdx.x] = val;
    }
}
```

**方法4：使用模板和编译期分支**
```cpp
template <bool USE_PATH_A>
__device__ float compute_path(float val) {
    if (USE_PATH_A) {
        return compute1(val);  // 编译期确定，无运行时分支
    } else {
        return compute2(val);
    }
}
```

**常见的分支发散场景：**

1. **边界检查**
```cpp
// 通常边界检查只在Grid的最后一个Block有发散，影响较小
if (idx < n) {
    // ...
}
```

2. **条件赋值**
```cpp
// 使用三元运算符或min/max代替if-else
data[idx] = (val > threshold) ? max_val : val;
```

3. **循环次数不同**
```cpp
// 尽量让warp内循环次数相同
int iterations = min(max_iter, n - idx);
for (int i = 0; i < iterations; i++) {
    // ...
}
```

**测量分支发散：**
- Nsight Compute：查看"Branch Statistics"
- 关键指标：Branch Efficiency（分支效率）
- 理想值：接近100%

**面试关键点：**
- 分支发散是warp内线程执行不同路径
- SIMT模型导致串行执行，性能下降
- 处理方法：重新组织线程、使用内建函数、模板分支
- 边界检查的发散通常影响不大

---

### Q9: 常用的CUDA性能分析工具有哪些？

**答案：**

**1. Nsight Systems（系统级分析）**

**功能：**
- 可视化整个应用的执行 timeline
- 显示CPU和GPU的活动
- 显示CUDA API调用、内核执行、内存传输
- 分析CPU/GPU的空闲时间和重叠情况

**使用场景：**
- 找出应用的整体瓶颈（CPU还是GPU）
- 分析多流执行的重叠情况
- 识别不必要的同步点
- 查看内存传输的效率

**常用命令：**
```bash
nsys profile --stats=true ./your_app
```

**关键指标：**
- GPU利用率
- 内核执行时间分布
- 内存传输时间
- CUDA API调用耗时

---

**2. Nsight Compute（内核级分析）**

**功能：**
- 详细分析单个内核的执行
- 提供硬件级别的性能计数器
- 识别内存访问模式、分支效率、占用率等
- 提供源代码级别的性能标注

**使用场景：**
- 内核级性能调优
- 分析内存访问模式（合并度、Bank冲突）
- 检查分支发散情况
- 分析指令吞吐量

**常用命令：**
```bash
ncu --set full -o profile_output ./your_app
ncu --section MemoryWorkloadAnalysis ./your_app
```

**关键分析指标：**

**内存相关：**
- `gld_transactions`：全局加载事务数
- `gst_transactions`：全局存储事务数
- `shared_load_transactions`：共享内存加载事务数
- `shared_store_transactions`：共享内存存储事务数
- `shared_efficiency`：共享内存效率
- `dram_read_throughput`：DRAM读取吞吐量
- `dram_write_throughput`：DRAM写入吞吐量

**计算相关：**
- `sm_efficiency`：SM效率
- `achieved_occupancy`：实际占用率
- `inst_per_warp`：每个warp的指令数
- `branch_efficiency`：分支效率

**计算 Roofline：**
- 计算强度（FLOP/Byte）
- 性能瓶颈是计算还是内存

---

**3. nvprof（旧版分析工具）**

**功能：**
- 命令行性能分析工具
- 提供内核和内存操作的汇总信息

**常用命令：**
```bash
# 基本分析
nvprof ./your_app

# 查看特定指标
nvprof --metrics achieved_occupancy,gld_efficiency ./your_app

# 查看所有指标
nvprof --query-events
nvprof --query-metrics
```

**注意：**nvprof在较新的CUDA版本中已被Nsight系列取代。

---

**4. CUDA-GDB（调试工具）**

**功能：**
- GPU程序的源代码级调试
- 支持检查设备内存
- 支持条件断点
- 支持检查特定线程的状态

**常用命令：**
```bash
cuda-gdb ./your_app
```

**常用调试命令：**
```
break kernel_name       # 在内核入口设置断点
info cuda threads       # 查看所有CUDA线程
thread 1.0.1.128        # 切换到特定线程
print variable          # 打印变量值
cuda thread <threadIdx> # 切换到特定线程
```

---

**性能分析流程：**

**第一步：整体分析（Nsight Systems）**
1. 查看GPU利用率是否足够高
2. 查看内存传输是否有优化空间
3. 查看是否有CPU/GPU空闲时间
4. 识别耗时最长的内核

**第二步：内核分析（Nsight Compute）**
1. 分析最耗时的几个内核
2. 检查内存带宽利用率
3. 检查占用率是否合理
4. 检查分支效率
5. 检查指令吞吐量

**第三步：针对性优化**
1. 根据分析结果确定优化方向
2. 实施优化
3. 重新profile验证效果

---

**常见性能问题的指标表现：**

| 问题 | 指标表现 |
|------|---------|
| 非合并访问 | `gld_transactions` 远高于理论值，`gld_efficiency` 低 |
| Bank冲突 | `shared_efficiency` 低，`shared_load_transactions` 高 |
| 分支发散 | `branch_efficiency` 低于90% |
| 低占用率 | `achieved_occupancy` 低于30% |
| 内存瓶颈 | DRAM带宽接近峰值，SM效率低 |
| 计算瓶颈 | SM效率高，DRAM带宽低 |

---

## 四、并行算法

### Q10: 如何实现高效的规约（Reduction）算法？有哪些优化？

**答案：**

**规约算法定义：**
将一个数组的所有元素通过某种运算（求和、最大值、最小值等）合并成单个值的操作。

**优化迭代过程：**

---

**版本1：最基础的规约（有问题）**
```cpp
__global__ void reduce_v1(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    
    // ❌ 问题：严重的Bank冲突
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```
**问题：**
- Bank冲突：sdata[tid]和sdata[tid+s]在同一个Bank
- 线程束发散严重

---

**版本2：解决Bank冲突**
```cpp
__global__ void reduce_v2(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    
    // ✅ 顺序颠倒，从大到小，无Bank冲突
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```
**改进：**
- Bank冲突消除
- 但仍有warp内的分支发散

---

**版本3：warp内展开，减少同步**
```cpp
__device__ void warp_reduce(volatile float* sdata, int tid) {
    // warp内不需要同步，使用volatile防止编译器优化
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_v3(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    
    // 从512开始，到64结束
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // warp内归约
    if (tid < 32) {
        warp_reduce(sdata, tid);
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```
**改进：**
- warp内不需要同步
- 减少了同步次数
- 使用volatile确保写入可见

---

**版本4：循环展开，增加ILP**
```cpp
__global__ void reduce_v4(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    // ✅ 每个线程加载多个元素，增加ILP
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    if (i < n) sum += g_idata[i];
    if (i + blockDim.x < n) sum += g_idata[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // 剩下的归约和v3一样
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid < 32) warp_reduce(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```
**改进：**
- 每个线程处理2个元素
- 增加指令级并行（ILP）
- 减少Block数量

---

**版本5：完全展开+模板**
```cpp
template <unsigned int blockSize>
__device__ void reduce_unroll(volatile float* sdata, int tid) {
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[tid + 64]; }
    if (tid < 32) warp_reduce(sdata, tid);
}

template <unsigned int blockSize>
__global__ void reduce_v5(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[blockSize];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 4) + threadIdx.x;
    
    // ✅ 每个线程加载4个元素
    float sum = 0.0f;
    if (i < n) sum += g_idata[i];
    if (i + blockSize < n) sum += g_idata[i + blockSize];
    if (i + 2 * blockSize < n) sum += g_idata[i + 2 * blockSize];
    if (i + 3 * blockSize < n) sum += g_idata[i + 3 * blockSize];
    sdata[tid] = sum;
    __syncthreads();
    
    reduce_unroll<blockSize>(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// 显式实例化
template __global__ void reduce_v5<1024>(float*, float*, int);
template __global__ void reduce_v5<512>(float*, float*, int);
```
**改进：**
- 模板参数实现编译期展开
- 每个线程处理更多元素
- 减少循环开销

---

**规约优化总结：**

| 优化技术 | 效果 |
|---------|------|
| 共享内存 | 减少全局内存访问 |
| 倒序归约 | 消除Bank冲突 |
| Warp内展开 | 减少同步，消除warp内分支 |
| 多元素加载 | 增加ILP，减少Block数量 |
| 循环展开 | 减少循环开销 |
| 模板元编程 | 编译期优化 |

**性能提升估算：**
- 基础版 → 消除Bank冲突：~2x
- → Warp展开：~1.5x
- → 多元素加载：~1.5-2x
- → 完全展开：~1.2x
- 总提升：可达10倍以上

---

## 五、高级特性

### Q11: CUDA流（Stream）的作用是什么？如何使用？

**答案：**

**定义：**
CUDA流是一系列按顺序执行的CUDA操作（内核执行、内存传输等）。不同流中的操作可以并行执行，同一流中的操作严格按提交顺序执行。

**流的作用：**

1. **实现并发执行**
   - 内核与内核并行
   - 内存传输与内核并行
   - 内存传输与内存传输并行（H2D和D2H可并行）

2. **隐藏数据传输延迟**
   - 将数据分块，一块计算时传输下一块
   - 计算和数据传输重叠执行

3. **细粒度的同步控制**
   - 可以在流级别进行同步
   - 不需要同步整个设备

---

**基本使用方法：**

```cpp
// 1. 创建流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 2. 在流中提交操作（注意使用Async版本）
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_data);
cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream);

// 3. 等待流完成
cudaStreamSynchronize(stream);

// 4. 销毁流
cudaStreamDestroy(stream);
```

---

**多流流水线示例：**

```cpp
const int NUM_STREAMS = 4;
cudaStream_t streams[NUM_STREAMS];
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
}

// 将数据分成4块
int chunk_size = n / NUM_STREAMS;

for (int i = 0; i < NUM_STREAMS; i++) {
    int offset = i * chunk_size;
    
    // ✅ 每个流独立处理一块数据
    cudaMemcpyAsync(d_data + offset, h_data + offset, 
                     chunk_size * sizeof(float),
                     cudaMemcpyHostToDevice, streams[i]);
    
    kernel<<<grid, block, 0, streams[i]>>>(d_data + offset, chunk_size);
    
    cudaMemcpyAsync(h_result + offset, d_result + offset,
                     chunk_size * sizeof(float),
                     cudaMemcpyDeviceToHost, streams[i]);
}

// 等待所有流完成
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
}
```

---

**深度流水（计算与传输完全重叠）：**

```cpp
// 假设有N个批次需要处理
int batches = 10;
int stream_count = 3;  // 深度为3的流水线

// 时间线：
// 流0: [H2D] [KERNEL] [D2H]
// 流1:       [H2D] [KERNEL] [D2H]
// 流2:             [H2D] [KERNEL] [D2H]

for (int batch = 0; batch < batches + stream_count - 1; batch++) {
    for (int s = 0; s < stream_count; s++) {
        int compute_batch = batch - s;
        
        if (compute_batch >= 0 && compute_batch < batches) {
            if (s == 0) {
                // 流0：H2D
                cudaMemcpyAsync(d_data[s], h_data[compute_batch], 
                               size, H2D, streams[s]);
            } else if (s == 1) {
                // 流1：Kernel
                kernel<<<grid, block, 0, streams[s]>>>(d_data[s-1], d_result[s]);
            } else {
                // 流2：D2H
                cudaMemcpyAsync(h_result[compute_batch], d_result[s-1],
                               size, D2H, streams[s]);
            }
        }
    }
}
```

---

**流的同步机制：**

**1. 流内隐式同步**
- 同一流的操作按提交顺序执行
- 前一个操作完成后才执行下一个

**2. 流同步**
```cpp
cudaStreamSynchronize(stream);  // 等待流中所有操作完成
```

**3. 事件同步**
```cpp
cudaEvent_t event;
cudaEventCreate(&event);

kernel1<<<grid, block, 0, stream1>>>();
cudaEventRecord(event, stream1);  // 记录事件

// stream2等待event完成后才继续
cudaStreamWaitEvent(stream2, event, 0);
kernel2<<<grid, block, 0, stream2>>>();
```

**4. 回调函数**
```cpp
void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void* data) {
    printf("Stream completed!\n");
}

kernel<<<grid, block, 0, stream>>>();
cudaStreamAddCallback(stream, my_callback, nullptr, 0);
```

---

**流的最佳实践：**

**1. 选择合适的流数量**
- 太少：无法充分利用并行性
- 太多：增加调度开销
- 通常建议：4-16个流（视GPU而定）

**2. 使用默认流（nullptr）要小心**
- 默认流与其他流隐式同步
- 建议显式创建流

**3. 固定主机内存**
```cpp
// 使用cudaMallocHost分配固定内存，支持异步传输
cudaMallocHost(&h_data, size);  // 页锁定内存，支持异步传输
```

**4. 数据分块大小**
- 太小：开销大
- 太大：无法有效重叠
- 建议：从1MB开始测试，找到最佳点

---

**流的性能收益：**

| 场景 | 潜在收益 |
|------|---------|
| 计算密集型 | 较小（10-20%） |
| 内存传输密集型 | 较大（50-200%） |
| 小内核多批次 | 很大（200%+） |

---

### Q12: 什么是动态并行（Dynamic Parallelism）？适用场景？

**答案：**

**定义：**
动态并行是CUDA的一项高级特性，允许在设备端（核函数内）创建新的核函数、分配设备内存、创建流等，而不需要CPU参与。

**支持的GPU：**
- 计算能力3.5及以上（Kepler及以后）

---

**基本使用：**

```cpp
__global__ void child_kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

__global__ void parent_kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        // 在父内核中启动子内核
        child_kernel<<<4, 64>>>(data, n);
        // 可以等待子内核完成
        cudaDeviceSynchronize();
    }
}
```

---

**适用场景：**

**1. 递归算法**
```cpp
// 快速排序 - 动态并行版本
__device__ void partition(int* data, int left, int right, int* pivot) {
    // ... 分区逻辑 ...
}

__global__ void quicksort(int* data, int left, int right) {
    if (left >= right) return;
    
    int pivot;
    partition(data, left, right, &pivot);
    
    // 递归启动两个子内核处理左右两部分
    if (right - left > 1024) {  // 超过阈值使用动态并行
        quicksort<<<1, 1>>>(data, left, pivot - 1);
        quicksort<<<1, 1>>>(data, pivot + 1, right);
    } else {
        // 小数据量，当前线程串行处理
        quicksort_serial(data, left, right);
    }
}
```

**2. 自适应算法**
```cpp
__global__ void adaptive_process(float* data, int n) {
    int idx = blockIdx.x;
    
    // 根据数据复杂度决定并行度
    float complexity = analyze_data(data + idx * BLOCK_SIZE);
    
    if (complexity > HIGH_THRESHOLD) {
        // 复杂任务，启动更多子线程
        process_detail<<<8, 64>>>(data + idx * BLOCK_SIZE);
    } else {
        // 简单任务，当前线程处理
        process_simple(data + idx * BLOCK_SIZE);
    }
}
```

**3. 树/图遍历**
```cpp
__global__ void traverse_node(Node* node) {
    // 处理当前节点
    process(node);
    
    // 为每个子节点启动新内核
    for (int i = 0; i < node->num_children; i++) {
        if (node->children[i]) {
            traverse_node<<<1, 64>>>(node->children[i]);
        }
    }
}
```

**4. 不规则并行任务**
```cpp
__global__ void process_tasks(Task* tasks, int num_tasks) {
    int idx = blockIdx.x;
    
    if (idx < num_tasks) {
        Task task = tasks[idx];
        
        // 根据任务类型动态分配资源
        if (task.type == LARGE_TASK) {
            // 大任务，用更多线程
            process_large<<<16, 128>>>(task);
        } else {
            // 小任务，少量线程
            process_small<<<1, 32>>>(task);
        }
    }
}
```

---

**动态并行的优缺点：**

**优点：**
1. **减少CPU-GPU通信**：不需要CPU调度子任务
2. **自适应并行**：根据运行时数据调整并行度
3. **简化递归算法**：自然表达递归并行
4. **更好的负载均衡**：细粒度任务分配

**缺点：**
1. **额外开销**：子内核启动有开销（比CPU启动快但仍有）
2. **资源限制**：子内核共享父内核的资源
3. **调试困难**：嵌套调用难以调试
4. **编程复杂**：需要考虑嵌套同步

---

**最佳实践：**

**1. 设置合理的粒度阈值**
```cpp
const int THRESHOLD = 1024;  // 低于此值不使用动态并行

if (problem_size > THRESHOLD) {
    child_kernel<<<...>>>(...);  // 大问题，并行
} else {
    serial_process(...);         // 小问题，串行
}
```

**2. 避免过度嵌套**
- 嵌套层级建议不超过3-4层
- 过深的嵌套会导致严重的开销

**3. 注意同步**
```cpp
// 等待子内核完成
cudaDeviceSynchronize();

// 或者使用事件
cudaEvent_t event;
cudaEventCreate(&event);
child_kernel<<<...>>>();
cudaEventRecord(event);
cudaEventSynchronize(event);
```

**4. 预分配内存**
- 避免在核函数内频繁分配内存
- 预先分配内存池

---

**性能考虑：**

| 操作 | 大致开销 |
|------|---------|
| 子内核启动 | ~5-10µs |
| 设备端同步 | ~1-2µs |
| 设备端内存分配 | 相对较高，尽量避免 |

**何时不应该使用动态并行：**
- 任务粒度太小（<1000个线程）
- 可以静态规划并行任务
- 严格的性能要求
- 简单的数据并行任务

---

## 六、系统与架构

### Q13: NVIDIA GPU架构的主要特点是什么？

**答案：**

**1. SIMT（单指令多线程）架构**

**特点：**
- 每个SM管理多个warp（32线程）
- warp内线程共享程序计数器
- 线程可以有独立的执行路径（分支发散）

**与SIMD的区别：**
- SIMD：向量寄存器，所有元素必须执行相同操作
- SIMT：每个线程有自己的寄存器，可以分支
- SIMT提供了更灵活的编程模型

---

**2. 内存层次结构**

```
┌─────────────────────────────────────────────────┐
│                  寄存器 (Registers)             │  最快，线程私有
│         (每个线程约255个，每个SM约64K个)        │
├─────────────────────────────────────────────────┤
│              共享内存 (Shared Memory)           │  快，Block共享
│              (每个SM约48-164KB)                 │
├─────────────────────────────────────────────────┤
│          L1缓存 / 常量缓存 / 纹理缓存           │
│              (每个SM约16-48KB)                  │
├─────────────────────────────────────────────────┤
│                  L2 缓存                        │  所有SM共享
│                (约4-80MB)                       │
├─────────────────────────────────────────────────┤
│              全局内存 (Global Memory)           │  最慢，最大
│                 (GB级)                          │
└─────────────────────────────────────────────────┘
```

---

**3. 各代架构的主要特点：**

**Fermi（费米）- 计算能力2.x**
- 首个完整的GPU计算架构
- 32个Bank的共享内存
- L1/L2缓存层次

**Kepler（开普勒）- 计算能力3.x**
- 动态并行
- Hyper-Q（多个CPU内核同时提交任务）
- Grid管理单元

**Maxwell（麦克斯韦）- 计算能力5.x**
- 更高的能效比
- 共享内存原子操作性能提升
- 统一内存

**Pascal（帕斯卡）- 计算能力6.x**
- NVLink高速互连
- 16位浮点数（FP16）性能提升
- 统一内存改进

**Volta（伏特）- 计算能力7.x**
- Tensor Core（张量核心）
- 独立线程调度（减少分支发散影响）
- CUDA Graph

**Turing（图灵）- 计算能力7.5**
- RT Core（光线追踪核心）
- Tensor Core改进
- 整数/浮点并发执行

**Ampere（安培）- 计算能力8.x**
- 稀疏Tensor Core
- 第三代Tensor Core
- 多实例GPU（MIG）
- BF16支持

**Ada Lovelace（阿达）- 计算能力8.9**
- 第四代Tensor Core
- 第三代RT Core
- FP8支持

**Hopper（霍珀）- 计算能力9.0**
- Transformer Engine
- DPX指令
- 更好的多GPU协作

---

**4. SM（流式多处理器）架构**

**SM的主要组件：**
- CUDA核心（整数和浮点单元）
- 特殊功能单元（SFU：超越函数等）
- 加载/存储单元（LD/ST）
- 寄存器文件
- 共享内存/L1缓存
- Warp调度器
- 指令缓存

**执行流程：**
1. Warp调度器选择就绪的warp
2. 发射指令到执行单元
3. 内存访问的延迟通过切换warp来隐藏

---

**5. 关键硬件指标：**

**计算能力**
- 表示GPU架构的功能集
- 决定支持哪些CUDA特性
- 如：7.0表示Volta架构

**内存带宽**
- DRAM接口宽度 × 数据率 / 8
- 例如：384-bit × 2.4Gbps / 8 = 1152 GB/s

**计算吞吐量**
- CUDA核心数 × 频率 × 操作数/周期
- Tensor Core有专门的吞吐量计算

**占用率相关限制**
- 每个SM的最大线程数（如2048）
- 每个SM的最大warp数（如64）
- 每个SM的最大Block数（如32）

---

### Q14: 统一内存（Unified Memory）是什么？有什么优缺点？

**答案：**

**定义：**
统一内存是CUDA 6.0引入的特性，提供一个单一的内存地址空间，CPU和GPU都可以通过相同的指针访问。系统自动在CPU和GPU之间迁移数据。

---

**基本使用：**

```cpp
// 传统方式 - 显式管理
float* h_data;
float* d_data;
cudaMallocHost(&h_data, size);  // 主机内存
cudaMalloc(&d_data, size);      // 设备内存
cudaMemcpy(d_data, h_data, size, H2D);  // 显式传输
kernel<<<grid, block>>>(d_data);
cudaMemcpy(h_data, d_data, size, D2H);

// 统一内存方式
float* data;
cudaMallocManaged(&data, size);  // 统一内存
kernel<<<grid, block>>>(data);   // GPU直接访问
cudaDeviceSynchronize();
// CPU可以直接访问data，不需要显式拷贝
```

---

**统一内存的工作原理：**

**页错误机制：**
1. 分配统一内存时，物理内存可能在CPU或GPU
2. 当CPU/GPU访问不在本地的内存页时，触发页错误
3. 驱动自动将数据页迁移到访问的设备

**内存迁移粒度：**
- 以页为单位（通常4KB或64KB）
- 按需迁移，只迁移需要的页

---

**优点：**

1. **简化编程**
   - 消除显式的内存拷贝代码
   - 减少出错概率
   - 更接近传统的CPU编程模型

2. **复杂数据结构**
   - 自然支持指针、链表、树等结构
   - 不需要手动序列化/反序列化

```cpp
// 复杂数据结构的统一内存
struct Node {
    float value;
    Node* left;
    Node* right;  // 指针在统一内存中可以直接使用
};

Node* root;
cudaMallocManaged(&root, sizeof(Node));
// 不需要手动设置指针偏移
```

3. **数据按需迁移**
   - 只迁移实际访问的数据
   - 某些场景下比手动拷贝更高效

4. **多GPU场景**
   - 数据可以在多个GPU间自动迁移
   - 简化多GPU编程

---

**缺点：**

1. **性能开销**
   - 页错误处理有开销
   - 按需迁移可能导致多次小传输
   - 通常比优化好的显式传输慢10-30%

2. **缺乏控制**
   - 程序员无法精确控制迁移时机
   - 可能在关键路径上发生迁移

3. **内存开销**
   - 系统需要维护页表
   - 某些情况下数据可能在两端都有副本

4. **调试困难**
   - 数据迁移是透明的
   - 难以分析性能问题的原因

---

**最佳实践：**

**1. 预取数据（Prefetching）**
```cpp
cudaMallocManaged(&data, size);

// 预取到GPU
cudaMemPrefetchAsync(data, size, device_id, stream);

kernel<<<grid, block, 0, stream>>>(data);

// 预取回CPU
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId, stream);
```

**2. 建议内存位置**
```cpp
// 建议分配在CPU内存
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

// 建议分配在GPU内存
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, device_id);

// 标记为只读，可以在多GPU间复制
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, device_id);
```

**3. 合理使用场景**
```cpp
// ✅ 适合：原型开发、复杂数据结构、低性能要求
cudaMallocManaged(&data, size);
process(data);  // 快速开发

// ❌ 不适合：高性能计算、已知访问模式
// 使用显式内存管理获得最佳性能
cudaMalloc(&d_data, size);
cudaMemcpyAsync(d_data, h_data, size, H2D, stream);
kernel<<<grid, block, 0, stream>>>(d_data);
```

**4. 监控页错误**
```bash
# 使用Nsight Systems查看统一内存活动
nsys profile --trace=cuda,unified_memory ./app

# 查看页错误统计
ncu --metrics unified_memory.* ./app
```

---

**性能比较：**

| 场景 | 统一内存 | 显式管理 |
|------|---------|----------|
| 简单应用，开发速度优先 | ✅ 好 | ❌ 开发慢 |
| 复杂数据结构（链表、树） | ✅ 很适合 | ❌ 需要序列化 |
| 不规则访问模式 | ✅ 可能更好 | ⚠️ 难优化 |
| 规则大块数据 | ⚠️ 慢10-30% | ✅ 最佳性能 |
| 低延迟要求 | ⚠️ 不确定 | ✅ 可预测 |

---

**面试关键点：**
- 统一内存提供单一地址空间
- 通过页错误和数据迁移实现
- 简化编程但有性能开销
- 可以通过预取和建议来优化
- 适合原型开发和复杂数据结构

---

## 七、编程实践

### Q15: 如何调试CUDA程序？

**答案：**

**CUDA调试工具和方法：**

---

**1. CUDA错误检查（必须！）**

**宏定义：**
```cpp
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_LAST_KERNEL() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Kernel launch error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
```

**使用示例：**
```cpp
CHECK_CUDA(cudaMalloc(&d_data, size));
kernel<<<grid, block>>>(d_data);
CHECK_LAST_KERNEL();  // 检查内核启动错误
CHECK_CUDA(cudaDeviceSynchronize());  // 检查内核执行错误
```

---

**2. CUDA-GDB调试**

**编译调试版本：**
```bash
nvcc -g -G -o debug_app source.cu  # -G生成设备调试信息
```

**常用命令：**
```bash
# 启动调试
cuda-gdb ./debug_app

# 设置断点
break kernel_name               # 内核函数入口
break file.cu:123              # 指定文件行号
break kernel_name if threadIdx.x == 0 && blockIdx.x == 0  # 条件断点

# 线程相关
info cuda threads               # 列出所有CUDA线程
info cuda blocks                # 列出所有活动的Block
cuda thread (0,0,0),(0,0,0)    # 切换到特定线程(blockIdx, threadIdx)

# 查看数据
print d_data[0]                 # 打印设备内存
print *d_data@10                # 打印数组前10个元素
x/32f d_data                    # 以浮点数格式查看32个元素

# 检查共享内存
print sdata                     # 共享内存变量

# 继续执行
continue
next                            # 单步执行（跳过函数）
step                            # 单步执行（进入函数）
finish                          # 执行完当前函数
```

---

**3. 内核中的printf**

```cpp
__global__ void debug_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 只打印特定线程，避免输出过多
    if (idx < 10 || (threadIdx.x == 0 && blockIdx.x < 5)) {
        printf("Block %d, Thread %d, idx %d, data = %f\n",
               blockIdx.x, threadIdx.x, idx, data[idx]);
    }
}
```

**注意事项：**
- 输出缓冲区有限（约1MB），过多printf会丢失信息
- 输出顺序不确定（并行执行）
- 仅用于调试，发布版本要移除

---

**4. 内存检查工具**

**cuda-memcheck：**
```bash
# 基本内存检查
cuda-memcheck ./app

# 检查越界访问
cuda-memcheck --tool memcheck ./app

# 检查未初始化变量
cuda-memcheck --tool initcheck ./app

# 检查共享内存Bank冲突
cuda-memcheck --tool racecheck ./app
```

**常见内存错误：**
- 越界访问（访问超过分配的内存）
- 访问已释放的内存
- 未初始化的内存读取
- 共享内存竞态条件

---

**5. 二分法调试**

```cpp
__global__ void buggy_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 步骤1：注释所有代码，只保留简单操作
    // data[idx] = 1.0f;  // 如果还崩溃，可能是启动配置问题
    
    // 步骤2：逐步添加代码
    float val = data[idx];
    
    // 步骤3：在可疑位置添加检查
    if (idx < n) {  // 确保边界检查
        // ...
    }
}
```

---

**6. 常用调试技巧**

**简化问题：**
```cpp
// 用小数据集测试
int test_n = 128;  // 而不是1000000

// 用简单的Block/Grid配置
dim3 block(32);    // 小Block，容易调试
dim3 grid(4);
```

**验证计算结果：**
```cpp
// CPU版本作为参考
void cpu_reference(float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = compute(input[i]);
    }
}

// 比较GPU和CPU结果
void verify(float* gpu_result, float* cpu_result, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > 1e-5) {
            printf("Mismatch at %d: GPU=%f, CPU=%f\n",
                   i, gpu_result[i], cpu_result[i]);
            break;
        }
    }
}
```

**使用可视化工具：**
```cpp
// 将中间结果保存下来可视化
__global__ void save_intermediate(float* data, float* intermediate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = step1(data[idx]);
    intermediate[idx] = val;  // 保存中间结果
    
    val = step2(val);
    data[idx] = val;
}

// CPU端查看或可视化中间数据
```

---

**常见错误类型及排查：**

| 错误现象 | 可能原因 | 排查方法 |
|---------|---------|---------|
| 内核启动就崩溃 | 启动配置错误、设备内存不足 | 检查grid/block大小，检查内存分配 |
| 结果全为0或NaN | 未初始化内存、越界访问、除零 | cuda-memcheck，边界检查 |
| 结果部分正确 | 同步问题、边界条件、Bank冲突 | 检查`__syncthreads()`，检查边界条件 |
| 程序挂起 | 死锁、无限循环、缺少同步 | 中断程序，查看栈回溯 |
| 性能异常低 | 分支发散、非合并访问、低占用率 | Nsight Compute分析 |

---

### Q16: CUDA编程中有哪些常见的陷阱？

**答案：**

---

**陷阱1: 忘记检查错误**

**问题：**
```cpp
// ❌ 没有错误检查
cudaMalloc(&d_data, size);
kernel<<<grid, block>>>(d_data);
// 出错也不知道
```

**解决：**
```cpp
// ✅ 始终检查错误
CHECK_CUDA(cudaMalloc(&d_data, size));
kernel<<<grid, block>>>(d_data);
CHECK_LAST_KERNEL();
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());
```

---

**陷阱2: 线程/Block索引计算错误**

**问题：**
```cpp
__global__ void bad_index(float* data, int n) {
    // ❌ 用了blockIdx而不是blockIdx.x
    int idx = blockIdx * blockDim.x + threadIdx.x;  // 编译错误
    
    // ❌ 搞反了x和y维度
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // 可能错误
    
    // ❌ 缺少边界检查
    data[idx] = ...;  // idx可能超过n
}
```

**解决：**
```cpp
__global__ void good_index(float* data, int n) {
    // ✅ 正确的索引计算
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ✅ 必须的边界检查
    if (idx < n) {
        data[idx] = ...;
    }
}
```

---

**陷阱3: 共享内存同步问题**

**问题1: 忘记同步**
```cpp
__global__ void no_sync(float* input) {
    __shared__ float sdata[128];
    sdata[threadIdx.x] = input[threadIdx.x];
    // ❌ 缺少__syncthreads()
    float val = sdata[127 - threadIdx.x];  // 可能读取未写入的值
}
```

**问题2: 分支内同步**
```cpp
__global__ void branch_sync(float* input) {
    __shared__ float sdata[128];
    
    if (threadIdx.x < 64) {
        sdata[threadIdx