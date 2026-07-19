# 🎯 CUDA 图像卷积端到端性能优化案例

> 场景：自动驾驶实时图像预处理系统，GPU 性能无法满足实时性要求
>
> **面试讲故事神器！按这个流程走一遍，面试官直接跪了**

---

## 📋 故事背景

**时间**：自动驾驶量产前 1 个月
**系统**：车载图像预处理模块
**现象**：
- 输入：1920x1080 RGB 图像，30fps
- 要求：单帧处理延迟 < 5ms，才能保证 30fps 实时输出
- 现状：朴素 CUDA 实现单帧处理需要 45ms，只能达到 22fps
- GPU 使用率只有 35%，内存带宽利用率低

**你作为 CUDA 性能优化工程师，负责解决这个问题**

---

## 🎬 故障复现步骤

### Step 1: 编译并运行基线版本
```bash
cd v1_bad
mkdir build && cd build
cmake ..
make
./naive_convolution
```

### Step 2: 观察性能指标
你会看到：
- CPU 参考实现：约 800ms/帧
- 朴素 CUDA 实现：约 45ms/帧
- 性能提升：约 18 倍
- 但是：离目标 5ms 还差 9 倍！

### Step 3: 使用 Nsight 分析（可选）
```bash
nsys profile ./naive_convolution
```

你会发现：
- 全局内存访问效率低（< 50%）
- 大量重复的全局内存读取
- 没有利用共享内存和常量内存
- 没有使用流进行重叠执行

---

## 🔍 优化过程（面试就按这个讲）

### 📍 优化 1: 朴素实现 - v1_bad

**问题现象**：
```
处理 1920x1080 图像:
  CPU: 800ms
  GPU: 45ms
  加速比: 18x
  内存带宽利用率: 42%
  计算利用率: 35%
```

**主要问题**：
1. ❌ 每个像素都从全局内存读取整个卷积核
2. ❌ 相邻像素重复读取相同的图像数据
3. ❌ 全局内存访问模式不连续，有 bank conflict
4. ❌ 没有利用数据局部性

**代码问题点**：
```cpp
__global__ void naive_convolution_kernel(float* input, float* output, 
                                         float* kernel, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
            for (int l = -KERNEL_RADIUS; l <= KERNEL_RADIUS; l++) {
                int input_row = row + k;
                int input_col = col + l;
                if (input_row >= 0 && input_row < height && 
                    input_col >= 0 && input_col < width) {
                    // ❌ 每次都从全局内存读 kernel
                    // ❌ 相邻线程重复读取相同的 input
                    sum += input[input_row * width + input_col] * 
                           kernel[(k + KERNEL_RADIUS) * KERNEL_SIZE + 
                                  (l + KERNEL_RADIUS)];
                }
            }
        }
        output[row * width + col] = sum;
    }
}
```

---

### 📍 优化 2: 共享内存 - v2_good

**优化思路**：
- 每个线程块将需要的图像数据加载到共享内存
- 块内所有线程共享这些数据，减少全局内存访问

**关键改动**：
```cpp
__global__ void shared_memory_convolution_kernel(float* input, float* output,
                                                  float* kernel, int width, int height) {
    __shared__ float shared_tile[TILE_SIZE + 2 * KERNEL_RADIUS]
                              [TILE_SIZE + 2 * KERNEL_RADIUS];
    
    // 1. 协作加载 tile 到共享内存
    int tile_row = threadIdx.y + KERNEL_RADIUS;
    int tile_col = threadIdx.x + KERNEL_RADIUS;
    int global_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int global_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // 加载中心区域
    if (global_row < height && global_col < width) {
        shared_tile[tile_row][tile_col] = input[global_row * width + global_col];
    } else {
        shared_tile[tile_row][tile_col] = 0.0f;
    }
    
    // 2. 同步，确保所有数据都加载完成
    __syncthreads();
    
    // 3. 所有线程从共享内存读取，而不是全局内存
    if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
        float sum = 0.0f;
        for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
            for (int l = -KERNEL_RADIUS; l <= KERNEL_RADIUS; l++) {
                // ✅ 从共享内存读取，快 100+ 倍
                sum += shared_tile[tile_row + k][tile_col + l] * 
                       kernel[(k + KERNEL_RADIUS) * KERNEL_SIZE + 
                              (l + KERNEL_RADIUS)];
            }
        }
        
        int output_row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int output_col = blockIdx.x * TILE_SIZE + threadIdx.x;
        if (output_row < height && output_col < width) {
            output[output_row * width + output_col] = sum;
        }
    }
}
```

**性能提升**：
- 处理时间：45ms → 18ms，**2.5 倍提升**
- 内存带宽利用率：42% → 78%
- 全局内存访问减少：约 90%

---

### 📍 优化 3: 常量内存 - v3_better

**优化思路**：
- 卷积核大小固定（5x5 = 25 个元素），适合放在常量内存
- 常量内存有缓存，所有 SM 共享，带宽更高
- 内核参数是只读的，所有线程访问相同数据

**关键改动**：
```cpp
// ✅ 卷积核声明为常量内存
__constant__ float d_const_kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void constant_memory_convolution_kernel(float* input, float* output,
                                                    int width, int height) {
    __shared__ float shared_tile[TILE_SIZE + 2 * KERNEL_RADIUS]
                              [TILE_SIZE + 2 * KERNEL_RADIUS];
    
    // ... 共享内存加载代码不变 ...
    
    __syncthreads();
    
    if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
        float sum = 0.0f;
        for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
            for (int l = -KERNEL_RADIUS; l <= KERNEL_RADIUS; l++) {
                // ✅ 从常量内存读取，有缓存
                sum += shared_tile[tile_row + k][tile_col + l] * 
                       d_const_kernel[(k + KERNEL_RADIUS) * KERNEL_SIZE + 
                                      (l + KERNEL_RADIUS)];
            }
        }
        // ...
    }
}

// Host 端：
// ✅ 将卷积核复制到常量内存
CUDA_CHECK(cudaMemcpyToSymbol(d_const_kernel, h_kernel, 
                               kernel_size * kernel_size * sizeof(float)));
```

**性能提升**：
- 处理时间：18ms → 12ms，**又 1.5 倍提升**
- 常量内存缓存命中：100%
- 寄存器使用减少

---

### 📍 优化 4: 分块 + 流处理 - v4_best

**优化思路**：
1. 将大图像分成多个批次
2. 使用多个 CUDA 流并行处理
3. 重叠数据传输和内核执行

**关键改动**：
```cpp
// 1. 创建多个流
const int NUM_STREAMS = 4;
cudaStream_t streams[NUM_STREAMS];
for (int i = 0; i < NUM_STREAMS; i++) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
}

// 2. 将图像分成多个批次
int batch_size = height / NUM_STREAMS;

// 3. 使用流重叠传输和执行
for (int i = 0; i < NUM_STREAMS; i++) {
    int start_row = i * batch_size;
    int end_row = (i == NUM_STREAMS - 1) ? height : (i + 1) * batch_size;
    int rows_to_process = end_row - start_row;
    size_t batch_bytes = width * rows_to_process * sizeof(float);
    
    // ✅ 异步拷贝 H2D，使用流
    CUDA_CHECK(cudaMemcpyAsync(d_input + start_row * width, 
                                h_input + start_row * width,
                                batch_bytes, cudaMemcpyHostToDevice, 
                                streams[i]));
    
    // ✅ 内核执行也使用流
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((width + TILE_SIZE - 1) / TILE_SIZE,
                  (rows_to_process + TILE_SIZE - 1) / TILE_SIZE);
    
    tiled_convolution_kernel<<<grid_dim, block_dim, 0, streams[i]>>>(
        d_input + start_row * width,
        d_output + start_row * width,
        width, rows_to_process
    );
    
    // ✅ 异步拷贝 D2H
    CUDA_CHECK(cudaMemcpyAsync(h_output + start_row * width,
                                d_output + start_row * width,
                                batch_bytes, cudaMemcpyDeviceToHost,
                                streams[i]));
}

// 4. 等待所有流完成
for (int i = 0; i < NUM_STREAMS; i++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
}
```

**性能提升**：
- 处理时间：12ms → 4.2ms，**又 2.8 倍提升**
- 终于达到目标 < 5ms！
- GPU 利用率：35% → 92%

---

## 📊 优化前后对比

| 版本 | 处理时间 | 加速比(vs CPU) | 加速比(vs 前一版) | GPU 利用率 | 带宽利用率 |
|-----|---------|----------------|------------------|-----------|-----------|
| **CPU 参考** | 800ms | 1x | - | - | - |
| **v1_bad (朴素)** | 45ms | 18x | 1x | 35% | 42% |
| **v2_good (共享内存)** | 18ms | 44x | 2.5x | 65% | 78% |
| **v3_better (常量内存)** | 12ms | 67x | 1.5x | 78% | 85% |
| **v4_best (流处理)** | 4.2ms | 190x | 2.8x | 92% | 90% |

**最终成果：从 45ms 优化到 4.2ms，总共 10.7 倍提升！** 🎉

---

## 💯 面试怎么讲这个故事（STAR 法则）

> **S (Situation)**
> "我们在做自动驾驶的实时图像预处理模块，用 5x5 高斯模糊处理 1080p 图像。
> 要求单帧延迟 < 5ms 才能达到 30fps 实时，但最初的朴素 CUDA 实现需要 45ms，
> 只能达到 22fps，GPU 利用率只有 35%。"
>
> **T (Task)**
> 我作为 CUDA 性能优化负责人，负责将性能提升到满足实时要求。
>
> **A (Action)**
> 我按这个流程优化的：
> 1. 先用 Nsight Systems 分析，发现主要瓶颈是全局内存访问效率低，
>    大量重复读取，相邻线程读相同数据但没有利用数据局部性。
>
> 2. 第一步用共享内存优化：每个线程块把需要的图像 tile 加载到共享内存，
>    包括 halo 区域，这样块内所有线程都从共享内存读，全局内存访问
>    减少了 90%，性能从 45ms 到 18ms，2.5 倍提升。
>
> 3. 第二步发现卷积核是 5x5 = 25 个元素，所有线程都只读相同的核，
>    适合放在常量内存，利用常量内存的 L1 缓存，又获得 1.5 倍提升，
>    到 12ms。
>
> 4. 但离 5ms 目标还有距离。观察 GPU 利用率只有 78%，说明还有优化空间。
>    于是用多流处理：把图像分成 4 个批次，用 4 个流重叠 H2D 拷贝、
>    内核执行、D2H 拷贝。GPU 利用率提升到 92%，最终达到 4.2ms。
>
> **R (Result)**
> 最终从 45ms 优化到 4.2ms，总共 10.7 倍提升，满足了 30fps 实时要求，
> GPU 利用率从 35% 提升到 92%。后来我把这个优化模式抽象成了通用的
> 图像卷积模板，团队其他图像处理算法都套用这个模板，平均获得了
> 8-12 倍的性能提升。"

---

## 🚀 编译运行所有版本

```bash
# 编译运行 v1
cd v1_bad && mkdir build && cd build && cmake .. && make && ./naive_convolution

# 编译运行 v2
cd ../../v2_good && mkdir build && cd build && cmake .. && make && ./shared_memory_convolution

# 编译运行 v3
cd ../../v3_better && mkdir build && cd build && cmake .. && make && ./constant_memory_convolution

# 编译运行 v4
cd ../../v4_best && mkdir build && cd build && cmake .. && make && ./tiled_convolution
```

对比 4 个版本的输出，感受 10 倍性能提升的快感！💪

---

## 📝 关键知识点总结

| 优化技术 | 适用场景 | 预期收益 | 实现难度 |
|---------|---------|---------|---------|
| **共享内存** | 有数据复用的算法 | 2-5x | ⭐⭐ |
| **常量内存** | 小的只读数据（< 64KB） | 1.2-2x | ⭐ |
| **合并访问** | 所有内存访问 | 2-10x | ⭐ |
| **流处理** | 可分块的数据处理 | 1.5-3x | ⭐⭐ |
| **分块 (Tiling)** | 大矩阵/图像操作 | 3-10x | ⭐⭐⭐ |

**记住：内存访问是大多数 CUDA 程序的瓶颈，先优化内存访问模式，再考虑计算优化！**
