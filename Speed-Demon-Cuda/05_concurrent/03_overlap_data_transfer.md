# 数据传输与计算重叠

## 分页锁定内存 (Pinned Memory)

### 什么是 Pinned Memory
Pinned Memory（也叫 Page-locked Memory）是物理内存中被锁定的页面，不会被操作系统交换到磁盘。这使得 GPU 的 DMA 引擎可以直接访问这些内存，实现更快的数据传输。

### 为什么需要 Pinned Memory
- **更高的传输带宽**：比可分页内存传输速度快 2-5 倍
- **支持异步传输**：异步传输必须使用 Pinned Memory
- **更可靠的传输**：避免页面交换导致的传输错误

### 分配 Pinned Memory
```cpp
// 方式 1：cudaMallocHost
float* h_data;
cudaMallocHost(&h_data, size * sizeof(float));

// 方式 2：cudaHostAlloc
cudaHostAlloc(&h_data, size * sizeof(float), cudaHostAllocDefault);

// 释放 Pinned Memory
cudaFreeHost(h_data);
```

### Pinned Memory 的类型
```cpp
// 默认：可被所有 CUDA 上下文使用
cudaHostAlloc(&ptr, size, cudaHostAllocDefault);

// 可写合并：提高 CPU 写入性能
cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined);

// 映射到设备地址空间，零拷贝
cudaHostAlloc(&ptr, size, cudaHostAllocMapped);

// 可被所有设备访问（需要 UVA）
cudaHostAlloc(&ptr, size, cudaHostAllocPortable);
```

---

## 异步数据传输

### 同步 vs 异步传输

#### 同步传输
```cpp
// 阻塞调用，传输完成后才返回
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
```

#### 异步传输
```cpp
// 非阻塞调用，立即返回
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
```

### 异步传输的关键条件
1. 主机内存必须是 Pinned Memory
2. 必须在非默认流中执行
3. 传输完成后需要同步

### 异步传输示例
```cpp
float *h_data, *d_data;
cudaStream_t stream;

// 分配 Pinned Memory
cudaMallocHost(&h_data, size * sizeof(float));
cudaMalloc(&d_data, size * sizeof(float));
cudaStreamCreate(&stream);

// 初始化数据...

// 异步传输主机到设备
cudaMemcpyAsync(d_data, h_data, size * sizeof(float),
                cudaMemcpyHostToDevice, stream);

// 可以在此期间执行其他 CPU 任务...

// 等待传输完成
cudaStreamSynchronize(stream);
```

---

## 计算与传输重叠技术

### 基本思想
将数据传输和计算安排在不同的流中，使它们可以并行执行。

### 单缓冲区重叠
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// stream1: 传输数据
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1);

// stream2: 计算（假设数据已经准备好）
kernel<<<grid, block, 0, stream2>>>(d_other_data);

// 等待所有操作完成
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

**时间线**:
```
Stream1: [Memcpy H->D]
Stream2:           [Kernel]
```

### 多阶段重叠（同一数据）
```cpp
// 阶段 1：传输数据
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

// 阶段 2：等待传输完成后计算
cudaStreamSynchronize(stream);
kernel<<<grid, block, 0, stream>>>(d_data);

// 阶段 3：传输结果回主机
cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream);
```

---

## 流水线设计

### 什么是流水线
将大数据分成多个块（chunks），每个块独立经过传输-计算-传输回的过程，形成流水线。

### 流水线示意图
```
Chunk 1:  [H->D] -> [Compute] -> [D->H]
Chunk 2:           [H->D] -> [Compute] -> [D->H]
Chunk 3:                     [H->D] -> [Compute] -> [D->H]
```

### 四阶段流水线示例

#### 1. 定义流水线参数
```cpp
const int NUM_STREAMS = 4;  // 流水线阶段数
const int CHUNK_SIZE = N / NUM_STREAMS;
```

#### 2. 分配资源
```cpp
float *h_data[NUM_STREAMS];
float *d_data[NUM_STREAMS];
float *h_result[NUM_STREAMS];
float *d_result[NUM_STREAMS];
cudaStream_t streams[NUM_STREAMS];

for (int i = 0; i < NUM_STREAMS; i++) {
    cudaMallocHost(&h_data[i], CHUNK_SIZE * sizeof(float));
    cudaMalloc(&d_data[i], CHUNK_SIZE * sizeof(float));
    cudaMallocHost(&h_result[i], CHUNK_SIZE * sizeof(float));
    cudaMalloc(&d_result[i], CHUNK_SIZE * sizeof(float));
    cudaStreamCreate(&streams[i]);
}
```

#### 3. 执行流水线
```cpp
for (int i = 0; i < NUM_STREAMS; i++) {
    int offset = i * CHUNK_SIZE;

    // 1. 异步传输数据到设备
    cudaMemcpyAsync(d_data[i], h_data[i],
                    CHUNK_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice,
                    streams[i]);

    // 2. 异步执行核函数
    kernel<<<grid, block, 0, streams[i]>>>(
        d_data[i], d_result[i], CHUNK_SIZE);

    // 3. 异步传输结果回主机
    cudaMemcpyAsync(h_result[i], d_result[i],
                    CHUNK_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    streams[i]);
}

// 等待所有流完成
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
}
```

### 流水线性能分析
- **理论加速比**：接近 NUM_STREAMS 倍
- **启动开销**：需要填充流水线
- **最佳阶段数**：根据硬件能力调整

---

## 操作步骤

### 1. 编译运行示例
```bash
# Windows
nvcc overlap_transfer.cu -o overlap_transfer.exe
.\overlap_transfer.exe

nvcc pipeline.cu -o pipeline.exe
.\pipeline.exe

# Linux
nvcc overlap_transfer.cu -o overlap_transfer
./overlap_transfer

nvcc pipeline.cu -o pipeline
./pipeline
```

### 2. 使用 Nsight 分析流水线
```bash
nsys profile -o pipeline_report ./pipeline
```

在 Nsight Systems GUI 中观察：
- Timeline 中各流的重叠情况
- 每个流的 H->D、Kernel、D->H 阶段
- 计算和传输的重叠程度

### 3. 调整流水线参数
尝试不同的 `NUM_STREAMS` 值，找到最佳性能点：
- 太小：流水线填充不足，重叠不够
- 太大：资源竞争，调度开销增加

---

## 最佳实践

### 内存管理
1. **及时释放 Pinned Memory**：Pinned Memory 是稀缺资源
2. **合理分配大小**：不要分配超过需要的 Pinned Memory
3. **考虑零拷贝**：对于少量数据，可以考虑使用 mapped memory

### 流水线设计
1. **选择合适的块大小**：平衡传输和计算开销
2. **避免过度细分**：块太小会增加调度开销
3. **使用事件精确控制**：可以用事件实现更复杂的依赖关系

### 性能优化
1. **使用 Write-Combined Memory**：提高 CPU 写入性能
2. **批量传输**：减少小传输的开销
3. **对齐内存**：确保内存地址对齐到 4KB 边界

---

## 常见问题

### Q1: 为什么异步传输没有生效？
**A**: 检查：
- 主机内存是否是 Pinned Memory
- 是否在显式流中执行
- 是否正确调用了同步函数

### Q2: 流水线性能没有提升？
**A**: 可能原因：
- 计算时间远小于传输时间（瓶颈在传输）
- 块大小设置不合理
- 设备的 DMA 引擎数量有限

### Q3: Pinned Memory 分配失败？
**A**: 可能是系统物理内存不足，减少分配大小或释放其他内存。

---

## 参考代码
- [overlap_transfer.cu](./overlap_transfer.cu) - 数据传输与计算重叠示例
- [pipeline.cu](./pipeline.cu) - 流水线示例
