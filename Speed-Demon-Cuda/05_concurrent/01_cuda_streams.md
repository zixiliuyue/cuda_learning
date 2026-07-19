# CUDA 流 (CUDA Streams)

## 什么是 CUDA 流概念

CUDA 流是一系列按顺序执行的操作（核函数启动、数据传输等）。不同流中的操作可以并发执行，这是实现 GPU 并发的关键机制。

### 核心概念
- **流 (Stream)**：GPU 操作的序列
- **并发执行**：不同流的操作可以并行执行
- **默认流 (Default Stream)**：null stream，隐式同步

### 为什么需要流
1. 隐藏数据传输延迟
2. 充分利用 GPU 资源
3. 实现任务级并行

---

## 流的创建与使用

### 1. 创建流
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
```

### 2. 设置流优先级
```cpp
cudaStream_t high_prio_stream;
cudaStreamCreateWithPriority(&high_prio_stream, cudaStreamNonBlocking, -1);
```

### 3. 在流中启动核函数
```cpp
kernel<<<grid, block, sharedMem, stream>>>(args...);
```

### 4. 在流中执行数据传输
```cpp
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

### 5. 销毁流
```cpp
cudaStreamDestroy(stream);
```

---

## 流的同步机制

### 1. 单流同步
```cpp
cudaStreamSynchronize(stream);
```

### 2. 事件同步
```cpp
cudaEvent_t event;
cudaEventCreate(&event);

// 记录事件到流中
cudaEventRecord(event, stream);

// 等待事件完成
cudaEventSynchronize(event);
```

### 3. 流间依赖
```cpp
// 让 stream2 等待 stream1 中的事件完成
cudaStreamWaitEvent(stream2, event, 0);
```

### 4. 查询流状态
```cpp
cudaError_t status = cudaStreamQuery(stream);
if (status == cudaSuccess) {
    // 流中所有操作已完成
} else if (status == cudaErrorNotReady) {
    // 流中仍有操作在执行
}
```

---

## 重叠执行示例

### 串行执行 vs 流并发执行

#### 串行方式（不使用流）
```cpp
// 所有操作在默认流中执行，顺序执行
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
kernel1<<<grid, block>>>(d_a);
cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
kernel2<<<grid, block>>>(d_b);
cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
```

**时间线**:
```
[Memcpy H->D] -> [Kernel1] -> [Memcpy D->H] -> [Memcpy H->D] -> [Kernel2] -> [Memcpy D->H]
```

#### 并发方式（使用流）
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// stream1 中的操作
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
kernel1<<<grid, block, 0, stream1>>>(d_a);
cudaMemcpyAsync(h_a, d_a, size, cudaMemcpyDeviceToHost, stream1);

// stream2 中的操作
cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2);
kernel2<<<grid, block, 0, stream2>>>(d_b);
cudaMemcpyAsync(h_b, d_b, size, cudaMemcpyDeviceToHost, stream2);

// 等待所有流完成
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

**时间线**:
```
Stream1: [Memcpy H->D] -> [Kernel1] -> [Memcpy D->H]
Stream2:              [Memcpy H->D] -> [Kernel2] -> [Memcpy D->H]
```

---

## 操作步骤

### 编译运行示例
```bash
# Windows
nvcc streams_basic.cu -o streams_basic.exe
.\streams_basic.exe

# Linux
nvcc streams_basic.cu -o streams_basic
./streams_basic
```

### 使用 Nsight 查看并发
1. 运行 `nsys profile ./streams_basic`
2. 使用 `nsys stats report1.qdrep` 查看统计
3. 使用 Nsight Systems GUI 打开报告查看时间线

---

## 注意事项

1. **默认流的隐式同步**：默认流会与其他流同步，尽量使用显式创建的流
2. **流的数量限制**：GPU 支持的并发流数量有限，不要创建过多流
3. **资源依赖**：确保并发操作之间没有数据依赖
4. **错误处理**：异步操作的错误需要在同步点检查

---

## 参考代码
- [streams_basic.cu](./streams_basic.cu) - 基本流使用示例
