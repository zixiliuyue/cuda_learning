# 并发核函数 (Concurrent Kernels)

## 多个核函数并发执行

### 什么是核函数并发
GPU 可以同时执行多个核函数，只要它们使用不同的流并且有足够的硬件资源。这是实现任务级并行的重要方式。

### 硬件支持条件
- **多引擎 (Multi-Engine)**：GPU 需要支持多个独立的引擎
- **流多处理器 (SM)**：足够的 SM 资源来容纳多个核函数
- **计算能力**：通常需要计算能力 3.5 及以上

---

## 如何最大化并发

### 1. 使用独立的流
每个核函数在不同的流中启动：
```cpp
cudaStream_t stream1, stream2, stream3;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamCreate(&stream3);

kernelA<<<grid, block, 0, stream1>>>(args);
kernelB<<<grid, block, 0, stream2>>>(args);
kernelC<<<grid, block, 0, stream3>>>(args);
```

### 2. 优化核函数资源占用
- **减少每个核函数的资源使用**：让更多核函数可以并发
- **调整 Grid/Block 大小**：避免单个核函数占用所有 SM

### 3. 避免隐式同步
- 不要在核函数之间使用默认流操作
- 避免不必要的 `cudaDeviceSynchronize()`

### 4. 使用 MPS (Multi-Process Service)
对于多进程场景，使用 MPS 可以提高并发效率：
```bash
# 启动 MPS 服务
nvidia-cuda-mps-control -d
```

---

## 性能对比示例

### 示例场景：三个独立的向量运算

#### 串行版本（所有核函数在同一个流中）
```cpp
// 所有核函数在默认流中顺序执行
vectorAdd<<<grid, block>>>(d_a, d_b, d_c, N);
vectorMul<<<grid, block>>>(d_c, d_d, d_e, N);
vectorScale<<<grid, block>>>(d_e, d_f, factor, N);
```

**时间线**:
```
[vectorAdd] -> [vectorMul] -> [vectorScale]
```

#### 并发版本（每个核函数在独立流中）
```cpp
cudaStream_t stream1, stream2, stream3;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamCreate(&stream3);

// 假设三个核函数处理不同的数据，没有依赖
vectorAdd<<<grid, block, 0, stream1>>>(d_a1, d_b1, d_c1, N);
vectorMul<<<grid, block, 0, stream2>>>(d_a2, d_b2, d_c2, N);
vectorScale<<<grid, block, 0, stream3>>>(d_a3, d_c3, factor, N);
```

**时间线**:
```
Stream1: [vectorAdd]
Stream2: [vectorMul]
Stream3: [vectorScale]
```

### 预期性能提升
- **理论加速比**：最高可达 3x（如果核函数完全独立）
- **实际加速比**：取决于硬件资源和核函数特性

---

## 检测并发能力

### 查询设备并发支持
```cpp
int deviceId = 0;
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, deviceId);

printf("异步引擎数量: %d\n", prop.asyncEngineCount);
printf("并发核函数支持: %s\n", prop.concurrentKernels ? "Yes" : "No");
```

### 关键参数
- `asyncEngineCount`：异步复制引擎数量
- `concurrentKernels`：是否支持并发核函数
- `multiProcessorCount`：SM 数量

---

## 操作步骤

### 1. 编译运行示例
```bash
# Windows
nvcc concurrent_kernels.cu -o concurrent_kernels.exe
.\concurrent_kernels.exe

# Linux
nvcc concurrent_kernels.cu -o concurrent_kernels
./concurrent_kernels
```

### 2. 使用 Nsight Systems 分析
```bash
# 生成性能报告
nsys profile -o concurrent_kernels_report ./concurrent_kernels

# 查看统计信息
nsys stats concurrent_kernels_report.qdrep
```

### 3. 在 Nsight Systems GUI 中查看
- 打开报告文件
- 查看 Timeline 面板
- 观察不同流中的核函数是否在时间上重叠

---

## 常见问题与解决

### 问题 1：核函数没有并发执行
**可能原因**:
- 核函数太大，占用了所有 SM
- 使用了默认流，存在隐式同步
- GPU 不支持并发核函数

**解决方案**:
- 减小核函数的 Grid/Block 大小
- 使用显式创建的流
- 检查设备属性

### 问题 2：并发性能提升不明显
**可能原因**:
- 核函数太小，启动开销占比大
- 数据依赖导致序列化
- 内存带宽成为瓶颈

**解决方案**:
- 增加每个核函数的工作量
- 确保没有数据依赖
- 优化内存访问模式

---

## 最佳实践

1. **从小规模开始测试**：先测试 2-3 个核函数的并发
2. **使用事件精确计时**：通过 cudaEvent 测量实际执行时间
3. **监控资源使用**：使用 Nsight 分析每个核函数的资源占用
4. **避免过度并发**：不要创建超过硬件能力的并发任务
5. **考虑流水线设计**：将计算和数据传输重叠

---

## 参考代码
- [concurrent_kernels.cu](./concurrent_kernels.cu) - 核函数并发示例
