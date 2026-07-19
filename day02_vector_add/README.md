# Day 2: 向量加法 — CUDA 执行模型与线程层次

## 学习目标

1. 一维线程组织的实际应用 — 每个线程处理一个数组元素
2. 全局线程 ID 的正确计算: `tid = blockIdx.x * blockDim.x + threadIdx.x`
3. 边界检查 — 防止线程访问超出数组范围的内存
4. GPU 结果与 CPU 结果对比验证
5. 使用 `CUDA_SAFE_CALL` 宏进行错误检查
6. 使用 `GpuTimer` 测量 Kernel 执行时间

## 核心概念

### 全局线程 ID 计算

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

这是 CUDA 编程中最常用的索引计算方式。

### 边界检查

```cpp
if (tid < n) {
    d_c[tid] = d_a[tid] + d_b[tid];
}
```

由于 Block 数量是向上取整的，最后几个 Block 中的部分线程可能会超出数组范围，必须通过 `if (tid < n)` 进行边界保护。

### Grid 维度计算

```cpp
int grid_size = (total_threads + block_size - 1) / block_size;
```

向上取整除法，确保覆盖所有元素。

## 线程分配策略

- 向量长度 N = 1,000,000
- 每个 Block 256 个线程（Warp 大小的整数倍）
- Block 数量 = (1,000,000 + 255) / 256 = 3907
- 总线程数 = 3907 * 256 = 1,000,192（多余 192 个线程通过边界检查跳过）

## 编译与运行

```powershell
cd day02_vector_add
.\build.bat
.\build\Release\vector_add.exe
```

## 练习题

1. 修改 `VECTOR_SIZE` 为不是 256 的整数倍的数值，验证边界检查是否生效
2. 将 `THREADS_PER_BLOCK` 改为 128 或 512，观察性能变化
3. 如果不加 `if (tid < n)` 边界检查，程序会发生什么？

---

## 验收标准

达到以下几点，说明你已掌握本天内容：

1. 程序打印出 Kernel 启动配置：`Block 维度 256`、`Grid 维度 3907 Blocks`、`总线程数 1000192`、`多余线程 192 个`，与向上取整的线程分配策略一致。
2. 「结果预览」表格中 GPU C[i] 与 CPU C[i] 数值逐列吻合，且性能总结的「结果验证」显示 **通过**（程序返回码为 0）。
3. 能在「性能总结」里读出 GPU Kernel 时间、CPU 计算时间与 GPU 加速比，并理解此处 Kernel 计时不含 H2D/D2H 传输开销。
4. 理解 100 万元素向量属于访存密集型（memory-bound）任务，加速比受显存带宽限制，而非无限增大。

## 动手挑战

基于 [main.cu](src/main.cu) 中的 `vector_add_kernel` 循序渐进地练习：

1. **验证边界检查**：将 `VECTOR_SIZE` 改为非 256 整数倍的值（如 1000003），确认多余线程数变化且结果仍验证通过；再临时去掉 `if (tid < n)`，观察是否触发非法内存访问错误。
2. **对比 Block 尺寸性能**：把 `THREADS_PER_BLOCK` 依次设为 64、128、256、512，记录每次的 Kernel 时间与加速比，找出本机的最优配置。
3. **改用统一内存**：把 `cudaMalloc` + `cudaMemcpy` 改写为 `cudaMallocManaged` 版本，去掉显式 H2D/D2H 拷贝，对比代码简洁度与端到端耗时的变化。
