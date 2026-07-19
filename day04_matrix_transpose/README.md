# Day 4: 矩阵转置 — 错误处理与调试基础

## 学习目标

1. 完整的 CUDA 错误处理流程
2. 使用 `CUDA_SAFE_CALL` 宏包裹所有 CUDA API 调用
3. 使用 `CHECK_KERNEL_ERROR` 检查异步 Kernel 错误
4. 理解非合并内存访问（转置天然存在）
5. 使用 `printf` 在 Kernel 中进行调试

## 核心概念

### 转置的内存访问模式

```cpp
// 读取: 连续访问（合并）
float val = d_in[row * width + col];

// 写入: 跨行跳跃（非合并）
d_out[col * height + row] = val;
```

### 错误处理宏

```cpp
CUDA_SAFE_CALL(cudaMalloc(&ptr, size));     // 检查 API 错误
CHECK_KERNEL_ERROR();                        // 检查 Kernel 错误
```

## 编译与运行

```powershell
cd day04_matrix_transpose
.\build.bat
.\build\Release\matrix_transpose.exe
```

---

## 验收标准

达到以下几点，说明你已掌握本天内容：

1. 调试版本 Kernel 打印出少量 `[Debug] Thread(x,y): in[..]=.. -> out[..]` 信息，说明你能用 Kernel 内 `printf` 观察线程行为，并理解为何限定 `blockIdx==0 && threadIdx<2` 只让少量线程打印。
2. 输入矩阵按 `in[row][col] = row * COLS + col` 初始化，转置结果左上角 4×4 与 CPU 参考一致，性能总结的「结果验证」显示 **通过**。
3. 「错误处理演示」两段正常执行：演示 1 成功捕获错误方向 `cudaMemcpy` 的错误码，演示 2 打印设备最大 Block 维度并判定当前 `(16,16)` 配置合法，说明你掌握 `CUDA_SAFE_CALL` / `CHECK_KERNEL_ERROR` / `cudaGetLastError` 的用法。
4. 能解释转置 Kernel 中读取 `d_in[row*width+col]` 是合并访问、写入 `d_out[col*height+row]` 是非合并访问，并理解这是转置性能受限的根源。

## 动手挑战

基于 [main.cu](src/main.cu) 中的 `matrix_transpose_kernel` 循序渐进地练习：

1. **扩大调试范围**：修改调试 Kernel 的打印条件（如放宽到 `threadIdx.x < 4 && threadIdx.y < 4`，或换一个 Block），验证线程 (x,y) 与输入/输出一维索引的对应关系。
2. **共享内存优化转置**：新增一个使用 `__shared__ float tile[16][16]` 的转置 Kernel，先按行合并读入共享内存、`__syncthreads()` 后再按合并方式写回，对比它与朴素版本的 Kernel 耗时。
3. **消除 Bank Conflict**：在上一步基础上把共享内存数组改为 `tile[16][16+1]`（padding 一列），观察耗时是否进一步下降，理解 Bank Conflict 的影响（衔接 Day 6）。
