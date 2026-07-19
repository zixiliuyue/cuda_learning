# Day 13: cuBLAS - NVIDIA 官方线性代数库

## 学习目标

1. **学会调用** cuBLAS 库完成单精度矩阵乘法（SGEMM），掌握 `cublasCreate` / `cublasSgemm` / `cublasDestroy` 的完整流程
2. **理解** cuBLAS 采用**列优先（Column-Major）**存储，并解释行优先数据如何通过调整参数顺序来适配
3. **实现**一个手写的 Tiled 矩阵乘法 Kernel（共享内存分块），作为性能对比基准
4. **测量并对比** cuBLAS 与手写 Kernel 在 2048×2048 矩阵上的耗时
5. **验证**两种实现结果一致性（容差 `1e-2`），解释浮点误差来源

---

## 前置依赖

- **Day 6 Tiled MatMul**：本天手写的 `matmul_tiled_kernel` 就是共享内存分块矩阵乘法，需先理解 tile 缓存原理
- **共享内存 / 合并访问**：理解 `__shared__` 与 `__syncthreads()` 的使用
- **计时工具**：复用 `common/include/timer.h` 中的 `GpuTimer`（基于 CUDA Event）

---

## 核心概念

### 1. 什么是 cuBLAS

cuBLAS 是 NVIDIA 对 BLAS（Basic Linear Algebra Subprograms）标准的 GPU 实现，分三个层级：

| 层级 | 运算类型 | 典型函数 |
|------|----------|----------|
| Level 1 | 向量-向量（如点积、AXPY） | `cublasSaxpy`、`cublasSdot` |
| Level 2 | 矩阵-向量（如 GEMV） | `cublasSgemv` |
| Level 3 | 矩阵-矩阵（如 GEMM） | `cublasSgemm` |

本天使用 Level 3 的 `cublasSgemm` 计算 `C = alpha * A * B + beta * C`（本例 `alpha=1.0`、`beta=0.0`）。

### 2. 列优先存储的坑

cuBLAS 沿袭 Fortran BLAS 的**列优先**约定，而 C/C++ 数组是**行优先**。本例利用一个恒等式巧妙规避转置开销：

```
行优先的 C = A * B
   等价于
列优先视角下的 C^T = B^T * A^T

因此调用时把 B 当作第一个矩阵、A 当作第二个矩阵传入，
就能在不显式转置的情况下得到正确的行优先结果。
```

对应源码中的调用（`CUBLAS_OP_N` 表示不转置）：

```c
cublasSgemm(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,  // A、B 都不转置
    n, n, n,                   // M, N, K
    &alpha,
    d_b, n,                    // 第一个矩阵传入 B
    d_a, n,                    // 第二个矩阵传入 A
    &beta,
    d_c, n);
```

### 3. 手写 Tiled Kernel 作为对比

`matmul_tiled_kernel` 使用 `TILE_WIDTH=16` 的共享内存分块：每个 block 把 A、B 的一个 16×16 tile 载入 `__shared__`，同步后累加，减少全局内存访问：

```
每个 block 循环遍历 K 维度的所有 tile：
    ds_a[ty][tx] = A 的 tile 元素（越界补 0）
    ds_b[ty][tx] = B 的 tile 元素（越界补 0）
    __syncthreads()
    for k in 0..TILE_WIDTH:  sum += ds_a[ty][k] * ds_b[k][tx]
    __syncthreads()
最终 c[row*n+col] = sum
```

### 4. 错误检查宏

源码定义了 `CUBLAS_SAFE_CALL`，检查返回的 `cublasStatus_t` 是否为 `CUBLAS_STATUS_SUCCESS`，非成功则打印状态码并退出。

---

## 文件结构

```
day13_cublas/
├── src/main.cu          # cuBLAS SGEMM + 手写 Tiled Kernel 对比
├── python/cublas.py     # Python 版（CuPy 封装 cuBLAS，否则 NumPy fallback）
├── CMakeLists.txt       # 版本感知架构选择 + 链接 cublas + /utf-8
├── build.bat            # Windows 编译脚本（生成 cublas_demo.exe）
├── build.sh             # Linux/macOS 编译脚本
└── README.md            # 本文件
```

关键参数：`MATRIX_SIZE=2048`、`TILE_WIDTH=16`、`TOLERANCE=1e-2`。

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat（内部调用 CMake + MSVC）

```powershell
cd day13_cublas
.\build.bat
.\build\Release\cublas_demo.exe
```

方式二：手动使用 CMake

```powershell
cd day13_cublas
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\cublas_demo.exe
```

### Linux / macOS

```bash
cd day13_cublas
bash build.sh
./build/cublas_demo
```

> CMakeLists 已通过 `target_link_libraries(cublas_demo cublas)` 链接 cuBLAS，无需手动指定 `-lcublas`。

### Python 版本

```bash
cd day13_cublas/python
python cublas.py
```

> 若安装了 CuPy（`pip install cupy-cuda12x`），会用 `cp.matmul`（底层调用 cuBLAS）在 512×512 矩阵上计时；否则自动 fallback 到 NumPy。

---

## 知识点总结

1. **cuBLAS 三层结构**：Level 1（向量）、Level 2（矩阵-向量）、Level 3（矩阵-矩阵），GEMM 属于 Level 3。
2. **句柄管理**：`cublasCreate(&handle)` 创建、`cublasDestroy(handle)` 销毁，句柄贯穿整个会话。
3. **`cublasSgemm` 语义**：`C = alpha * op(A) * op(B) + beta * C`，`op` 由 `CUBLAS_OP_N`/`CUBLAS_OP_T` 控制。
4. **列优先适配**：行优先的 `A*B` 用「传 B 再传 A」的方式实现，避免显式转置。
5. **手写 Tiled Kernel**：`TILE_WIDTH=16` 共享内存分块，是理解库为何高效的对照组。
6. **结果验证容差**：GPU 与不同实现的浮点累加顺序不同，用 `1e-2` 容差判定一致。
7. **性能预期**：在大矩阵（2048×2048）上，高度调优的 cuBLAS 通常显著快于朴素的手写 Tiled Kernel。

---

## 验收标准

- 程序打印设备信息、矩阵尺寸（2048×2048，约 16 MB × 3）后正常运行到底
- 输出「cuBLAS SGEMM」与「手写 CUDA Kernel」两项耗时（ms）
- 「结果一致性」显示**通过**（两种实现在容差 `1e-2` 内一致）
- 一般情况下 cuBLAS 耗时明显小于手写 Kernel，说明你理解了库级优化的价值

---

## 动手挑战

1. **尺寸扫描**：把 `MATRIX_SIZE` 改为 512、1024、4096，记录两种实现的耗时并计算加速比，观察规模对差距的影响。
2. **换精度 / 换运算**：把数据与 `cublasSgemm` 换成双精度 `cublasDgemm`（数组改 `double`），对比精度与速度变化；或尝试用 `cublasSgemv` 做矩阵-向量乘。
3. **优化手写 Kernel**：给 `matmul_tiled_kernel` 增大 `TILE_WIDTH`（如 32）或让每线程计算多个输出元素（register blocking），看能否缩小与 cuBLAS 的差距。

---

## 常见问题

- **Q：结果一致性为何用 `1e-2` 这么大的容差？**
  A：cuBLAS 内部可能使用不同的分块与累加顺序（甚至 Tensor Core），与手写 Kernel 的浮点累加次序不同，单精度下累积误差偏大属于正常现象。

- **Q：为什么要交换 A、B 的传参顺序？**
  A：因为 cuBLAS 是列优先，而我们的数据是行优先。利用 `C^T = B^T * A^T` 的恒等关系，交换顺序即可在不转置的情况下得到正确的行优先结果。

- **Q：链接时报 `unresolved external symbol cublasSgemm`？**
  A：说明未链接 cuBLAS 库。使用本项目的 CMake 会自动 `target_link_libraries(... cublas)`；若手动用 nvcc 编译需加 `-lcublas`。

---

## 延伸阅读

- [cuBLAS 官方文档](https://docs.nvidia.com/cuda/cublas/index.html)
- [CuPy 文档（Python 侧封装 cuBLAS/cuFFT 等）](https://docs.cupy.dev/en/stable/)
