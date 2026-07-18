# Day 32: cuSPARSE 稀疏矩阵运算

## 学习目标

1. 理解稀疏矩阵的存储格式，重点掌握 CSR（Compressed Sparse Row）格式
2. 掌握 cuSPARSE Generic API 的使用流程（CUDA 11+ 推荐，CUDA 13 兼容）
3. 使用 `cusparseSpMV` 完成稀疏矩阵-向量乘法（SpMV）
4. 对比稀疏 SpMV 与稠密矩阵-向量乘法在不同稀疏度下的性能差异
5. 理解稀疏计算在图算法、有限元、推荐系统、GNN 中的应用价值

## 核心概念

### 为什么需要稀疏矩阵

在很多实际问题中，矩阵的绝大多数元素都是 0（例如社交网络的邻接矩阵、有限元刚度矩阵）。
如果仍用稠密格式存储 4096×4096 的 float 矩阵，需要约 64 MB 显存，且计算时会对
大量 0 元素做无意义的乘加运算。稀疏格式只存储非零元素，能同时节省显存和算力。

### CSR（压缩稀疏行）格式

CSR 用三个数组表示稀疏矩阵：

| 数组 | 长度 | 含义 |
|------|------|------|
| `values`  | nnz   | 按行优先顺序存储的所有非零元素值 |
| `col_idx` | nnz   | 每个非零元素所在的列号 |
| `row_ptr` | n + 1 | `row_ptr[i]` 为第 i 行首个非零元素在 values 中的下标，`row_ptr[n] = nnz` |

其中 `nnz`（number of non-zeros）为非零元素总数。

### cuSPARSE Generic API 调用流程

```cpp
cusparseCreate(&handle);                       // 1. 创建句柄
cusparseCreateCsr(&matA, ...);                 // 2. 创建 CSR 稀疏矩阵描述符
cusparseCreateDnVec(&vecX, ...);               // 3. 创建稠密向量描述符
cusparseSpMV_bufferSize(...);                  // 4. 查询工作缓冲区大小
cudaMalloc(&buffer, buffer_size);              // 5. 分配缓冲区
cusparseSpMV(...);                             // 6. 执行 y = alpha*A*x + beta*y
// 7. 销毁描述符与句柄
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `src/main.cu` | C++/CUDA 主程序，使用 cuSPARSE SpMV 并与稠密 Kernel 对比 |
| `python/cusparse.py` | Python 版本（scipy.sparse / numpy 模拟） |
| `CMakeLists.txt` | CMake 构建配置（链接 cusparse 库） |
| `build.bat` | Windows 一键编译（sm_120 架构） |
| `build.sh` | macOS/Linux 一键编译 |

## 编译与运行

### Windows (有 NVIDIA GPU)

```powershell
cd day32_cusparse
.\build.bat
.\build\Release\cusparse_spmv.exe
```

### macOS / 无 GPU 环境

```bash
cd day32_cusparse/python
python cusparse.py
```

## 预期输出

```
=============================================================
     Day 32: cuSPARSE - 稀疏矩阵-向量乘法（SpMV）
=============================================================

... (设备信息) ...

[主机] 矩阵规模: 4096 x 4096

-------------------------------------------------------------
[测试 1] cuSPARSE CSR SpMV（不同稀疏度）
-------------------------------------------------------------
  [稀疏度 90%] 非零元素 nnz = ... (实际密度 ...%)
  [稀疏度 95%] 非零元素 nnz = ... (实际密度 ...%)
  [稀疏度 99%] 非零元素 nnz = ... (实际密度 ...%)

-------------------------------------------------------------
[测试 2] 稠密矩阵-向量乘法（手写 Kernel）
-------------------------------------------------------------
  ...

=============================================================
                        性能总结
=============================================================
  方法                         | 耗时(ms)
  ---------------------------- | ------------
  CSR SpMV (稀疏度 90%)        | ...
  CSR SpMV (稀疏度 95%)        | ...
  CSR SpMV (稀疏度 99%)        | ...
  稠密矩阵-向量乘法            | ...
=============================================================
```

## 关键 API 说明

- `cusparseCreateCsr()`：以 CSR 三数组构造稀疏矩阵描述符，指定索引类型、索引基址与数值类型
- `cusparseCreateDnVec()`：构造稠密向量描述符
- `cusparseSpMV_bufferSize()`：查询 SpMV 所需的外部工作缓冲区大小
- `cusparseSpMV()`：执行 `y = alpha * op(A) * x + beta * y`
- `CUSPARSE_SPMV_ALG_DEFAULT`：让 cuSPARSE 自动选择合适的 SpMV 算法

## 练习题

1. 将数值类型改为 double（`CUDA_R_64F`），观察精度与性能变化
2. 尝试 `cusparseSpMM`（稀疏矩阵-稠密矩阵乘法），处理多个右端向量
3. 对比 COO 格式（`cusparseCreateCoo`）与 CSR 格式的差异
4. 统计不同稀疏度下的显存占用，验证稀疏存储的节省效果

## 常见问题

**Q: 链接报错 "unresolved external symbol cusparseSpMV"?**

A: 未链接 cusparse 库。确认 CMakeLists.txt 中包含
`target_link_libraries(cusparse_spmv PUBLIC cusparse)`。

**Q: 旧版 cuSPARSE API（如 cusparseScsrmv）为什么不用？**

A: 传统（legacy）API 从 CUDA 11 起逐步弃用，Generic API 更通用且在 CUDA 13 中受支持，
是官方推荐写法。

## 进阶阅读

- [cuSPARSE Library Documentation](https://docs.nvidia.com/cuda/cusparse/index.html)
- [cuSPARSE Generic API — SpMV](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmv)
