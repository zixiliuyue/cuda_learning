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
