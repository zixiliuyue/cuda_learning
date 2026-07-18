# Day 3: 矩阵加法 — CUDA 内存模型与数据传输

## 学习目标

1. 二维线程组织的实际应用
2. 二维索引计算：`row = blockIdx.y * blockDim.y + threadIdx.y`
3. 行优先（Row-Major）内存布局
4. 显式内存管理：`cudaMalloc` + `cudaMemcpy`
5. 统一内存：`cudaMallocManaged`
6. 主机↔设备数据传输的完整流程

## 核心概念

### 二维索引计算

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### 行优先展平

```cpp
int idx = row * width + col;  // C/C++ 默认行优先
```

### 显式内存 vs 统一内存

| 特性 | 显式内存 (`cudaMalloc`) | 统一内存 (`cudaMallocManaged`) |
|------|------------------------|-------------------------------|
| 数据传输 | 手动 `cudaMemcpy` | 自动页迁移 |
| 代码复杂度 | 较复杂 | 较简单 |
| 性能可控性 | 高 | 低 |
| 适用场景 | 生产环境 | 快速原型 |

## 编译与运行

```powershell
cd day03_matrix_add
.\build.bat
.\build\Release\matrix_add.exe
```
