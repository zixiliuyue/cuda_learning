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

---

## 验收标准

达到以下几点，说明你已掌握本天内容：

1. 程序先后跑完「第一部分：显式内存管理」和「第二部分：统一内存」两段，打印出 1024×1024 矩阵、Block 维度 `(16, 16)`、Grid 维度 `(64, 64)` 的配置信息。
2. 性能总结中「显式内存结果验证」与「统一内存结果验证」都显示 **通过**，说明两种内存管理方式得到相同的正确结果。
3. 能对比读出显式内存的 `Kernel + H2D + D2H` 各段耗时与统一内存的 `Kernel+自动数据迁移` 耗时，并理解统一内存把拷贝开销隐藏进了首次访问的缺页迁移中。
4. 能解释矩阵左上角 4×4 预览中 `C = A + B` 逐元素成立，并理解二维坐标 `(row, col)` 到一维索引 `row * width + col` 的行优先映射。

## 动手挑战

基于 [main.cu](src/main.cu) 中的 `matrix_add_kernel` 与 `matrix_add_kernel_unified` 循序渐进地练习：

1. **改变矩阵尺寸与 Block 形状**：调整 `MATRIX_ROWS/MATRIX_COLS`（如非 16 整数倍的 1000×1000）以及 `BLOCK_SIZE_X/BLOCK_SIZE_Y`（如 32×8、8×32），验证边界检查生效并对比 Kernel 耗时。
2. **公平对比两种内存方式**：给统一内存版本在 Kernel 前加 `cudaMemPrefetchAsync` 预取，或在计时中把显式内存的 H2D+Kernel+D2H 合并统计，得到端到端更公平的性能对比。
3. **新增行优先 vs 列优先实验**：增加一个按 `col * height + row` 索引写入的 Kernel 变体，观察非合并访问对矩阵加法耗时的影响（为 Day 4 转置做铺垫）。
