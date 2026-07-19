# Day 6: 共享内存优化 — Tiled Matrix Multiplication

## 学习目标

1. 共享内存（Shared Memory）的概念和使用方法
2. `__shared__` 关键字与内存分配
3. `__syncthreads()` 线程同步的作用
4. Tiling 技术：分块计算减少全局内存访问
5. Bank Conflict 原理与避免策略

## 核心概念

### Tiling 原理

将大矩阵分成 `TILE_WIDTH x TILE_WIDTH` 的小块：
- 将 Tile 加载到共享内存中，Block 内所有线程共享
- 全局内存访问次数从 `O(N^3)` 降低到 `O(N^3 / TILE_WIDTH)`

### 共享内存使用

```cpp
__shared__ float ds_a[TILE_WIDTH][TILE_WIDTH];
__shared__ float ds_b[TILE_WIDTH][TILE_WIDTH];
```

### 关键同步点

```cpp
__syncthreads();  // 加载完成后同步
// ... 计算 ...
__syncthreads();  // 计算完成后同步（防止覆盖共享内存）
```

## 性能对比

| 实现 | 全局内存访问 | 相对速度 |
|------|-------------|----------|
| CPU | N/A | 基准 |
| GPU 朴素 | 2*N^3 | 中等 |
| GPU Tiled | 2*N^3/TILE_WIDTH | 最快 |

## 编译与运行

```powershell
cd day06_tiled_matmul
.\build.bat
.\build\Release\tiled_matmul.exe
```

---

## 验收标准

达到以下几点，说明你已掌握本天内容：

1. 程序对 1024×1024 方阵依次跑完 CPU 参考、GPU 朴素、GPU Tiled 三种实现，性能总结中「朴素结果验证」与「Tiled 结果验证」都显示 **通过**（容差 `1e-3`）。
2. 能从「性能总结」读出三行加速比：GPU 朴素相对 CPU、GPU Tiled 相对 CPU，以及 `Tiled vs 朴素 xx更快`，且 Tiled 明显快于朴素实现，体现共享内存优化的收益。
3. 能解释 `TILE_WIDTH=16` 时共享内存用量为 `2 * 16 * 16 * 4 = 2KB`，远低于每 Block 共享内存上限，并说明两处 `__syncthreads()` 分别防止「加载未完成就计算」和「计算未完成就覆盖」。
4. 理解 Tiled 把全局内存访问从每元素 `2*width` 次降到约 `2*width/TILE_WIDTH` 次，是加速的根本原因。

## 动手挑战

基于 [main.cu](src/main.cu) 中的 `matmul_naive_kernel` 与 `matmul_tiled_kernel` 循序渐进地练习：

1. **调节 Tile 大小**：把 `TILE_WIDTH` 和 `BLOCK_SIZE` 同步改为 8 与 32（二者必须相等），重新测速，观察加速比随 Tile 增大的变化以及共享内存用量约束。
2. **验证非方阵与边界**：将 `MATRIX_SIZE` 改为非 16 整数倍的值（如 1000），确认 Tiled Kernel 中的越界填 0 逻辑仍让结果验证通过。
3. **消除 Bank Conflict / 提高计算强度**：把 `ds_a`/`ds_b` 改为 `[TILE_WIDTH][TILE_WIDTH+1]` padding 版本对比耗时，或让每个线程计算多个输出元素（寄存器分块），进一步逼近朴素实现之上的性能上限。
