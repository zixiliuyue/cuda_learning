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
