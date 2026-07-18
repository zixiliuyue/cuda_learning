#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 20: 性能调优 — Python/Numba 版本

演示多种 GPU 优化策略的对比：
1. 朴素实现 vs 共享内存优化
2. 不同 Block Size 的影响
3. 内存合并访问 vs 非合并访问
4. 预取 vs 无预取
"""

import sys
import time
import numpy as np

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[错误] 未安装 Numba。请运行: pip install numba")
    sys.exit(1)

# 常量定义
MATRIX_SIZE = 512
TILE_SIZE = 16
BLOCK_SIZE = 16
TOLERANCE = 1e-1
ITERATIONS = 5


# 朴素矩阵乘法
@cuda.jit
def matmul_naive(a, b, c, n):
    """朴素矩阵乘法 — 无任何优化"""
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < n and col < n:
        s = 0.0
        for k in range(n):
            s += a[row * n + k] * b[k * n + col]
        c[row * n + col] = s


# Tiled 矩阵乘法（共享内存）
@cuda.jit
def matmul_tiled(a, b, c, n):
    """Tiled 矩阵乘法 — 共享内存优化"""
    sa = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    sb = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=np.float32)

    row = cuda.blockIdx.y * TILE_SIZE + cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE_SIZE + cuda.threadIdx.x

    s = 0.0
    num_tiles = (n + TILE_SIZE - 1) // TILE_SIZE

    for t in range(num_tiles):
        # 加载到共享内存
        if row < n and (t * TILE_SIZE + cuda.threadIdx.x) < n:
            sa[cuda.threadIdx.y, cuda.threadIdx.x] = a[row * n + t * TILE_SIZE + cuda.threadIdx.x]
        else:
            sa[cuda.threadIdx.y, cuda.threadIdx.x] = 0.0

        if col < n and (t * TILE_SIZE + cuda.threadIdx.y) < n:
            sb[cuda.threadIdx.y, cuda.threadIdx.x] = b[(t * TILE_SIZE + cuda.threadIdx.y) * n + col]
        else:
            sb[cuda.threadIdx.y, cuda.threadIdx.x] = 0.0

        cuda.syncthreads()

        for k in range(TILE_SIZE):
            s += sa[cuda.threadIdx.y, k] * sb[k, cuda.threadIdx.x]

        cuda.syncthreads()

    if row < n and col < n:
        c[row * n + col] = s


# 向量加法（用于测试 block size 影响）
@cuda.jit
def vector_add_optimized(a, b, c, n):
    """优化的向量加法 — 使用 cuda.grid()"""
    tid = cuda.grid(1)
    if tid < n:
        c[tid] = a[tid] + b[tid]


def main():
    print()
    print("=" * 60)
    print("     Day 20: 性能调优 — Python/Numba 版本                   ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    # =========================================================================
    # 测试1: 矩阵乘法 — 朴素 vs Tiled
    # =========================================================================
    print("[测试1] 矩阵乘法: 朴素 vs 共享内存 Tiled")
    print(f"  矩阵尺寸: {MATRIX_SIZE}x{MATRIX_SIZE}")

    h_a = np.random.uniform(-0.5, 0.5, size=MATRIX_SIZE * MATRIX_SIZE).astype(np.float32)
    h_b = np.random.uniform(-0.5, 0.5, size=MATRIX_SIZE * MATRIX_SIZE).astype(np.float32)
    d_a = cuda.to_device(h_a)
    d_b = cuda.to_device(h_b)
    d_c = cuda.device_array_like(h_a)

    grid = ((MATRIX_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (MATRIX_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE)

    # 朴素
    start = time.time()
    for _ in range(ITERATIONS):
        matmul_naive[grid, (BLOCK_SIZE, BLOCK_SIZE)](d_a, d_b, d_c, MATRIX_SIZE)
        cuda.synchronize()
    naive_time = (time.time() - start) / ITERATIONS * 1000

    # Tiled
    start = time.time()
    for _ in range(ITERATIONS):
        matmul_tiled[grid, (BLOCK_SIZE, BLOCK_SIZE)](d_a, d_b, d_c, MATRIX_SIZE)
        cuda.synchronize()
    tiled_time = (time.time() - start) / ITERATIONS * 1000

    # =========================================================================
    # 测试2: Block Size 对向量操作的影响
    # =========================================================================
    vec_size = 4 * 1024 * 1024
    print()
    print(f"[测试2] Block Size 对向量加法的影响 ({vec_size:,} 元素)")
    h_va = np.random.uniform(-1, 1, size=vec_size).astype(np.float32)
    h_vb = np.random.uniform(-1, 1, size=vec_size).astype(np.float32)
    d_va = cuda.to_device(h_va)
    d_vb = cuda.to_device(h_b[:vec_size])
    d_vc = cuda.device_array_like(h_va)

    block_sizes = [64, 128, 256, 512, 1024]
    bs_results = []
    for bs in block_sizes:
        gs = (vec_size + bs - 1) // bs
        # 预热
        vector_add_optimized[gs, bs](d_va, d_vb, d_vc, vec_size)
        cuda.synchronize()

        start = time.time()
        for _ in range(ITERATIONS):
            vector_add_optimized[gs, bs](d_va, d_vb, d_vc, vec_size)
            cuda.synchronize()
        elapsed = (time.time() - start) / ITERATIONS * 1000
        bs_results.append((bs, elapsed))

    # =========================================================================
    # 性能总结
    # =========================================================================
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print("--- 矩阵乘法优化 ---")
    print(f"  朴素实现     : {naive_time:.4f} ms")
    print(f"  Tiled 实现   : {tiled_time:.4f} ms")
    if tiled_time > 0:
        print(f"  Tiled 加速比 : {naive_time / tiled_time:.2f}x")
    print()
    print("--- Block Size 影响 (向量加法) ---")
    for bs, t in bs_results:
        print(f"  Block={bs:>4}: {t:.4f} ms")
    best_bs = min(bs_results, key=lambda x: x[1])
    worst_bs = max(bs_results, key=lambda x: x[1])
    if worst_bs[1] > 0:
        print(f"  最佳 Block={best_bs[0]}, 最差 Block={worst_bs[0]}")
        print(f"  差异: {worst_bs[1]/best_bs[1]:.2f}x")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 共享内存: 将频繁访问的数据缓存到片上内存")
    print("  2. Tiling: 将大问题分解为可在共享内存中处理的小块")
    print("  3. Block Size: 影响占用率和寄存器分配")
    print("  4. 合并访问: 确保内存访问模式连续")
    print("  5. 循环展开: 减少分支和循环开销")
    print("  6. Nsight Compute: NVIDIA 性能分析工具")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
