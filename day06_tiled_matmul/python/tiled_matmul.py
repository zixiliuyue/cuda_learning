#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 6: Tiled Matrix Multiplication — Python/Numba 版本"""

import sys
import time
import numpy as np

try:
    from numba import cuda
except ImportError:
    print("[错误] 未安装 Numba。请运行: pip install numba")
    sys.exit(1)

MATRIX_SIZE = 512  # Numba 共享内存限制较小
TILE_WIDTH = 16
BLOCK_SIZE = 16
TOLERANCE = 1e-2


@cuda.jit
def matmul_naive_kernel(d_a, d_b, d_c, width):
    """朴素矩阵乘法"""
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if col < width and row < width:
        sum_val = 0.0
        for k in range(width):
            sum_val += d_a[row * width + k] * d_b[k * width + col]
        d_c[row * width + col] = sum_val


@cuda.jit
def matmul_tiled_kernel(d_a, d_b, d_c, width):
    """Tiled 矩阵乘法（共享内存优化）"""
    # 声明共享内存
    ds_a = cuda.shared.array((TILE_WIDTH, TILE_WIDTH), dtype=np.float32)
    ds_b = cuda.shared.array((TILE_WIDTH, TILE_WIDTH), dtype=np.float32)

    col = cuda.blockIdx.x * TILE_WIDTH + cuda.threadIdx.x
    row = cuda.blockIdx.y * TILE_WIDTH + cuda.threadIdx.y

    sum_val = 0.0
    num_tiles = (width + TILE_WIDTH - 1) // TILE_WIDTH

    for m in range(num_tiles):
        a_col = m * TILE_WIDTH + cuda.threadIdx.x
        b_row = m * TILE_WIDTH + cuda.threadIdx.y

        if row < width and a_col < width:
            ds_a[cuda.threadIdx.y, cuda.threadIdx.x] = d_a[row * width + a_col]
        else:
            ds_a[cuda.threadIdx.y, cuda.threadIdx.x] = 0.0

        if b_row < width and col < width:
            ds_b[cuda.threadIdx.y, cuda.threadIdx.x] = d_b[b_row * width + col]
        else:
            ds_b[cuda.threadIdx.y, cuda.threadIdx.x] = 0.0

        cuda.syncthreads()

        for k in range(TILE_WIDTH):
            sum_val += ds_a[cuda.threadIdx.y, k] * ds_b[k, cuda.threadIdx.x]

        cuda.syncthreads()

    if row < width and col < width:
        d_c[row * width + col] = sum_val


def main():
    print()
    print("=" * 60)
    print("     Day 6: Tiled Matrix Multiplication — Python/Numba     ")
    print("=" * 60)
    print()

    print(f"[主机] 矩阵尺寸: {MATRIX_SIZE} x {MATRIX_SIZE}")

    h_a = np.random.uniform(-0.5, 0.5, size=MATRIX_SIZE * MATRIX_SIZE).astype(np.float32)
    h_b = np.random.uniform(-0.5, 0.5, size=MATRIX_SIZE * MATRIX_SIZE).astype(np.float32)

    d_a = cuda.to_device(h_a)
    d_b = cuda.to_device(h_b)
    d_c = cuda.device_array_like(h_a)

    grid_x = (MATRIX_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (MATRIX_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    # 朴素实现
    start = time.time()
    matmul_naive_kernel[(grid_x, grid_y), (BLOCK_SIZE, BLOCK_SIZE)](d_a, d_b, d_c, MATRIX_SIZE)
    cuda.synchronize()
    naive_time = (time.time() - start) * 1000
    h_c_naive = d_c.copy_to_host()

    # Tiled 实现
    start = time.time()
    matmul_tiled_kernel[(grid_x, grid_y), (BLOCK_SIZE, BLOCK_SIZE)](d_a, d_b, d_c, MATRIX_SIZE)
    cuda.synchronize()
    tiled_time = (time.time() - start) * 1000
    h_c_tiled = d_c.copy_to_host()

    # CPU 验证
    h_a_2d = h_a.reshape(MATRIX_SIZE, MATRIX_SIZE)
    h_b_2d = h_b.reshape(MATRIX_SIZE, MATRIX_SIZE)
    h_c_ref = (h_a_2d @ h_b_2d).flatten()

    max_diff_naive = np.max(np.abs(h_c_naive - h_c_ref))
    max_diff_tiled = np.max(np.abs(h_c_tiled - h_c_ref))

    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  GPU 朴素实现  : {naive_time:.2f} ms")
    print(f"  GPU Tiled 实现: {tiled_time:.2f} ms")
    print(f"  Tiled vs 朴素 : {naive_time / tiled_time:.2f}x 更快")
    print(f"  朴素结果验证  : {'通过' if max_diff_naive < TOLERANCE else '失败'} (误差: {max_diff_naive:.2e})")
    print(f"  Tiled 结果验证: {'通过' if max_diff_tiled < TOLERANCE else '失败'} (误差: {max_diff_tiled:.2e})")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
