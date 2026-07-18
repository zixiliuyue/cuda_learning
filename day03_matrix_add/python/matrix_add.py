#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 3: 矩阵加法 — Python/Numba 版本"""

import sys
import time
import numpy as np

try:
    from numba import cuda
except ImportError:
    print("[错误] 未安装 Numba。请运行: pip install numba")
    sys.exit(1)

MATRIX_ROWS = 1024
MATRIX_COLS = 1024
BLOCK_X = 16
BLOCK_Y = 16
TOLERANCE = 1e-5


@cuda.jit
def matrix_add_kernel(d_a, d_b, d_c, width, height):
    """矩阵加法 Kernel — Numba 版本"""
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if col < width and row < height:
        idx = row * width + col
        d_c[idx] = d_a[idx] + d_b[idx]


def main():
    print()
    print("=" * 60)
    print("     Day 3: 矩阵加法 — Python/Numba 版本                  ")
    print("=" * 60)
    print()

    num_elements = MATRIX_ROWS * MATRIX_COLS
    print(f"[主机] 矩阵尺寸: {MATRIX_ROWS} x {MATRIX_COLS}")

    # 创建输入数据
    h_a = np.random.uniform(-1.0, 1.0, size=num_elements).astype(np.float32)
    h_b = np.random.uniform(-1.0, 1.0, size=num_elements).astype(np.float32)

    # 复制到设备
    d_a = cuda.to_device(h_a)
    d_b = cuda.to_device(h_b)
    d_c = cuda.device_array_like(h_a)

    # 配置 Grid 和 Block
    grid_x = (MATRIX_COLS + BLOCK_X - 1) // BLOCK_X
    grid_y = (MATRIX_ROWS + BLOCK_Y - 1) // BLOCK_Y

    # 启动 Kernel
    start = time.time()
    matrix_add_kernel[(grid_x, grid_y), (BLOCK_X, BLOCK_Y)](d_a, d_b, d_c, MATRIX_COLS, MATRIX_ROWS)
    cuda.synchronize()
    gpu_time = (time.time() - start) * 1000

    # 复制结果回主机
    h_c = d_c.copy_to_host()

    # CPU 验证
    start = time.time()
    h_c_ref = h_a + h_b
    cpu_time = (time.time() - start) * 1000

    max_diff = np.max(np.abs(h_c - h_c_ref))
    is_correct = max_diff < TOLERANCE

    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  GPU Kernel 时间 : {gpu_time:.4f} ms")
    print(f"  CPU 计算时间    : {cpu_time:.4f} ms")
    print(f"  结果验证        : {'通过' if is_correct else '失败'} (最大误差: {max_diff:.2e})")
    print("=" * 60)
    print()

    return 0 if is_correct else 1


if __name__ == "__main__":
    sys.exit(main())
