#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 4: 矩阵转置 — Python/Numba 版本"""

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
def matrix_transpose_kernel(d_in, d_out, width, height):
    """矩阵转置 Kernel"""
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if col < width and row < height:
        val = d_in[row * width + col]
        d_out[col * height + row] = val


def main():
    print()
    print("=" * 60)
    print("     Day 4: 矩阵转置 — Python/Numba 版本                  ")
    print("=" * 60)
    print()

    num_elements = MATRIX_ROWS * MATRIX_COLS
    print(f"[主机] 矩阵尺寸: {MATRIX_ROWS} x {MATRIX_COLS}")

    # 初始化输入矩阵（使用模式便于验证）
    h_in = np.arange(num_elements, dtype=np.float32)
    h_out = np.zeros_like(h_in)

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(h_in)

    grid_x = (MATRIX_COLS + BLOCK_X - 1) // BLOCK_X
    grid_y = (MATRIX_ROWS + BLOCK_Y - 1) // BLOCK_Y

    start = time.time()
    matrix_transpose_kernel[(grid_x, grid_y), (BLOCK_X, BLOCK_Y)](
        d_in, d_out, MATRIX_COLS, MATRIX_ROWS)
    cuda.synchronize()
    gpu_time = (time.time() - start) * 1000

    h_out = d_out.copy_to_host()

    # CPU 验证
    start = time.time()
    h_out_ref = np.zeros_like(h_in)
    for row in range(MATRIX_ROWS):
        for col in range(MATRIX_COLS):
            h_out_ref[col * MATRIX_ROWS + row] = h_in[row * MATRIX_COLS + col]
    cpu_time = (time.time() - start) * 1000

    max_diff = np.max(np.abs(h_out - h_out_ref))
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
