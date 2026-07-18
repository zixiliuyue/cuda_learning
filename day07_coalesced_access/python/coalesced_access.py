#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 7: 全局内存合并访问 — Python/Numba 版本

对比行优先 vs 列优先内存访问的性能差异，
演示全局内存合并访问（Coalesced Access）的重要性。
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
ROWS = 2048
COLS = 2048
BLOCK_SIZE = 256
TOLERANCE = 1e-5
ITERATIONS = 10


# =============================================================================
# Kernel: 行优先读取 — 合并访问（高效）
# =============================================================================
@cuda.jit
def sum_rows_kernel(data, result, rows, cols):
    """
    行优先读取：线程按行方向读取连续内存地址。
    同一 warp 内的线程访问相邻地址，形成合并访问。
    """
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < rows:
        row_sum = 0.0
        for col in range(cols):
            row_sum += data[row * cols + col]
        result[row] = row_sum


# =============================================================================
# Kernel: 列优先读取 — 非合并访问（低效）
# =============================================================================
@cuda.jit
def sum_cols_kernel(data, result, cols, rows):
    """
    列优先读取：线程按列方向读取不连续内存地址。
    同一 warp 内的线程访问间隔 cols 的地址，无法合并。
    """
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if col < cols:
        col_sum = 0.0
        for row in range(rows):
            col_sum += data[row * cols + col]
        result[col] = col_sum


# =============================================================================
# CPU 参考实现
# =============================================================================
def sum_rows_cpu(data, rows, cols):
    """CPU 行求和"""
    return np.sum(data, axis=1)


def sum_cols_cpu(data, rows, cols):
    """CPU 列求和"""
    return np.sum(data, axis=0)


def main():
    print()
    print("=" * 60)
    print("     Day 7: 全局内存合并访问 — Python/Numba 版本            ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    # 数据准备
    total = ROWS * COLS
    print(f"[主机] 矩阵尺寸: {ROWS} x {COLS} ({total/1e6:.1f}M 元素)")
    h_data = np.random.uniform(-1.0, 1.0, size=(ROWS, COLS)).astype(np.float32)
    h_data_flat = h_data.flatten()

    # GPU 数据传输
    d_data = cuda.to_device(h_data_flat)
    d_result = cuda.device_array(ROWS, dtype=np.float32)

    # Grid 配置
    grid_size = (ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE
    print(f"[主机] Grid: {grid_size} Blocks, Block: {BLOCK_SIZE} 线程")

    # ---- 行优先读取（合并访问）----
    print()
    print("[主机] 测试行优先读取（合并访问）...")
    start = time.time()
    for _ in range(ITERATIONS):
        sum_rows_kernel[grid_size, BLOCK_SIZE](d_data, d_result, ROWS, COLS)
        cuda.synchronize()
    row_time = (time.time() - start) / ITERATIONS * 1000
    h_row_result = d_result.copy_to_host()

    # ---- 列优先读取（非合并访问）----
    d_result2 = cuda.device_array(COLS, dtype=np.float32)
    grid_size2 = (COLS + BLOCK_SIZE - 1) // BLOCK_SIZE
    print("[主机] 测试列优先读取（非合并访问）...")
    start = time.time()
    for _ in range(ITERATIONS):
        sum_cols_kernel[grid_size2, BLOCK_SIZE](d_data, d_result2, COLS, ROWS)
        cuda.synchronize()
    col_time = (time.time() - start) / ITERATIONS * 1000
    h_col_result = d_result2.copy_to_host()

    # CPU 参考计算
    start = time.time()
    cpu_row_ref = sum_rows_cpu(h_data, ROWS, COLS)
    cpu_col_ref = sum_cols_cpu(h_data, ROWS, COLS)
    cpu_time = (time.time() - start) * 1000

    # 验证结果
    row_diff = np.max(np.abs(h_row_result - cpu_row_ref))
    col_diff = np.max(np.abs(h_col_result - cpu_col_ref))
    row_ok = row_diff < TOLERANCE
    col_ok = col_diff < TOLERANCE

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  行优先（合并访问）   : {row_time:.4f} ms")
    print(f"  列优先（非合并访问） : {col_time:.4f} ms")
    if col_time > 0:
        print(f"  合并访问加速比       : {col_time / row_time:.2f} x")
    print(f"  CPU 计算时间         : {cpu_time:.4f} ms")
    print(f"  行求和验证           : {'通过' if row_ok else '失败'} (误差: {row_diff:.2e})")
    print(f"  列求和验证           : {'通过' if col_ok else '失败'} (误差: {col_diff:.2e})")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 合并访问: 同一 warp 内线程访问连续内存地址")
    print("  2. 行优先读取利用了内存的连续性，带宽利用率高")
    print("  3. 列优先读取导致跨步访问，浪费大量内存带宽")
    print("  4. 合并访问可以将性能提升数倍至数十倍")
    print("  5. 设计 Kernel 时应优先考虑内存访问模式")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
