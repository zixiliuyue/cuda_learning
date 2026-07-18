#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 13: cuBLAS 线性代数 — Python 版本

演示矩阵乘法的 GPU 加速实现。
优先使用 CuPy（封装 cuBLAS），不可用时 fallback 到 NumPy。
"""

import sys
import time
import numpy as np

# 尝试导入 CuPy（封装 cuBLAS）
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("[提示] 未安装 CuPy，将使用 NumPy fallback。")
    print("[提示] 安装 CuPy: pip install cupy-cuda12x (根据 CUDA 版本选择)")

# 常量定义
MATRIX_SIZE = 512
TOLERANCE = 1e-2
ITERATIONS = 5


def matmul_numpy(a, b):
    """NumPy 矩阵乘法（CPU，使用 BLAS）"""
    return a @ b


def matmul_cupy(a_np, b_np):
    """CuPy 矩阵乘法（GPU，封装 cuBLAS）"""
    a_gpu = cp.asarray(a_np)
    b_gpu = cp.asarray(b_np)
    c_gpu = cp.matmul(a_gpu, b_gpu)
    return cp.asnumpy(c_gpu)


def matmul_cupy_timing(a_np, b_np, iterations):
    """CuPy 矩阵乘法 — 带计时"""
    a_gpu = cp.asarray(a_np)
    b_gpu = cp.asarray(b_np)

    # 预热
    for _ in range(3):
        c = cp.matmul(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()

    # 计时
    start = time.time()
    for _ in range(iterations):
        c = cp.matmul(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()
    elapsed = (time.time() - start) / iterations * 1000

    return cp.asnumpy(c), elapsed


def main():
    print()
    print("=" * 60)
    print("     Day 13: cuBLAS 线性代数 — Python 版本                  ")
    print("=" * 60)
    print()

    # 数据准备
    print(f"[主机] 矩阵尺寸: {MATRIX_SIZE} x {MATRIX_SIZE}")
    h_a = np.random.uniform(-1.0, 1.0, size=(MATRIX_SIZE, MATRIX_SIZE)).astype(np.float32)
    h_b = np.random.uniform(-1.0, 1.0, size=(MATRIX_SIZE, MATRIX_SIZE)).astype(np.float32)

    if CUPY_AVAILABLE:
        print(f"[设备] 使用 CuPy (cuBLAS 后端)")
        # CuPy 矩阵乘法
        h_c_cupy, cupy_time = matmul_cupy_timing(h_a, h_b, ITERATIONS)
    else:
        print(f"[设备] 使用 NumPy (CPU fallback)")

    # NumPy 矩阵乘法
    print("[主机] 运行 NumPy 矩阵乘法...")
    # 预热
    for _ in range(3):
        _ = matmul_numpy(h_a, h_b)

    start = time.time()
    for _ in range(ITERATIONS):
        h_c_numpy = matmul_numpy(h_a, h_b)
    numpy_time = (time.time() - start) / ITERATIONS * 1000

    # 验证
    max_diff = np.max(np.abs(h_c_numpy - h_a @ h_b))
    if CUPY_AVAILABLE:
        cupy_diff = np.max(np.abs(h_c_cupy - h_a @ h_b))
        cupy_ok = cupy_diff < TOLERANCE
    else:
        cupy_diff = 0.0
        cupy_ok = True

    numpy_ok = max_diff < TOLERANCE

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    if CUPY_AVAILABLE:
        print(f"  CuPy (cuBLAS GPU): {cupy_time:.4f} ms")
        print(f"  NumPy (CPU)       : {numpy_time:.4f} ms")
        if cupy_time > 0:
            print(f"  GPU 加速比        : {numpy_time / cupy_time:.2f} x")
        print(f"  CuPy 验证         : {'通过' if cupy_ok else '失败'} (误差: {cupy_diff:.2e})")
    else:
        print(f"  NumPy (CPU)       : {numpy_time:.4f} ms")
        print(f"  CuPy              : 未安装（使用 NumPy fallback）")
    print(f"  NumPy 验证        : {'通过' if numpy_ok else '失败'} (误差: {max_diff:.2e})")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. cuBLAS: NVIDIA 高度优化的 BLAS 库（GPU）")
    print("  2. CuPy: Python 的 GPU 加速 NumPy 替代品")
    print("  3. cp.matmul() 内部调用 cuBLAS 的 sgemm/dgemm")
    print("  4. 矩阵乘法是 GPU 最擅长的操作之一")
    print("  5. cuBLAS 还支持: 向量运算、矩阵分解、求解线性方程组")
    print("  6. 大矩阵(>1024)上 GPU 优势更明显")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
