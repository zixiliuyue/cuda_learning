#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumPy向量化对比测试
对比纯Python循环、NumPy向量化、广播操作的性能
"""

import time
from typing import Dict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("警告: NumPy未安装，将跳过此benchmark")


def pure_python_matrix_multiply(a: list, b: list) -> list:
    """纯Python矩阵乘法"""
    n = len(a)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if a[i][k] == 0:
                continue
            for j in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result


def pure_python_element_wise(a: list, b: list) -> list:
    """纯Python逐元素运算"""
    return [[x * y + x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


if NUMPY_AVAILABLE:
    def numpy_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """NumPy矩阵乘法"""
        return np.dot(a, b)

    def numpy_element_wise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """NumPy逐元素向量化运算"""
        return a * b + a + b

    def numpy_broadcast(a: np.ndarray, scalar: float) -> np.ndarray:
        """NumPy广播运算"""
        return a * scalar + np.sin(a) + np.cos(a)


def run_benchmark() -> Dict:
    """运行NumPy benchmark"""
    print("=" * 60)
    print("NumPy向量化对比测试")
    print("=" * 60)
    
    size = 200
    
    results = {
        "category": "NumPy向量化",
        "tests": []
    }
    
    if not NUMPY_AVAILABLE:
        print("NumPy未安装，跳过此测试")
        print()
        return results
    
    # 准备数据
    print(f"准备数据 (矩阵大小: {size}x{size})...\n")
    np.random.seed(42)
    np_a = np.random.rand(size, size).astype(np.float64)
    np_b = np.random.rand(size, size).astype(np.float64)
    py_a = np_a.tolist()
    py_b = np_b.tolist()
    
    # 逐元素运算 - 纯Python
    start = time.perf_counter()
    _ = pure_python_element_wise(py_a, py_b)
    py_elem_time = time.perf_counter() - start
    print(f"纯Python逐元素: {py_elem_time:.4f} 秒")
    results["tests"].append({
        "name": "纯Python逐元素",
        "time": py_elem_time,
        "speedup": 1.0
    })
    
    # 逐元素运算 - NumPy向量化
    start = time.perf_counter()
    _ = numpy_element_wise(np_a, np_b)
    np_elem_time = time.perf_counter() - start
    speedup = py_elem_time / np_elem_time
    print(f"NumPy逐元素向量化: {np_elem_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "NumPy逐元素向量化",
        "time": np_elem_time,
        "speedup": speedup
    })
    
    # 矩阵乘法 - 纯Python
    start = time.perf_counter()
    _ = pure_python_matrix_multiply(py_a, py_b)
    py_matmul_time = time.perf_counter() - start
    print(f"\n纯Python矩阵乘法: {py_matmul_time:.4f} 秒")
    results["tests"].append({
        "name": "纯Python矩阵乘法",
        "time": py_matmul_time,
        "speedup": 1.0
    })
    
    # 矩阵乘法 - NumPy
    start = time.perf_counter()
    _ = numpy_matrix_multiply(np_a, np_b)
    np_matmul_time = time.perf_counter() - start
    speedup = py_matmul_time / np_matmul_time
    print(f"NumPy矩阵乘法: {np_matmul_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "NumPy矩阵乘法",
        "time": np_matmul_time,
        "speedup": speedup
    })
    
    # 广播运算
    start = time.perf_counter()
    _ = numpy_broadcast(np_a, 2.5)
    broadcast_time = time.perf_counter() - start
    print(f"\nNumPy广播运算: {broadcast_time:.4f} 秒")
    results["tests"].append({
        "name": "NumPy广播运算",
        "time": broadcast_time,
        "speedup": py_elem_time / broadcast_time
    })
    
    print()
    return results


if __name__ == "__main__":
    run_benchmark()
