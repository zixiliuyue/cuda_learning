#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba加速对比测试
对比纯Python、Numba JIT编译、Numba并行模式的性能
"""

import time
from typing import Dict

try:
    import numpy as np
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("警告: Numba未安装，将跳过此benchmark")


def pure_python_calc(size: int) -> float:
    """纯Python计算"""
    result = 0.0
    for i in range(size):
        result += (i * 0.1) ** 2 + (i * 0.2) ** 3
    return result


if NUMBA_AVAILABLE:
    @jit(nopython=False)
    def numba_jit_calc(size: int) -> float:
        """Numba JIT编译"""
        result = 0.0
        for i in range(size):
            result += (i * 0.1) ** 2 + (i * 0.2) ** 3
        return result

    @njit
    def numba_njit_calc(size: int) -> float:
        """Numba nopython模式"""
        result = 0.0
        for i in range(size):
            result += (i * 0.1) ** 2 + (i * 0.2) ** 3
        return result

    @njit(parallel=True)
    def numba_parallel_calc(size: int) -> float:
        """Numba并行模式"""
        result = 0.0
        for i in prange(size):
            result += (i * 0.1) ** 2 + (i * 0.2) ** 3
        return result


def run_benchmark() -> Dict:
    """运行Numba benchmark"""
    print("=" * 60)
    print("Numba加速对比测试")
    print("=" * 60)
    
    size = 5_000_000
    
    results = {
        "category": "Numba加速",
        "tests": []
    }
    
    if not NUMBA_AVAILABLE:
        print("Numba未安装，跳过此测试")
        print()
        return results
    
    # 预热Numba
    print("预热Numba...")
    numba_jit_calc(100)
    numba_njit_calc(100)
    numba_parallel_calc(100)
    print("预热完成\n")
    
    # 纯Python
    start = time.perf_counter()
    _ = pure_python_calc(size)
    pure_time = time.perf_counter() - start
    print(f"纯Python: {pure_time:.4f} 秒")
    results["tests"].append({
        "name": "纯Python",
        "time": pure_time,
        "speedup": 1.0
    })
    
    # Numba JIT
    start = time.perf_counter()
    _ = numba_jit_calc(size)
    jit_time = time.perf_counter() - start
    speedup = pure_time / jit_time
    print(f"Numba JIT: {jit_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "Numba JIT",
        "time": jit_time,
        "speedup": speedup
    })
    
    # Numba NJIT
    start = time.perf_counter()
    _ = numba_njit_calc(size)
    njit_time = time.perf_counter() - start
    speedup = pure_time / njit_time
    print(f"Numba NJIT: {njit_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "Numba NJIT",
        "time": njit_time,
        "speedup": speedup
    })
    
    # Numba Parallel
    start = time.perf_counter()
    _ = numba_parallel_calc(size)
    parallel_time = time.perf_counter() - start
    speedup = pure_time / parallel_time
    print(f"Numba Parallel: {parallel_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "Numba Parallel",
        "time": parallel_time,
        "speedup": speedup
    })
    
    print()
    return results


if __name__ == "__main__":
    run_benchmark()
