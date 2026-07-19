#!/usr/bin/env python3
"""
Numba JIT 性能对比基准测试
包含：蒙特卡洛π计算、矩阵乘法、数组求和等场景
"""

import numpy as np
from numba import jit, njit, prange, vectorize, float64
import time
from typing import Callable, Dict, List


def benchmark(func: Callable, *args, runs: int = 5, warmup: bool = True) -> Dict:
    """
    基准测试函数
    
    Args:
        func: 要测试的函数
        *args: 函数参数
        runs: 运行次数
        warmup: 是否预热（用于JIT编译）
        
    Returns:
        包含测试结果的字典
    """
    if warmup:
        func(*args)
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000,
        'result': result
    }


def print_results(title: str, results: Dict, base_time: float = None):
    """打印测试结果"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  平均时间: {results['mean']:.2f} ms")
    print(f"  标准差:   {results['std']:.2f} ms")
    print(f"  最小时间: {results['min']:.2f} ms")
    print(f"  最大时间: {results['max']:.2f} ms")
    
    if base_time is not None and base_time > 0:
        speedup = base_time / results['mean']
        print(f"  加速比:   {speedup:.2f}x")
    
    print(f"{'='*60}")


# ==================== 测试场景1: 蒙特卡洛 π 计算 ====================

def monte_carlo_pi_python(n: int) -> float:
    """纯Python实现"""
    count = 0
    for _ in range(n):
        x = np.random.random()
        y = np.random.random()
        if x * x + y * y <= 1.0:
            count += 1
    return 4.0 * count / n


@jit
def monte_carlo_pi_jit(n: int) -> float:
    """@jit 装饰器（可能降级到object模式）"""
    count = 0
    for _ in range(n):
        x = np.random.random()
        y = np.random.random()
        if x * x + y * y <= 1.0:
            count += 1
    return 4.0 * count / n


@njit
def monte_carlo_pi_njit(n: int) -> float:
    """@njit 装饰器（强制nopython模式）"""
    count = 0
    for _ in range(n):
        x = np.random.random()
        y = np.random.random()
        if x * x + y * y <= 1.0:
            count += 1
    return 4.0 * count / n


@njit(parallel=True)
def monte_carlo_pi_parallel(n: int) -> float:
    """@njit + 并行"""
    count = 0
    for i in prange(n):
        x = np.random.random()
        y = np.random.random()
        if x * x + y * y <= 1.0:
            count += 1
    return 4.0 * count / n


def monte_carlo_pi_numpy(n: int) -> float:
    """NumPy 向量化实现"""
    x = np.random.random(n)
    y = np.random.random(n)
    count = np.sum(x * x + y * y <= 1.0)
    return 4.0 * count / n


# ==================== 测试场景2: 矩阵乘法 ====================

def matmul_python(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """纯Python矩阵乘法"""
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for l in range(k):
                C[i, j] += A[i, l] * B[l, j]
    return C


@njit
def matmul_njit(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Numba矩阵乘法"""
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for l in range(k):
                C[i, j] += A[i, l] * B[l, j]
    return C


@njit(parallel=True)
def matmul_parallel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Numba并行矩阵乘法"""
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    C = np.zeros((n, m))
    for i in prange(n):
        for j in range(m):
            for l in range(k):
                C[i, j] += A[i, l] * B[l, j]
    return C


def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """NumPy矩阵乘法"""
    return A @ B


# ==================== 测试场景3: 数组求和 ====================

def sum_python(arr: np.ndarray) -> float:
    """纯Python求和"""
    total = 0.0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total


@njit
def sum_njit(arr: np.ndarray) -> float:
    """Numba求和"""
    total = 0.0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total


@njit(parallel=True)
def sum_parallel(arr: np.ndarray) -> float:
    """Numba并行求和"""
    total = 0.0
    for i in prange(arr.shape[0]):
        total += arr[i]
    return total


def sum_numpy(arr: np.ndarray) -> float:
    """NumPy求和"""
    return np.sum(arr)


# ==================== 测试场景4: @vectorize 向量化 ====================

@vectorize([float64(float64)])
def vectorize_demo(x: float) -> float:
    """向量化函数"""
    return np.sin(x) ** 2 + np.cos(x) ** 2


@vectorize([float64(float64, float64)], target='cpu')
def vectorize_cpu(a: float, b: float) -> float:
    """CPU向量化"""
    return np.sqrt(a ** 2 + b ** 2)


@vectorize([float64(float64, float64)], target='parallel')
def vectorize_parallel(a: float, b: float) -> float:
    """多核并行向量化"""
    return np.sqrt(a ** 2 + b ** 2)


# ==================== 主测试函数 ====================

def run_monte_carlo_benchmark():
    """运行蒙特卡洛测试"""
    print("\n" + "="*70)
    print("  测试场景1: 蒙特卡洛 π 计算")
    print("="*70)
    
    n = 1_000_000
    print(f"\n迭代次数: {n:,}")
    
    print("\n[1/5] Python 纯循环...")
    python_result = benchmark(monte_carlo_pi_python, n, runs=3, warmup=False)
    base_time = python_result['mean']
    print_results("Python 纯循环", python_result)
    
    print("\n[2/5] @jit 装饰器...")
    jit_result = benchmark(monte_carlo_pi_jit, n, runs=5)
    print_results("@jit 装饰器", jit_result, base_time)
    
    print("\n[3/5] @njit 装饰器...")
    njit_result = benchmark(monte_carlo_pi_njit, n, runs=5)
    print_results("@njit 装饰器", njit_result, base_time)
    
    print("\n[4/5] @njit + 并行...")
    parallel_result = benchmark(monte_carlo_pi_parallel, n, runs=5)
    print_results("@njit + 并行", parallel_result, base_time)
    
    print("\n[5/5] NumPy 向量化...")
    numpy_result = benchmark(monte_carlo_pi_numpy, n, runs=5)
    print_results("NumPy 向量化", numpy_result, base_time)
    
    return {
        'python': python_result,
        'jit': jit_result,
        'njit': njit_result,
        'parallel': parallel_result,
        'numpy': numpy_result
    }


def run_matmul_benchmark():
    """运行矩阵乘法测试"""
    print("\n" + "="*70)
    print("  测试场景2: 矩阵乘法")
    print("="*70)
    
    size = 100
    A = np.random.random((size, size)).astype(np.float64)
    B = np.random.random((size, size)).astype(np.float64)
    print(f"\n矩阵大小: {size}x{size}")
    
    print("\n[1/4] Python 纯循环...")
    python_result = benchmark(matmul_python, A, B, runs=2, warmup=False)
    base_time = python_result['mean']
    print_results("Python 纯循环", python_result)
    
    print("\n[2/4] @njit 装饰器...")
    njit_result = benchmark(matmul_njit, A, B, runs=5)
    print_results("@njit 装饰器", njit_result, base_time)
    
    print("\n[3/4] @njit + 并行...")
    parallel_result = benchmark(matmul_parallel, A, B, runs=5)
    print_results("@njit + 并行", parallel_result, base_time)
    
    print("\n[4/4] NumPy (BLAS)...")
    numpy_result = benchmark(matmul_numpy, A, B, runs=5)
    print_results("NumPy (BLAS)", numpy_result, base_time)
    
    return {
        'python': python_result,
        'njit': njit_result,
        'parallel': parallel_result,
        'numpy': numpy_result
    }


def run_sum_benchmark():
    """运行数组求和测试"""
    print("\n" + "="*70)
    print("  测试场景3: 数组求和")
    print("="*70)
    
    size = 10_000_000
    arr = np.random.random(size).astype(np.float64)
    print(f"\n数组大小: {size:,}")
    
    print("\n[1/4] Python 纯循环...")
    python_result = benchmark(sum_python, arr, runs=3, warmup=False)
    base_time = python_result['mean']
    print_results("Python 纯循环", python_result)
    
    print("\n[2/4] @njit 装饰器...")
    njit_result = benchmark(sum_njit, arr, runs=5)
    print_results("@njit 装饰器", njit_result, base_time)
    
    print("\n[3/4] @njit + 并行...")
    parallel_result = benchmark(sum_parallel, arr, runs=5)
    print_results("@njit + 并行", parallel_result, base_time)
    
    print("\n[4/4] NumPy sum...")
    numpy_result = benchmark(sum_numpy, arr, runs=5)
    print_results("NumPy sum", numpy_result, base_time)
    
    return {
        'python': python_result,
        'njit': njit_result,
        'parallel': parallel_result,
        'numpy': numpy_result
    }


def run_vectorize_benchmark():
    """运行向量化测试"""
    print("\n" + "="*70)
    print("  测试场景4: @vectorize 向量化")
    print("="*70)
    
    size = 1_000_000
    x = np.random.random(size).astype(np.float64)
    y = np.random.random(size).astype(np.float64)
    print(f"\n数组大小: {size:,}")
    
    print("\n[1/3] @vectorize (cpu)...")
    cpu_result = benchmark(vectorize_cpu, x, y, runs=5)
    print_results("@vectorize (cpu)", cpu_result)
    
    print("\n[2/3] @vectorize (parallel)...")
    parallel_result = benchmark(vectorize_parallel, x, y, runs=5)
    print_results("@vectorize (parallel)", parallel_result, cpu_result['mean'])
    
    print("\n[3/3] NumPy 原生...")
    def numpy_func(a, b):
        return np.sqrt(a ** 2 + b ** 2)
    numpy_result = benchmark(numpy_func, x, y, runs=5)
    print_results("NumPy 原生", numpy_result, cpu_result['mean'])
    
    return {
        'cpu': cpu_result,
        'parallel': parallel_result,
        'numpy': numpy_result
    }


def generate_summary_report(results: Dict):
    """生成汇总报告"""
    print("\n\n" + "="*80)
    print("  性能测试汇总报告")
    print("="*80)
    
    print("\n【蒙特卡洛 π 计算】")
    mc = results['monte_carlo']
    base_time = mc['python']['mean']
    print(f"  {'实现方式':<20} {'平均时间(ms)':<15} {'加速比':<10} {'每秒迭代':<15}")
    print(f"  {'-'*70}")
    print(f"  {'Python 纯循环':<20} {mc['python']['mean']:<15.2f} {1.0:<10.1f}x {int(1e6/mc['python']['mean']*1000):<15,}")
    print(f"  {'@jit':<20} {mc['jit']['mean']:<15.2f} {base_time/mc['jit']['mean']:<10.1f}x {int(1e6/mc['jit']['mean']*1000):<15,}")
    print(f"  {'@njit':<20} {mc['njit']['mean']:<15.2f} {base_time/mc['njit']['mean']:<10.1f}x {int(1e6/mc['njit']['mean']*1000):<15,}")
    print(f"  {'@njit + parallel':<20} {mc['parallel']['mean']:<15.2f} {base_time/mc['parallel']['mean']:<10.1f}x {int(1e6/mc['parallel']['mean']*1000):<15,}")
    print(f"  {'NumPy':<20} {mc['numpy']['mean']:<15.2f} {base_time/mc['numpy']['mean']:<10.1f}x {int(1e6/mc['numpy']['mean']*1000):<15,}")
    
    print("\n【矩阵乘法 (100x100)】")
    mm = results['matmul']
    base_time = mm['python']['mean']
    print(f"  {'实现方式':<20} {'平均时间(ms)':<15} {'加速比':<10}")
    print(f"  {'-'*55}")
    print(f"  {'Python 纯循环':<20} {mm['python']['mean']:<15.2f} {1.0:<10.1f}x")
    print(f"  {'@njit':<20} {mm['njit']['mean']:<15.2f} {base_time/mm['njit']['mean']:<10.1f}x")
    print(f"  {'@njit + parallel':<20} {mm['parallel']['mean']:<15.2f} {base_time/mm['parallel']['mean']:<10.1f}x")
    print(f"  {'NumPy (BLAS)':<20} {mm['numpy']['mean']:<15.2f} {base_time/mm['numpy']['mean']:<10.1f}x")
    
    print("\n【数组求和 (10M元素)】")
    sm = results['sum']
    base_time = sm['python']['mean']
    print(f"  {'实现方式':<20} {'平均时间(ms)':<15} {'加速比':<10}")
    print(f"  {'-'*55}")
    print(f"  {'Python 纯循环':<20} {sm['python']['mean']:<15.2f} {1.0:<10.1f}x")
    print(f"  {'@njit':<20} {sm['njit']['mean']:<15.2f} {base_time/sm['njit']['mean']:<10.1f}x")
    print(f"  {'@njit + parallel':<20} {sm['parallel']['mean']:<15.2f} {base_time/sm['parallel']['mean']:<10.1f}x")
    print(f"  {'NumPy sum':<20} {sm['numpy']['mean']:<15.2f} {base_time/sm['numpy']['mean']:<10.1f}x")
    
    print("\n" + "="*80)
    print("  关键发现:")
    print(f"  - @njit 相比纯 Python 获得 {int(base_time/sm['njit']['mean'])}x 左右的加速")
    print(f"  - 并行化在大计算量时可额外获得 {int(sm['njit']['mean']/sm['parallel']['mean'])}x 加速")
    print(f"  - 对于 BLAS 优化的操作（如矩阵乘法），NumPy 仍然最快")
    print("="*80)


def main():
    print("Numba JIT 性能基准测试")
    print("="*70)
    print(f"NumPy 版本: {np.__version__}")
    
    try:
        import numba
        print(f"Numba 版本: {numba.__version__}")
    except ImportError:
        print("警告: Numba 未安装，请先运行: pip install numba")
        return
    
    print(f"CPU 核心数: {numba.config.NUMBA_NUM_THREADS}")
    print("="*70)
    
    results = {}
    
    results['monte_carlo'] = run_monte_carlo_benchmark()
    results['matmul'] = run_matmul_benchmark()
    results['sum'] = run_sum_benchmark()
    results['vectorize'] = run_vectorize_benchmark()
    
    generate_summary_report(results)


if __name__ == '__main__':
    main()
