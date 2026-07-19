#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumPy向量化性能基准测试
========================
包含7个核心测试场景，对应02_numpy_vectorization.md中的内容
"""

import time
import random
import numpy as np
from typing import Callable, Dict, List, Tuple


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self, name: str, warmup: int = 2, iterations: int = 10):
        self.name = name
        self.warmup = warmup
        self.iterations = iterations
        self.times = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def run(self, func: Callable, *args, **kwargs):
        """运行函数并计时"""
        # 预热
        for _ in range(self.warmup):
            func(*args, **kwargs)
        
        # 正式计时
        self.times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            self.times.append((end - start) * 1000)  # 转换为毫秒
        
        return self
    
    @property
    def mean(self) -> float:
        return np.mean(self.times)
    
    @property
    def min(self) -> float:
        return np.min(self.times)
    
    @property
    def max(self) -> float:
        return np.max(self.times)
    
    @property
    def std(self) -> float:
        return np.std(self.times)


def print_section(title: str):
    """打印测试章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(test_name: str, timer_python: PerformanceTimer = None, 
                 timer_numpy: PerformanceTimer = None, 
                 extra: Dict[str, float] = None):
    """打印测试结果"""
    print(f"\n{test_name}:")
    print("-" * 60)
    
    if timer_python:
        print(f"  Python: {timer_python.mean:.4f} ms "
              f"(min: {timer_python.min:.4f}, max: {timer_python.max:.4f})")
    
    if timer_numpy:
        print(f"  NumPy:  {timer_numpy.mean:.4f} ms "
              f"(min: {timer_numpy.min:.4f}, max: {timer_numpy.max:.4f})")
    
    if timer_python and timer_numpy:
        speedup = timer_python.mean / timer_numpy.mean
        print(f"  加速比: {speedup:.2f}x")
    
    if extra:
        for key, value in extra.items():
            print(f"  {key}: {value}")


# ============================================================================
# 测试1: 向量化计算原理 - 消除Python循环
# ============================================================================
def test_vectorization_principle():
    print_section("测试1: 向量化计算原理 - 消除Python循环")
    
    size = 1_000_000
    data_np = np.random.rand(size)
    data_list = data_np.tolist()
    
    # Python循环求和
    def python_sum(arr):
        result = 0
        for x in arr:
            result += x
        return result
    
    # NumPy向量化求和
    def numpy_sum(arr):
        return arr.sum()
    
    # Python循环逐元素平方
    def python_square(arr):
        return [x ** 2 for x in arr]
    
    # NumPy向量化逐元素平方
    def numpy_square(arr):
        return arr ** 2
    
    # 测试1: 求和
    timer_python = PerformanceTimer("Python Sum").run(python_sum, data_list)
    timer_numpy = PerformanceTimer("NumPy Sum").run(numpy_sum, data_np)
    print_result("求和测试 (100万元素)", timer_python, timer_numpy)
    
    # 测试2: 逐元素平方
    timer_python2 = PerformanceTimer("Python Square").run(python_square, data_list)
    timer_numpy2 = PerformanceTimer("NumPy Square").run(numpy_square, data_np)
    print_result("逐元素平方测试 (100万元素)", timer_python2, timer_numpy2)


# ============================================================================
# 测试2: 广播机制性能
# ============================================================================
def test_broadcast_performance():
    print_section("测试2: 广播机制性能")
    
    rows, cols = 10000, 1000
    a = np.random.rand(rows, cols)
    b = np.random.rand(cols)
    
    # 非广播方式：使用tile显式扩展
    def no_broadcast():
        b_expanded = np.tile(b, (rows, 1))
        return a + b_expanded
    
    # 广播方式：自动广播
    def with_broadcast():
        return a + b
    
    timer_no_br = PerformanceTimer("No Broadcast").run(no_broadcast)
    timer_with_br = PerformanceTimer("With Broadcast").run(with_broadcast)
    
    # 计算内存使用
    memory_no_br = rows * cols * 8 / (1024 * 1024)  # MB
    memory_with_br = 0
    
    print_result("广播 vs 显式扩展", timer_no_br, timer_with_br,
                 {"额外内存节省": f"{memory_no_br:.2f} MB"})


# ============================================================================
# 测试3: ufunc性能
# ============================================================================
def test_ufunc_performance():
    print_section("测试3: 通用函数ufunc性能")
    
    size = 1_000_000
    a = np.random.rand(size)
    b = np.random.rand(size)
    a_list = a.tolist()
    b_list = b.tolist()
    
    # Python循环计算
    def python_math_operation(a_list, b_list):
        result = []
        for x, y in zip(a_list, b_list):
            result.append(math.sin(x) * math.exp(y) + math.sqrt(x * y))
        return result
    
    # NumPy ufunc
    def numpy_math_operation(a, b):
        return np.sin(a) * np.exp(b) + np.sqrt(a * b)
    
    import math
    
    timer_python = PerformanceTimer("Python Math", iterations=5).run(
        python_math_operation, a_list, b_list)
    timer_numpy = PerformanceTimer("NumPy Math", iterations=5).run(
        numpy_math_operation, a, b)
    
    print_result("复杂数学运算 (sin + exp + sqrt)", timer_python, timer_numpy)


# ============================================================================
# 测试4: 视图 vs 拷贝
# ============================================================================
def test_view_vs_copy():
    print_section("测试4: 视图 vs 拷贝")
    
    size = 2000
    a = np.random.rand(size, size)
    
    # 视图操作：切片
    def view_operation():
        return a[::2, ::2]
    
    # 拷贝操作：花式索引
    def copy_operation():
        rows = np.random.choice(size, size // 2)
        cols = np.random.choice(size, size // 2)
        return a[rows][:, cols]
    
    timer_view = PerformanceTimer("View Operation").run(view_operation)
    timer_copy = PerformanceTimer("Copy Operation").run(copy_operation)
    
    print_result("视图 vs 拷贝", timer_copy, timer_view)
    
    # 检查是否真的是视图
    slice_view = a[::2, ::2]
    is_view = slice_view.base is a
    print(f"\n  验证: 切片是否为视图? {is_view}")


# ============================================================================
# 测试5: 内存布局优化
# ============================================================================
def test_memory_layout():
    print_section("测试5: 内存布局优化 (C顺序 vs Fortran顺序)")
    
    size = 2000
    a_c = np.random.rand(size, size)  # 默认C顺序
    a_f = np.asfortranarray(a_c)      # Fortran顺序
    
    # 按行访问C顺序数组（快）
    def row_major_c():
        total = 0
        for i in range(size):
            total += a_c[i, :].sum()
        return total
    
    # 按列访问C顺序数组（慢）
    def col_major_c():
        total = 0
        for j in range(size):
            total += a_c[:, j].sum()
        return total
    
    # 按列访问Fortran顺序数组（快）
    def col_major_f():
        total = 0
        for j in range(size):
            total += a_f[:, j].sum()
        return total
    
    timer_row_c = PerformanceTimer("Row-major (C)", iterations=3).run(row_major_c)
    timer_col_c = PerformanceTimer("Col-major (C)", iterations=3).run(col_major_c)
    timer_col_f = PerformanceTimer("Col-major (F)", iterations=3).run(col_major_f)
    
    print(f"\n内存布局对访问性能的影响:")
    print("-" * 60)
    print(f"  按行访问(C顺序): {timer_row_c.mean:.4f} ms")
    print(f"  按列访问(C顺序): {timer_col_c.mean:.4f} ms")
    print(f"  按列访问(F顺序): {timer_col_f.mean:.4f} ms")
    print(f"  行/列加速比(C):  {timer_col_c.mean / timer_row_c.mean:.2f}x")
    print(f"  布局优化加速:     {timer_col_c.mean / timer_col_f.mean:.2f}x")


# ============================================================================
# 测试6: 花式索引性能坑
# ============================================================================
def test_fancy_indexing():
    print_section("测试6: 花式索引性能坑")
    
    size = 100000
    a = np.random.rand(size)
    n_indices = 10000
    
    indices = np.random.randint(0, size, n_indices)
    mask = a > 0.5
    
    # 切片索引（快）
    def slice_indexing():
        return a[:n_indices]
    
    # 花式索引（慢）
    def fancy_indexing():
        return a[indices]
    
    # np.take（比花式索引稍快）
    def take_indexing():
        return a.take(indices)
    
    # 重复布尔索引（性能坑）
    def repeated_boolean_indexing():
        result = []
        for _ in range(100):
            result.append(a[mask])
        return result
    
    # 优化：预先获取索引
    def optimized_boolean_indexing():
        indices_where = np.where(mask)[0]
        result = []
        for _ in range(100):
            result.append(a[indices_where])
        return result
    
    timer_slice = PerformanceTimer("Slice Indexing").run(slice_indexing)
    timer_fancy = PerformanceTimer("Fancy Indexing").run(fancy_indexing)
    timer_take = PerformanceTimer("Take Indexing").run(take_indexing)
    
    print(f"\n不同索引方式性能对比:")
    print("-" * 60)
    print(f"  切片索引:   {timer_slice.mean:.6f} ms")
    print(f"  花式索引:   {timer_fancy.mean:.6f} ms")
    print(f"  np.take:    {timer_take.mean:.6f} ms")
    print(f"  切片/花式:  {timer_fancy.mean / timer_slice.mean:.2f}x")
    
    # 测试重复布尔索引
    timer_repeated = PerformanceTimer("Repeated Boolean", iterations=3).run(
        repeated_boolean_indexing)
    timer_optimized = PerformanceTimer("Optimized Boolean", iterations=3).run(
        optimized_boolean_indexing)
    
    print(f"\n重复布尔索引优化:")
    print("-" * 60)
    print(f"  重复布尔索引: {timer_repeated.mean:.4f} ms")
    print(f"  优化后索引:   {timer_optimized.mean:.4f} ms")
    print(f"  加速比:        {timer_repeated.mean / timer_optimized.mean:.2f}x")


# ============================================================================
# 测试7: 真实案例 - VaR计算 (200x加速)
# ============================================================================
def test_var_calculation():
    print_section("测试7: 真实案例 - VaR计算 (金融风险计算)")
    
    n_historical = 100000
    n_assets = 50
    iterations = 10000
    
    # 生成模拟数据
    returns = np.random.randn(n_historical) * 0.01
    weights = np.random.rand(n_assets)
    weights = weights / weights.sum()
    
    returns_list = returns.tolist()
    weights_list = weights.tolist()
    
    # Python实现
    def python_var_calculation():
        portfolio_values = []
        for _ in range(iterations):
            scenario_returns = []
            for _ in range(n_assets):
                idx = random.randint(0, n_historical - 1)
                scenario_returns.append(returns_list[idx])
            portfolio_return = sum(r * w for r, w in zip(scenario_returns, weights_list))
            portfolio_values.append(portfolio_return)
        portfolio_values.sort()
        var_95 = portfolio_values[int(iterations * 0.05)]
        return var_95
    
    # NumPy向量化实现
    def numpy_var_calculation():
        random_indices = np.random.randint(0, n_historical, size=(iterations, n_assets))
        scenario_returns = returns[random_indices]
        portfolio_returns = scenario_returns @ weights
        var_95 = np.percentile(portfolio_returns, 5)
        return var_95
    
    # 优化的NumPy实现
    def optimized_var_calculation():
        random_indices = np.random.randint(0, n_historical, size=(iterations, n_assets))
        returns_contig = np.ascontiguousarray(returns)
        scenario_returns = np.take(returns_contig, random_indices)
        portfolio_returns = np.empty(iterations, dtype=np.float64)
        np.dot(scenario_returns, weights, out=portfolio_returns)
        portfolio_returns.sort()
        var_95 = portfolio_returns[int(iterations * 0.05)]
        return var_95
    
    # 运行测试（Python版本只运行1次，因为太慢）
    print("  运行Python实现... (可能需要几秒)")
    timer_python = PerformanceTimer("Python VaR", warmup=0, iterations=1).run(
        python_var_calculation)
    
    timer_numpy = PerformanceTimer("NumPy VaR", iterations=3).run(
        numpy_var_calculation)
    
    timer_optimized = PerformanceTimer("Optimized VaR", iterations=3).run(
        optimized_var_calculation)
    
    print(f"\nVaR计算性能对比:")
    print("-" * 60)
    print(f"  Python实现:      {timer_python.mean:.4f} ms")
    print(f"  NumPy实现:       {timer_numpy.mean:.4f} ms")
    print(f"  优化NumPy:       {timer_optimized.mean:.4f} ms")
    print(f"  NumPy加速比:     {timer_python.mean / timer_numpy.mean:.2f}x")
    print(f"  最终加速比:      {timer_python.mean / timer_optimized.mean:.2f}x")


# ============================================================================
# 综合性能汇总
# ============================================================================
def print_summary():
    print_section("NumPy向量化优化总结")
    
    summary = """
各优化技术的典型加速比:

  1. 消除Python循环:       100x - 1000x
     - 原因: 避免Python解释器开销，利用C级优化和SIMD指令
     
  2. 使用广播机制:          2x - 10x
     - 原因: 避免不必要的内存复制，节省内存带宽
     
  3. 使用ufunc函数:         5x - 50x
     - 原因: 高度优化的C/Fortran实现，支持SIMD
     
  4. 避免不必要的拷贝:      2x - 100x
     - 原因: 减少内存分配和数据复制开销
     
  5. 优化内存布局:          2x - 10x
     - 原因: 提高缓存命中率，减少内存访问延迟
     
  6. 避免花式索引:          5x - 50x
     - 原因: 避免随机内存访问和不必要的数据复制


关键性能指标建议:

  - Python循环占比:        < 5%
  - 内存拷贝次数:           最小化
  - 内存访问连续性:         最大化
  - 向量化程度:             > 95%


优化Checklist:

  [ ] 尽可能使用内置ufunc代替自定义函数
  [ ] 消除所有Python级别的循环
  [ ] 合理使用广播，避免不必要的数组扩展
  [ ] 检查是否有不必要的数组拷贝
  [ ] 根据访问模式选择合适的内存布局
  [ ] 尽量使用切片索引，避免花式索引
  [ ] 使用np.take, np.dot等高性能函数
  [ ] 预分配输出数组，避免动态分配
"""
    print(summary)


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "NumPy向量化性能基准测试" + " " * 32 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    # 打印环境信息
    print(f"\n环境信息:")
    print(f"  NumPy版本: {np.__version__}")
    print(f"  测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行所有测试
    test_vectorization_principle()
    test_broadcast_performance()
    test_ufunc_performance()
    test_view_vs_copy()
    test_memory_layout()
    test_fancy_indexing()
    test_var_calculation()
    
    # 打印总结
    print_summary()
    
    print("\n" + "=" * 80)
    print("  测试完成！详细说明请参考 02_numpy_vectorization.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
