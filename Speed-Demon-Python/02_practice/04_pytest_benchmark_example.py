#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest-benchmark 使用示例
这个文件包含多个基准测试函数，用于比较不同实现的性能

运行方式:
    pytest 04_pytest_benchmark_example.py --benchmark-autosave
    pytest 04_pytest_benchmark_example.py --benchmark-histogram
    pytest 04_pytest_benchmark_example.py --benchmark-compare=0001
"""

import pytest
import math
import random
from collections import deque


# =============================================================================
# 第一组测试: 列表创建和操作
# =============================================================================

def create_list_with_append(n):
    """使用 append 创建列表"""
    result = []
    for i in range(n):
        result.append(i)
    return result


def create_list_with_comprehension(n):
    """使用列表推导式创建列表"""
    return [i for i in range(n)]


def create_list_with_range(n):
    """直接使用 list(range)"""
    return list(range(n))


def test_list_creation_append(benchmark):
    """基准测试: append 方式创建列表"""
    result = benchmark(create_list_with_append, 10000)
    assert len(result) == 10000
    assert result[-1] == 9999


def test_list_creation_comprehension(benchmark):
    """基准测试: 列表推导式创建列表"""
    result = benchmark(create_list_with_comprehension, 10000)
    assert len(result) == 10000
    assert result[-1] == 9999


def test_list_creation_range(benchmark):
    """基准测试: list(range) 创建列表"""
    result = benchmark(create_list_with_range, 10000)
    assert len(result) == 10000
    assert result[-1] == 9999


# =============================================================================
# 第二组测试: 字符串拼接
# =============================================================================

def string_concat_with_plus(strings):
    """使用 += 拼接字符串（慢）"""
    result = ""
    for s in strings:
        result += s
    return result


def string_concat_with_join(strings):
    """使用 ''.join() 拼接字符串（快）"""
    return ''.join(strings)


def string_concat_with_fstring(strings):
    """使用 f-string 格式化"""
    result = ""
    for s in strings:
        result = f"{result}{s}"
    return result


@pytest.fixture
def sample_strings():
    """提供测试用的字符串列表"""
    return [f"string_{i:05d}_test" for i in range(1000)]


def test_string_concat_plus(benchmark, sample_strings):
    """基准测试: += 字符串拼接"""
    result = benchmark(string_concat_with_plus, sample_strings)
    assert len(result) == sum(len(s) for s in sample_strings)


def test_string_concat_join(benchmark, sample_strings):
    """基准测试: ''.join() 字符串拼接"""
    result = benchmark(string_concat_with_join, sample_strings)
    assert len(result) == sum(len(s) for s in sample_strings)


def test_string_concat_fstring(benchmark, sample_strings):
    """基准测试: f-string 字符串拼接"""
    result = benchmark(string_concat_with_fstring, sample_strings)
    assert len(result) == sum(len(s) for s in sample_strings)


# =============================================================================
# 第三组测试: 数学计算
# =============================================================================

def math_loop_naive(numbers):
    """朴素循环计算"""
    result = []
    for x in numbers:
        val = math.sqrt(x) + math.sin(x) + math.cos(x) + math.tan(x / 1000)
        result.append(val)
    return result


def math_loop_optimized(numbers):
    """优化版本: 局部变量绑定"""
    sqrt = math.sqrt
    sin = math.sin
    cos = math.cos
    tan = math.tan
    result = []
    append = result.append
    for x in numbers:
        append(sqrt(x) + sin(x) + cos(x) + tan(x / 1000))
    return result


def math_list_comprehension(numbers):
    """使用列表推导式"""
    return [
        math.sqrt(x) + math.sin(x) + math.cos(x) + math.tan(x / 1000)
        for x in numbers
    ]


@pytest.fixture
def sample_numbers():
    """提供测试用的数字列表"""
    return [float(i) for i in range(10000)]


@pytest.mark.benchmark(
    group="math-operations",
    min_rounds=50,
    warmup=True,
)
def test_math_naive_loop(benchmark, sample_numbers):
    """基准测试: 朴素循环数学计算"""
    result = benchmark(math_loop_naive, sample_numbers)
    assert len(result) == len(sample_numbers)


@pytest.mark.benchmark(
    group="math-operations",
    min_rounds=50,
    warmup=True,
)
def test_math_optimized_loop(benchmark, sample_numbers):
    """基准测试: 优化循环数学计算"""
    result = benchmark(math_loop_optimized, sample_numbers)
    assert len(result) == len(sample_numbers)


@pytest.mark.benchmark(
    group="math-operations",
    min_rounds=50,
    warmup=True,
)
def test_math_list_comprehension(benchmark, sample_numbers):
    """基准测试: 列表推导式数学计算"""
    result = benchmark(math_list_comprehension, sample_numbers)
    assert len(result) == len(sample_numbers)


# =============================================================================
# 第四组测试: 数据结构性能
# =============================================================================

def sum_with_list(n):
    """使用列表存储后求和"""
    data = [i for i in range(n)]
    return sum(data)


def sum_with_generator(n):
    """使用生成器求和（节省内存）"""
    return sum(i for i in range(n))


def sum_with_deque(n):
    """使用 deque 存储后求和"""
    data = deque(range(n))
    return sum(data)


@pytest.mark.benchmark(group="data-structures")
def test_sum_with_list(benchmark):
    """基准测试: 列表求和"""
    result = benchmark(sum_with_list, 100000)
    assert result == sum(range(100000))


@pytest.mark.benchmark(group="data-structures")
def test_sum_with_generator(benchmark):
    """基准测试: 生成器求和"""
    result = benchmark(sum_with_generator, 100000)
    assert result == sum(range(100000))


@pytest.mark.benchmark(group="data-structures")
def test_sum_with_deque(benchmark):
    """基准测试: deque 求和"""
    result = benchmark(sum_with_deque, 100000)
    assert result == sum(range(100000))


# =============================================================================
# 第五组测试: 查找操作
# =============================================================================

def lookup_in_list(needle, haystack):
    """在列表中查找"""
    return needle in haystack


def lookup_in_set(needle, haystack_set):
    """在集合中查找"""
    return needle in haystack_set


def lookup_in_dict(needle, haystack_dict):
    """在字典中查找"""
    return needle in haystack_dict


@pytest.fixture
def large_haystack():
    """提供一个大的搜索范围"""
    return list(range(100000))


@pytest.fixture
def large_haystack_set(large_haystack):
    """转换为集合"""
    return set(large_haystack)


@pytest.fixture
def large_haystack_dict(large_haystack):
    """转换为字典"""
    return {x: True for x in large_haystack}


@pytest.mark.benchmark(group="lookup-operations")
def test_lookup_in_list(benchmark, large_haystack):
    """基准测试: 列表查找（O(n)）"""
    # 查找一个不存在的元素，触发最坏情况
    result = benchmark(lookup_in_list, -1, large_haystack)
    assert result is False


@pytest.mark.benchmark(group="lookup-operations")
def test_lookup_in_set(benchmark, large_haystack_set):
    """基准测试: 集合查找（O(1)）"""
    result = benchmark(lookup_in_set, -1, large_haystack_set)
    assert result is False


@pytest.mark.benchmark(group="lookup-operations")
def test_lookup_in_dict(benchmark, large_haystack_dict):
    """基准测试: 字典查找（O(1)）"""
    result = benchmark(lookup_in_dict, -1, large_haystack_dict)
    assert result is False


# =============================================================================
# 额外示例: 使用 benchmark 上下文管理器
# =============================================================================

def test_benchmark_context_manager(benchmark):
    """使用 benchmark 作为上下文管理器"""
    
    def setup():
        """准备测试数据 - 在计时外执行"""
        data = [random.random() for _ in range(1000)]
        return (data,), {}
    
    def process(data):
        """被测试的函数"""
        return sorted([x ** 2 + math.sqrt(x) for x in data])
    
    # 使用 setup 参数在计时外准备数据
    result = benchmark.pedantic(process, setup=setup, rounds=100)
    assert len(result) == 1000


# =============================================================================
# 运行提示
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("pytest-benchmark 示例")
    print("=" * 60)
    print("\n运行方法:")
    print("  pytest 04_pytest_benchmark_example.py --benchmark-autosave")
    print("\n高级运行:")
    print("  pytest 04_pytest_benchmark_example.py --benchmark-columns=mean,median")
    print("  pytest 04_pytest_benchmark_example.py --benchmark-histogram")
    print("  pytest 04_pytest_benchmark_example.py --benchmark-compare=0001")
    print("  pytest 04_pytest_benchmark_example.py --benchmark-json=results.json")
    print("\n按组运行:")
    print("  pytest 04_pytest_benchmark_example.py -k 'math'")
    print("  pytest 04_pytest_benchmark_example.py -k 'string'")
    print("=" * 60)
