#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
循环优化对比测试
对比纯Python循环、内置函数、列表推导式、生成器表达式的性能
"""

import time
from typing import Dict, List


def pure_python_loop(data: List[int]) -> float:
    """纯Python循环"""
    start = time.perf_counter()
    result = []
    for x in data:
        if x % 2 == 0:
            result.append(x * x)
    return time.perf_counter() - start


def list_comprehension(data: List[int]) -> float:
    """列表推导式"""
    start = time.perf_counter()
    result = [x * x for x in data if x % 2 == 0]
    return time.perf_counter() - start


def generator_expression(data: List[int]) -> float:
    """生成器表达式"""
    start = time.perf_counter()
    result = list(x * x for x in data if x % 2 == 0)
    return time.perf_counter() - start


def builtin_map_filter(data: List[int]) -> float:
    """内置函数 map + filter"""
    start = time.perf_counter()
    result = list(map(lambda x: x * x, filter(lambda x: x % 2 == 0, data)))
    return time.perf_counter() - start


def while_loop(data: List[int]) -> float:
    """while循环"""
    start = time.perf_counter()
    result = []
    i = 0
    n = len(data)
    while i < n:
        x = data[i]
        if x % 2 == 0:
            result.append(x * x)
        i += 1
    return time.perf_counter() - start


def run_benchmark() -> Dict:
    """运行循环优化 benchmark"""
    print("=" * 60)
    print("循环优化对比测试")
    print("=" * 60)
    
    size = 1_000_000
    data = list(range(size))
    
    results = {
        "category": "循环优化",
        "tests": []
    }
    
    # 纯Python循环
    pure_time = pure_python_loop(data)
    print(f"纯Python循环: {pure_time:.4f} 秒")
    results["tests"].append({
        "name": "纯Python循环",
        "time": pure_time,
        "speedup": 1.0
    })
    
    # while循环
    while_time = while_loop(data)
    speedup = pure_time / while_time
    print(f"while循环: {while_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "while循环",
        "time": while_time,
        "speedup": speedup
    })
    
    # 列表推导式
    list_comp_time = list_comprehension(data)
    speedup = pure_time / list_comp_time
    print(f"列表推导式: {list_comp_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "列表推导式",
        "time": list_comp_time,
        "speedup": speedup
    })
    
    # 生成器表达式
    gen_time = generator_expression(data)
    speedup = pure_time / gen_time
    print(f"生成器表达式: {gen_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "生成器表达式",
        "time": gen_time,
        "speedup": speedup
    })
    
    # 内置函数 map + filter
    builtin_time = builtin_map_filter(data)
    speedup = pure_time / builtin_time
    print(f"内置函数map+filter: {builtin_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "内置函数map+filter",
        "time": builtin_time,
        "speedup": speedup
    })
    
    print()
    return results


if __name__ == "__main__":
    run_benchmark()
