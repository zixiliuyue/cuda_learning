#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据结构性能对比测试
对比List, Set, Dict在常见操作下的性能差异
"""

import time
from typing import Dict, List


def benchmark_list_operations(size: int) -> Dict[str, float]:
    """测试List操作性能"""
    results = {}
    
    # 创建List
    start = time.perf_counter()
    lst = list(range(size))
    results["创建"] = time.perf_counter() - start
    
    # 查找元素（末尾）
    start = time.perf_counter()
    _ = size - 1 in lst
    results["查找(末尾)"] = time.perf_counter() - start
    
    # 查找元素（开头）
    start = time.perf_counter()
    _ = 0 in lst
    results["查找(开头)"] = time.perf_counter() - start
    
    # 追加元素
    start = time.perf_counter()
    lst.append(size)
    results["追加"] = time.perf_counter() - start
    
    # 头部插入
    start = time.perf_counter()
    lst.insert(0, -1)
    results["头部插入"] = time.perf_counter() - start
    
    return results


def benchmark_set_operations(size: int) -> Dict[str, float]:
    """测试Set操作性能"""
    results = {}
    
    # 创建Set
    start = time.perf_counter()
    s = set(range(size))
    results["创建"] = time.perf_counter() - start
    
    # 查找元素
    start = time.perf_counter()
    _ = size - 1 in s
    results["查找"] = time.perf_counter() - start
    
    # 添加元素
    start = time.perf_counter()
    s.add(size)
    results["添加"] = time.perf_counter() - start
    
    # 集合交集
    s2 = set(range(size // 2, size * 2))
    start = time.perf_counter()
    _ = s & s2
    results["交集"] = time.perf_counter() - start
    
    return results


def benchmark_dict_operations(size: int) -> Dict[str, float]:
    """测试Dict操作性能"""
    results = {}
    
    # 创建Dict
    start = time.perf_counter()
    d = {i: i * 2 for i in range(size)}
    results["创建"] = time.perf_counter() - start
    
    # 查找键
    start = time.perf_counter()
    _ = d.get(size - 1)
    results["查找键"] = time.perf_counter() - start
    
    # 设置值
    start = time.perf_counter()
    d[size] = size * 2
    results["设置值"] = time.perf_counter() - start
    
    # 遍历键
    start = time.perf_counter()
    _ = list(d.keys())
    results["遍历键"] = time.perf_counter() - start
    
    return results


def run_benchmark() -> Dict:
    """运行数据结构 benchmark"""
    print("=" * 60)
    print("数据结构性能对比测试")
    print("=" * 60)
    
    size = 1_000_000
    
    results = {
        "category": "数据结构性能",
        "tests": []
    }
    
    # List测试
    print("\n--- List 性能 ---")
    list_results = benchmark_list_operations(size)
    for op, t in list_results.items():
        print(f"{op}: {t * 1000:.4f} 毫秒")
        results["tests"].append({
            "name": f"List-{op}",
            "time": t,
            "speedup": 1.0
        })
    
    # Set测试
    print("\n--- Set 性能 ---")
    set_results = benchmark_set_operations(size)
    for op, t in set_results.items():
        print(f"{op}: {t * 1000:.4f} 毫秒")
        results["tests"].append({
            "name": f"Set-{op}",
            "time": t,
            "speedup": 1.0
        })
    
    # Dict测试
    print("\n--- Dict 性能 ---")
    dict_results = benchmark_dict_operations(size)
    for op, t in dict_results.items():
        print(f"{op}: {t * 1000:.4f} 毫秒")
        results["tests"].append({
            "name": f"Dict-{op}",
            "time": t,
            "speedup": 1.0
        })
    
    print()
    return results


if __name__ == "__main__":
    run_benchmark()
