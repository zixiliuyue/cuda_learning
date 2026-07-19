#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIL性能对比测试
对比单线程CPU密集型任务在纯Python和多线程下的性能
"""

import time
import threading
from typing import List, Dict


def cpu_intensive_task(n: int) -> int:
    """CPU密集型计算任务"""
    result = 0
    for i in range(n):
        result += i * i
    return result


def run_single_thread(num_iterations: int) -> float:
    """单线程运行"""
    start_time = time.perf_counter()
    cpu_intensive_task(num_iterations)
    return time.perf_counter() - start_time


def run_multi_thread(num_iterations: int, num_threads: int) -> float:
    """多线程运行（受GIL限制）"""
    start_time = time.perf_counter()
    
    threads = []
    iterations_per_thread = num_iterations // num_threads
    
    for _ in range(num_threads):
        t = threading.Thread(target=cpu_intensive_task, args=(iterations_per_thread,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    return time.perf_counter() - start_time


def run_benchmark() -> Dict:
    """运行GIL benchmark"""
    print("=" * 60)
    print("GIL性能对比测试")
    print("=" * 60)
    
    num_iterations = 10_000_000
    
    results = {
        "category": "GIL性能",
        "tests": []
    }
    
    # 单线程
    single_time = run_single_thread(num_iterations)
    print(f"单线程: {single_time:.4f} 秒")
    results["tests"].append({
        "name": "单线程",
        "time": single_time,
        "speedup": 1.0
    })
    
    # 多线程（2线程）
    multi_time_2 = run_multi_thread(num_iterations, 2)
    speedup_2 = single_time / multi_time_2
    print(f"2线程: {multi_time_2:.4f} 秒 (加速比: {speedup_2:.2f}x)")
    results["tests"].append({
        "name": "2线程",
        "time": multi_time_2,
        "speedup": speedup_2
    })
    
    # 多线程（4线程）
    multi_time_4 = run_multi_thread(num_iterations, 4)
    speedup_4 = single_time / multi_time_4
    print(f"4线程: {multi_time_4:.4f} 秒 (加速比: {speedup_4:.2f}x)")
    results["tests"].append({
        "name": "4线程",
        "time": multi_time_4,
        "speedup": speedup_4
    })
    
    print()
    return results


if __name__ == "__main__":
    run_benchmark()
