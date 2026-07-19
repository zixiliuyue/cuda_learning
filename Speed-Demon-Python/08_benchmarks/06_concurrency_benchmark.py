#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发性能对比测试
对比单线程、多线程、多进程、asyncio在CPU密集和IO密集任务的性能
"""

import time
import threading
import multiprocessing
import asyncio
from typing import Dict, List


def cpu_task(n: int) -> int:
    """CPU密集型任务"""
    result = 0
    for i in range(n):
        result += i * i
    return result


def io_task(duration: float) -> None:
    """模拟IO密集型任务"""
    time.sleep(duration)


def run_sequential_cpu(num_tasks: int, iterations: int) -> float:
    """顺序执行CPU密集任务"""
    start = time.perf_counter()
    for _ in range(num_tasks):
        cpu_task(iterations)
    return time.perf_counter() - start


def run_threading_cpu(num_tasks: int, iterations: int) -> float:
    """多线程执行CPU密集任务"""
    start = time.perf_counter()
    threads = []
    for _ in range(num_tasks):
        t = threading.Thread(target=cpu_task, args=(iterations,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


def run_multiprocessing_cpu(num_tasks: int, iterations: int) -> float:
    """多进程执行CPU密集任务"""
    start = time.perf_counter()
    processes = []
    for _ in range(num_tasks):
        p = multiprocessing.Process(target=cpu_task, args=(iterations,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    return time.perf_counter() - start


def run_sequential_io(num_tasks: int, duration: float) -> float:
    """顺序执行IO密集任务"""
    start = time.perf_counter()
    for _ in range(num_tasks):
        io_task(duration)
    return time.perf_counter() - start


def run_threading_io(num_tasks: int, duration: float) -> float:
    """多线程执行IO密集任务"""
    start = time.perf_counter()
    threads = []
    for _ in range(num_tasks):
        t = threading.Thread(target=io_task, args=(duration,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


async def async_io_task(duration: float) -> None:
    """异步IO任务"""
    await asyncio.sleep(duration)


async def run_asyncio_io(num_tasks: int, duration: float) -> float:
    """asyncio执行IO密集任务"""
    start = time.perf_counter()
    tasks = [async_io_task(duration) for _ in range(num_tasks)]
    await asyncio.gather(*tasks)
    return time.perf_counter() - start


def run_benchmark() -> Dict:
    """运行并发性能 benchmark"""
    print("=" * 60)
    print("并发性能对比测试")
    print("=" * 60)
    
    results = {
        "category": "并发性能",
        "tests": []
    }
    
    # CPU密集任务测试
    num_cpu_tasks = 4
    iterations_per_task = 2_000_000
    
    print(f"\n--- CPU密集任务 ({num_cpu_tasks}个任务, 每个{iterations_per_task}次迭代) ---")
    
    seq_cpu_time = run_sequential_cpu(num_cpu_tasks, iterations_per_task)
    print(f"顺序执行: {seq_cpu_time:.4f} 秒")
    results["tests"].append({
        "name": "CPU-顺序执行",
        "time": seq_cpu_time,
        "speedup": 1.0
    })
    
    thread_cpu_time = run_threading_cpu(num_cpu_tasks, iterations_per_task)
    speedup = seq_cpu_time / thread_cpu_time
    print(f"多线程: {thread_cpu_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "CPU-多线程",
        "time": thread_cpu_time,
        "speedup": speedup
    })
    
    mp_cpu_time = run_multiprocessing_cpu(num_cpu_tasks, iterations_per_task)
    speedup = seq_cpu_time / mp_cpu_time
    print(f"多进程: {mp_cpu_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "CPU-多进程",
        "time": mp_cpu_time,
        "speedup": speedup
    })
    
    # IO密集任务测试
    num_io_tasks = 10
    io_duration = 0.1
    
    print(f"\n--- IO密集任务 ({num_io_tasks}个任务, 每个{io_duration}秒) ---")
    
    seq_io_time = run_sequential_io(num_io_tasks, io_duration)
    print(f"顺序执行: {seq_io_time:.4f} 秒")
    results["tests"].append({
        "name": "IO-顺序执行",
        "time": seq_io_time,
        "speedup": 1.0
    })
    
    thread_io_time = run_threading_io(num_io_tasks, io_duration)
    speedup = seq_io_time / thread_io_time
    print(f"多线程: {thread_io_time:.4f} 秒 (加速比: {speedup:.2f}x)")
    results["tests"].append({
        "name": "IO-多线程",
        "time": thread_io_time,
        "speedup": speedup
    })
    
    try:
        async_io_time = asyncio.run(run_asyncio_io(num_io_tasks, io_duration))
        speedup = seq_io_time / async_io_time
        print(f"asyncio: {async_io_time:.4f} 秒 (加速比: {speedup:.2f}x)")
        results["tests"].append({
            "name": "IO-asyncio",
            "time": async_io_time,
            "speedup": speedup
        })
    except Exception as e:
        print(f"asyncio运行出错: {e}")
    
    print()
    return results


if __name__ == "__main__":
    run_benchmark()
