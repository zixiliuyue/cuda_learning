#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发数选择策略测试
测试不同并发数（线程数/进程数）对性能的影响
帮助理解如何选择合适的并发数
"""

import time
import os
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def cpu_intensive_task(task_id, size=500000):
    """CPU 密集型任务"""
    result = 0
    for i in range(size):
        result += math.sqrt(i) * math.sin(i)
    return result


def io_intensive_task(task_id, wait_time=0.1):
    """I/O 密集型任务"""
    time.sleep(wait_time)
    return f"任务 {task_id} 完成"


def test_thread_pool_sizes(task_func, task_name, sizes, num_tasks=100, **kwargs):
    """测试不同线程池大小的性能"""
    print(f"\n{'='*70}")
    print(f"测试: {task_name} (线程池)")
    print(f"{'='*70}")
    print(f"任务总数: {num_tasks}")
    print(f"{'线程数':<10} {'总耗时(s)':<12} {'平均每任务(ms)':<18} {'加速比':<10}")
    print(f"{'-'*70}")
    
    results = []
    
    # 先测单线程作为基准
    start = time.time()
    for i in range(num_tasks):
        task_func(i, **kwargs)
    single_time = time.time() - start
    
    print(f"{'1 (单线程)':<10} {single_time:<12.2f} {single_time/num_tasks*1000:<18.2f} {'1.00x':<10}")
    results.append((1, single_time))
    
    # 测试不同线程数
    for size in sizes:
        if size == 1:
            continue
            
        start = time.time()
        with ThreadPoolExecutor(max_workers=size) as executor:
            list(executor.map(
                lambda i: task_func(i, **kwargs),
                range(num_tasks)
            ))
        elapsed = time.time() - start
        
        speedup = single_time / elapsed
        print(f"{size:<10} {elapsed:<12.2f} {elapsed/num_tasks*1000:<18.2f} {speedup:.2f}x")
        results.append((size, elapsed))
    
    return results


def test_process_pool_sizes(task_func, task_name, sizes, num_tasks=50, **kwargs):
    """测试不同进程池大小的性能"""
    print(f"\n{'='*70}")
    print(f"测试: {task_name} (进程池)")
    print(f"{'='*70}")
    print(f"任务总数: {num_tasks}")
    print(f"{'进程数':<10} {'总耗时(s)':<12} {'平均每任务(ms)':<18} {'加速比':<10}")
    print(f"{'-'*70}")
    
    results = []
    
    # 先测单进程作为基准
    start = time.time()
    for i in range(num_tasks):
        task_func(i, **kwargs)
    single_time = time.time() - start
    
    print(f"{'1 (单进程)':<10} {single_time:<12.2f} {single_time/num_tasks*1000:<18.2f} {'1.00x':<10}")
    results.append((1, single_time))
    
    # 测试不同进程数
    for size in sizes:
        if size == 1:
            continue
            
        start = time.time()
        with ProcessPoolExecutor(max_workers=size) as executor:
            list(executor.map(
                lambda i: task_func(i, **kwargs),
                range(num_tasks)
            ))
        elapsed = time.time() - start
        
        speedup = single_time / elapsed
        print(f"{size:<10} {elapsed:<12.2f} {elapsed/num_tasks*1000:<18.2f} {speedup:.2f}x")
        results.append((size, elapsed))
    
    return results


def analyze_results(results, cpu_count, task_type):
    """分析测试结果"""
    print(f"\n{'='*70}")
    print(f"结果分析: {task_type}")
    print(f"{'='*70}")
    
    best_size = min(results, key=lambda x: x[1])[0]
    best_time = min(results, key=lambda x: x[1])[1]
    base_time = results[0][1]
    
    print(f"系统 CPU 核心数: {cpu_count}")
    print(f"最佳并发数: {best_size}")
    print(f"最佳性能: {best_time:.2f} 秒")
    print(f"最大加速比: {base_time/best_time:.2f}x")
    
    print(f"\n性能随并发数变化:")
    for i in range(1, len(results)):
        prev_size, prev_time = results[i-1]
        curr_size, curr_time = results[i]
        change = (prev_time - curr_time) / prev_time * 100
        
        if change > 10:
            trend = "↗ 大幅提升"
        elif change > 0:
            trend = "↗ 略有提升"
        elif change > -10:
            trend = "→ 基本持平"
        else:
            trend = "↘ 明显下降"
        
        print(f"  {prev_size} → {curr_size}: 时间变化 {change:+.1f}% {trend}")
    
    return best_size


def print_recommendations(cpu_count, best_io_threads, best_cpu_processes):
    """打印并发数选择建议"""
    print(f"\n{'='*70}")
    print("并发数选择策略总结与建议")
    print(f"{'='*70}")
    
    recommendations = f"""
系统信息:
  - CPU 核心数: {cpu_count}

线程池大小建议 (ThreadPoolExecutor):
  1. I/O 密集型任务:
     - 理论公式: 线程数 = CPU 核心数 × (1 + I/O 等待时间 / 计算时间)
     - 实际测试最佳: {best_io_threads} 线程
     - 经验范围: {cpu_count * 2} ~ {cpu_count * 10} 线程
     - 原因: I/O 等待期间线程会释放 GIL，其他线程可以执行

  2. CPU 密集型任务:
     - 理论公式: 线程数 = CPU 核心数 或 CPU 核心数 + 1
     - 不建议使用多线程（受 GIL 限制，性能提升有限甚至下降）

进程池大小建议 (ProcessPoolExecutor):
  1. CPU 密集型任务:
     - 理论公式: 进程数 = CPU 核心数
     - 实际测试最佳: {best_cpu_processes} 进程
     - 经验范围: {cpu_count} ~ {cpu_count * 2} 进程
     - 原因: 每个进程有独立 GIL，可以真正并行执行

  2. I/O 密集型任务:
     - 通常不建议使用多进程（进程创建开销大）
     - 如果必须使用，建议进程数 = {cpu_count * 2}

通用建议:
  1. 先测试，再选择：不同场景最优并发数不同
  2. 留有裕量：不要榨干所有资源，给系统留一些
  3. 监控调整：上线后监控性能，根据实际情况调整
  4. 考虑开销：并发数越大，线程/进程切换和管理开销也越大

快速决策树:

    这是什么类型的任务？
      │
      ├─→ I/O 密集型 (网络请求、文件读写、数据库查询)
      │     │
      │     └─→ 使用 ThreadPoolExecutor，线程数 = {cpu_count * 5}
      │
      └─→ CPU 密集型 (数学计算、图像处理、数据压缩)
            │
            └─→ 使用 ProcessPoolExecutor，进程数 = {cpu_count}
    """
    
    print(recommendations)


def main():
    cpu_count = os.cpu_count()
    print(f"并发数选择策略测试程序")
    print(f"系统 CPU 核心数: {cpu_count}\n")
    
    # 测试 1: I/O 密集型任务 - 线程池
    thread_sizes = [2, 4, 8, 16, 32, 50, 100]
    io_results = test_thread_pool_sizes(
        io_intensive_task,
        "I/O 密集型任务",
        thread_sizes,
        num_tasks=100,
        wait_time=0.1
    )
    best_io_threads = analyze_results(io_results, cpu_count, "I/O 密集型 (线程池)")
    
    # 测试 2: CPU 密集型任务 - 进程池
    process_sizes = list(range(2, min(cpu_count * 2 + 1, 32), 2))
    cpu_results = test_process_pool_sizes(
        cpu_intensive_task,
        "CPU 密集型任务",
        process_sizes,
        num_tasks=cpu_count * 4,
        size=200000
    )
    best_cpu_processes = analyze_results(cpu_results, cpu_count, "CPU 密集型 (进程池)")
    
    # 打印建议
    print_recommendations(cpu_count, best_io_threads, best_cpu_processes)
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
