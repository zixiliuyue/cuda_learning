#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU 密集型任务性能对比测试
对比：单线程、多线程、多进程在 CPU 密集型任务上的表现
"""

import time
import os
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def cpu_intensive_task(n):
    """CPU 密集型任务：计算数学函数"""
    result = 0
    for i in range(n):
        result += math.sqrt(i) * math.sin(i) * math.cos(i)
    return result


def run_single_thread(num_tasks, task_size):
    """单线程执行"""
    print(f"\n{'='*60}")
    print(f"单线程模式：{num_tasks} 个任务，每个任务 {task_size:,} 次计算")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    results = []
    for _ in range(num_tasks):
        results.append(cpu_intensive_task(task_size))
    
    elapsed = time.time() - start_time
    print(f"总耗时: {elapsed:.2f} 秒")
    print(f"平均每任务: {elapsed/num_tasks:.2f} 秒")
    
    return elapsed


def run_multi_thread(num_tasks, task_size, num_workers):
    """多线程执行"""
    print(f"\n{'='*60}")
    print(f"多线程模式：{num_tasks} 个任务，{num_workers} 个线程，每个任务 {task_size:,} 次计算")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            cpu_intensive_task,
            [task_size] * num_tasks
        ))
    
    elapsed = time.time() - start_time
    print(f"总耗时: {elapsed:.2f} 秒")
    print(f"平均每任务: {elapsed/num_tasks:.2f} 秒")
    
    return elapsed


def run_multi_process(num_tasks, task_size, num_workers):
    """多进程执行"""
    print(f"\n{'='*60}")
    print(f"多进程模式：{num_tasks} 个任务，{num_workers} 个进程，每个任务 {task_size:,} 次计算")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            cpu_intensive_task,
            [task_size] * num_tasks
        ))
    
    elapsed = time.time() - start_time
    print(f"总耗时: {elapsed:.2f} 秒")
    print(f"平均每任务: {elapsed/num_tasks:.2f} 秒")
    
    return elapsed


def compare_results(single_time, thread_time, process_time):
    """对比结果"""
    print(f"\n{'='*60}")
    print("性能对比总结")
    print(f"{'='*60}")
    
    print(f"单线程耗时:  {single_time:.2f} 秒 (基准)")
    print(f"多线程耗时:  {thread_time:.2f} 秒 (加速比: {single_time/thread_time:.2f}x)")
    print(f"多进程耗时:  {process_time:.2f} 秒 (加速比: {single_time/process_time:.2f}x)")
    
    print(f"\n{'='*60}")
    print("结论分析")
    print(f"{'='*60}")
    
    if thread_time > single_time:
        print("✓ 多线程比单线程慢 - 这是由于 GIL 的存在，CPU 密集型任务无法真正并行")
    else:
        print("✓ 多线程比单线程快 - 可能是任务中有 I/O 成分或其他释放 GIL 的操作")
    
    if process_time < single_time:
        print(f"✓ 多进程比单线程快 {single_time/process_time:.2f} 倍 - 真正利用了多核 CPU")
    else:
        print("✗ 多进程比单线程慢 - 可能是任务太小，进程创建开销超过并行收益")
    
    if process_time < thread_time:
        print(f"✓ 多进程比多线程快 {thread_time/process_time:.2f} 倍 - CPU 密集型任务推荐使用多进程")


def main():
    cpu_count = os.cpu_count()
    print(f"系统信息: {cpu_count} 个 CPU 核心")
    
    num_tasks = cpu_count  # 任务数等于核心数
    task_size = 2000000    # 每个任务的计算量
    
    print(f"测试配置: {num_tasks} 个任务，每个任务 {task_size:,} 次计算")
    
    single_time = run_single_thread(num_tasks, task_size)
    thread_time = run_multi_thread(num_tasks, task_size, num_workers=cpu_count)
    process_time = run_multi_process(num_tasks, task_size, num_workers=cpu_count)
    
    compare_results(single_time, thread_time, process_time)
    
    print(f"\n{'='*60}")
    print("为什么会这样？")
    print(f"{'='*60}")
    print("""
1. 单线程: 只有一个线程在执行，无法利用多核 CPU
2. 多线程: 由于 GIL（全局解释器锁）的存在，同一时间只有一个线程在执行 Python 字节码
   而且线程切换还有额外开销，所以可能比单线程更慢
3. 多进程: 每个进程有独立的 Python 解释器和 GIL，可以真正并行执行
   虽然进程创建有开销，但对于 CPU 密集型任务，并行收益大于开销

→ 结论：CPU 密集型任务应该使用多进程（ProcessPoolExecutor）
    """)


if __name__ == "__main__":
    main()
