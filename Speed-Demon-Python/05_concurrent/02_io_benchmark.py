#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O 密集型任务性能对比测试
对比：单线程、多线程、多进程在 I/O 密集型任务上的表现
"""

import time
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def io_intensive_task(task_id, sleep_time=0.5):
    """模拟 I/O 密集型任务：网络请求、文件读写等"""
    print(f"任务 {task_id}: 开始执行，模拟 I/O 等待 {sleep_time} 秒...")
    
    time.sleep(sleep_time)  # 模拟 I/O 等待
    
    print(f"任务 {task_id}: 执行完成")
    return f"任务 {task_id} 结果"


def run_single_thread(num_tasks, sleep_time):
    """单线程执行"""
    print(f"\n{'='*60}")
    print(f"单线程模式：{num_tasks} 个任务，每个任务等待 {sleep_time} 秒")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    results = []
    for i in range(num_tasks):
        results.append(io_intensive_task(i, sleep_time))
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.2f} 秒")
    print(f"理论串行耗时: {num_tasks * sleep_time:.2f} 秒")
    print(f"效率: {(num_tasks * sleep_time) / elapsed * 100:.1f}%")
    
    return elapsed


def run_multi_thread(num_tasks, sleep_time, num_workers):
    """多线程执行"""
    print(f"\n{'='*60}")
    print(f"多线程模式：{num_tasks} 个任务，{num_workers} 个线程，每个任务等待 {sleep_time} 秒")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            io_intensive_task,
            range(num_tasks),
            [sleep_time] * num_tasks
        ))
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.2f} 秒")
    print(f"理论串行耗时: {num_tasks * sleep_time:.2f} 秒")
    print(f"理论最佳并行耗时: {max(sleep_time, (num_tasks + num_workers - 1) // num_workers * sleep_time):.2f} 秒")
    print(f"效率: {(num_tasks * sleep_time) / elapsed * 100:.1f}%")
    
    return elapsed


def run_multi_process(num_tasks, sleep_time, num_workers):
    """多进程执行"""
    print(f"\n{'='*60}")
    print(f"多进程模式：{num_tasks} 个任务，{num_workers} 个进程，每个任务等待 {sleep_time} 秒")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            io_intensive_task,
            range(num_tasks),
            [sleep_time] * num_tasks
        ))
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.2f} 秒")
    print(f"理论串行耗时: {num_tasks * sleep_time:.2f} 秒")
    print(f"效率: {(num_tasks * sleep_time) / elapsed * 100:.1f}%")
    
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
    
    if thread_time < single_time:
        print(f"✓ 多线程比单线程快 {single_time/thread_time:.2f} 倍 - I/O 等待时 GIL 被释放，其他线程可以执行")
    else:
        print("✗ 多线程比单线程慢 - 可能是任务太小或并发数设置不当")
    
    if abs(thread_time - process_time) / thread_time < 0.1:  # 相差不超过 10%
        print("✓ 多线程和多进程性能相近 - 对于 I/O 密集型任务，两者都能有效利用等待时间")
    elif thread_time < process_time:
        print("✓ 多线程比多进程快 - 进程创建和通信开销超过了并行收益")
    
    print(f"\n{'='*60}")
    print("为什么会这样？")
    print(f"{'='*60}")
    print("""
1. 单线程: 必须等待一个任务完成后才能开始下一个，I/O 时间被浪费了
2. 多线程: 当一个线程在等待 I/O 时，它会释放 GIL，其他线程可以继续执行
   这样 I/O 等待时间被有效利用，整体性能显著提升
3. 多进程: 虽然也能并行执行，但进程创建和通信开销比线程大
   对于 I/O 密集型任务，多线程通常就足够了

→ 结论：I/O 密集型任务推荐使用多线程（ThreadPoolExecutor）
    """)


def main():
    cpu_count = os.cpu_count()
    print(f"系统信息: {cpu_count} 个 CPU 核心")
    
    num_tasks = 20           # 任务数量
    sleep_time = 0.5         # 每个任务的 I/O 等待时间
    num_workers = min(num_tasks, cpu_count * 5)  # I/O 密集型推荐更多 worker
    
    print(f"测试配置: {num_tasks} 个任务，每个任务等待 {sleep_time} 秒")
    print(f"并发数: {num_workers}")
    
    single_time = run_single_thread(num_tasks, sleep_time)
    thread_time = run_multi_thread(num_tasks, sleep_time, num_workers)
    process_time = run_multi_process(num_tasks, sleep_time, min(num_workers, cpu_count * 2))
    
    compare_results(single_time, thread_time, process_time)


if __name__ == "__main__":
    main()
