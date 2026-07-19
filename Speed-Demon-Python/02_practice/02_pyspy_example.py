#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
py-spy 使用示例
这个脚本设计为运行时间较长，适合用 py-spy 进行采样分析
"""

import math
import time
import random
from threading import Thread


def slow_sort_algorithm(arr):
    """
    冒泡排序 - 故意写得很慢
    时间复杂度: O(n²)
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def fast_sort_algorithm(arr):
    """
    Python内置排序 - 快速
    时间复杂度: O(n log n)
    """
    return sorted(arr)


def heavy_math_calculation(n):
    """
    密集数学计算
    """
    result = 0
    for i in range(n):
        result += math.sqrt(i) + math.sin(i) + math.cos(i) + math.tan(i / 1000)
    return result


def string_processing_task():
    """
    字符串处理任务
    """
    result = []
    for i in range(10000):
        s = f"string_{i}_{i * 2}"
        s_upper = s.upper()
        s_reversed = s_upper[::-1]
        result.append(s_reversed)
    return result


def nested_loop_operation(size):
    """
    嵌套循环操作
    """
    result = []
    for i in range(size):
        inner_result = []
        for j in range(size // 10):
            inner_result.append(i * j + math.sqrt(i + j))
        result.append(sum(inner_result))
    return result


def simulated_workload_a():
    """
    模拟工作负载 A - 主要是排序和数学计算
    """
    print("开始工作负载 A...")
    
    # 生成随机数据
    data = [random.randint(1, 10000) for _ in range(5000)]
    
    # 使用慢排序
    slow_sort_algorithm(data.copy())
    
    # 进行数学计算
    heavy_math_calculation(100000)
    
    print("工作负载 A 完成!")


def simulated_workload_b():
    """
    模拟工作负载 B - 主要是字符串处理和嵌套循环
    """
    print("开始工作负载 B...")
    
    # 字符串处理
    string_processing_task()
    
    # 嵌套循环操作
    nested_loop_operation(300)
    
    print("工作负载 B 完成!")


def long_running_task():
    """
    长时间运行的任务 - 适合用 py-spy 附加分析
    """
    print("=" * 60)
    print("长时间运行的任务已启动")
    print("提示: 可以在另一个终端运行以下命令分析:")
    print("  py-spy top -- python 02_pyspy_example.py")
    print("  py-spy record -o pyspy_flame.svg -- python 02_pyspy_example.py")
    print("或者找到进程ID后附加:")
    print("  py-spy top --pid <进程ID>")
    print("=" * 60)
    
    for iteration in range(5):
        print(f"\n--- 迭代 {iteration + 1}/5 ---")
        
        # 交替执行不同的工作负载
        if iteration % 2 == 0:
            simulated_workload_a()
        else:
            simulated_workload_b()
        
        # 短暂休息，便于观察
        time.sleep(0.5)


def threaded_example():
    """
    多线程示例 - 演示 py-spy 如何分析多线程程序
    """
    print("\n" + "=" * 60)
    print("多线程示例已启动")
    print("=" * 60)
    
    def worker1():
        for _ in range(3):
            heavy_math_calculation(50000)
            time.sleep(0.1)
    
    def worker2():
        for _ in range(3):
            string_processing_task()
            time.sleep(0.1)
    
    thread1 = Thread(target=worker1, name="计算线程")
    thread2 = Thread(target=worker2, name="处理线程")
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()


def optimized_vs_unoptimized():
    """
    比较优化版和非优化版的性能差异
    """
    print("\n" + "=" * 60)
    print("比较优化版和非优化版的性能")
    print("=" * 60)
    
    data = [random.randint(1, 10000) for _ in range(5000)]
    
    print("使用冒泡排序 (慢)...")
    start = time.time()
    slow_sort_algorithm(data.copy())
    print(f"冒泡排序耗时: {time.time() - start:.3f} 秒")
    
    print("使用内置排序 (快)...")
    start = time.time()
    fast_sort_algorithm(data.copy())
    print(f"内置排序耗时: {time.time() - start:.3f} 秒")


if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    random.seed(42)
    
    try:
        # 运行长时间任务
        long_running_task()
        
        # 运行多线程示例
        threaded_example()
        
        # 比较性能差异
        optimized_vs_unoptimized()
        
        print("\n" + "=" * 60)
        print("所有任务完成!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
        print("\n提示: 如果你使用 py-spy 分析正在运行的程序，")
        print("火焰图应该已经生成了，你可以查看它来分析性能热点")
