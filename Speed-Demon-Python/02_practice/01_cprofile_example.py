#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cProfile 使用示例
演示如何使用内置的 cProfile 模块进行性能分析
"""

import cProfile
import pstats
import math


def slow_fibonacci(n):
    """
    递归计算斐波那契数列 - 故意写得很慢用于演示
    时间复杂度: O(2^n)
    """
    if n <= 1:
        return n
    return slow_fibonacci(n - 1) + slow_fibonacci(n - 2)


def fast_fibonacci(n):
    """
    迭代计算斐波那契数列 - 优化版本
    时间复杂度: O(n)
    """
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def math_intensive_task(numbers):
    """
    进行密集数学计算的函数
    """
    result = []
    for x in numbers:
        val = math.sqrt(x) + math.sin(x) + math.cos(x)
        result.append(val)
    return result


def process_data():
    """
    主处理函数，调用各个子函数
    """
    print("开始处理数据...")
    
    # 调用慢的斐波那契函数（n不要太大，否则太慢）
    fib_result = slow_fibonacci(30)
    print(f"慢方法斐波那契(30) = {fib_result}")
    
    # 调用快的斐波那契函数
    fast_fib_result = fast_fibonacci(100)
    print(f"快方法斐波那契(100) = {fast_fib_result}")
    
    # 调用数学计算函数
    math_result = math_intensive_task(range(100000))
    print(f"数学计算完成，结果长度: {len(math_result)}")
    
    print("数据处理完成!")


def profile_with_context():
    """
    使用上下文管理器方式进行性能分析
    """
    print("\n=== 使用上下文管理器方式分析 ===")
    with cProfile.Profile() as profiler:
        process_data()
    
    # 打印统计结果，按累计时间排序
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(10)  # 只显示前10条


def profile_with_manual_enable():
    """
    使用手动开启/关闭方式进行性能分析
    """
    print("\n=== 使用手动开启/关闭方式分析 ===")
    profiler = cProfile.Profile()
    profiler.enable()
    
    process_data()
    
    profiler.disable()
    
    # 保存结果到文件
    profiler.dump_stats("cprofile_results.prof")
    print("分析结果已保存到 cprofile_results.prof")
    
    # 打印统计结果，按函数本身耗时排序
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)


def analyze_saved_profile():
    """
    分析已保存的 profile 文件
    """
    print("\n=== 分析已保存的 profile 文件 ===")
    stats = pstats.Stats("cprofile_results.prof")
    
    # 按调用次数排序
    print("\n--- 按调用次数排序的前10个函数 ---")
    stats.sort_stats(pstats.SortKey.CALLS)
    stats.print_stats(10)
    
    # 只显示包含 'fibonacci' 的函数
    print("\n--- 只显示包含 'fibonacci' 的函数 ---")
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats("fibonacci")


if __name__ == "__main__":
    # 方式一: 命令行调用
    # python -m cProfile -s cumulative 01_cprofile_example.py
    
    # 方式二: 代码内调用 - 使用上下文管理器
    profile_with_context()
    
    # 方式三: 代码内调用 - 手动开启/关闭
    profile_with_manual_enable()
    
    # 方式四: 分析已保存的文件
    analyze_saved_profile()
    
    print("\n" + "=" * 60)
    print("提示: 也可以使用命令行方式运行:")
    print("  python -m cProfile -s cumulative 01_cprofile_example.py")
    print("  python -m cProfile -o profile_output.prof 01_cprofile_example.py")
    print("  snakeviz profile_output.prof  # 需要先安装 snakeviz")
    print("=" * 60)
