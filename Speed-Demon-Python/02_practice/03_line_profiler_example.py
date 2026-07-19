#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
line_profiler 使用示例
使用 @profile 装饰器标记需要逐行分析的函数

运行方式:
    kernprof -l -v 03_line_profiler_example.py
"""

import math
import random


# 注意: @profile 装饰器由 kernprof 提供，不需要导入
@profile
def slow_data_processing(data):
    """
    慢速数据处理函数 - 使用传统的循环和 append
    这个函数会被逐行分析性能
    """
    result = []
    
    # 第一层循环
    for i in range(len(data)):
        item = data[i]
        
        # 对每个元素进行多次数学计算
        processed = 0
        for j in range(10):
            processed += math.sqrt(item + j) + math.sin(item + j)
        
        # 格式化字符串
        formatted = f"Item_{i:05d}_value_{processed:.2f}"
        
        # 添加到结果列表
        result.append(formatted)
    
    return result


@profile
def faster_data_processing(data):
    """
    相对较快的数据处理函数 - 使用列表推导式和优化
    """
    # 使用预计算和列表推导式
    def process_item(args):
        i, item = args
        processed = sum(math.sqrt(item + j) + math.sin(item + j) for j in range(10))
        return f"Item_{i:05d}_value_{processed:.2f}"
    
    return [process_item(args) for args in enumerate(data)]


@profile
def matrix_operation_naive(matrix):
    """
    朴素的矩阵操作 - 三重循环
    """
    n = len(matrix)
    result = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += matrix[i][k] * matrix[k][j]
            result[i][j] = math.sqrt(total) if total > 0 else 0.0
    
    return result


@profile
def string_manipulation_slow(strings):
    """
    慢速字符串处理
    """
    result = ""
    
    for s in strings:
        # 多次字符串操作
        s_upper = s.upper()
        s_lower = s.lower()
        s_reversed = s[::-1]
        s_stripped = s.strip()
        
        # 字符串拼接（慢）
        result += s_upper + "_" + s_lower + "_" + s_reversed + "_" + s_stripped + "|"
    
    return result


@profile
def string_manipulation_fast(strings):
    """
    快速字符串处理 - 使用列表 join
    """
    result_parts = []
    
    for s in strings:
        s_upper = s.upper()
        s_lower = s.lower()
        s_reversed = s[::-1]
        s_stripped = s.strip()
        
        # 使用列表 append（快）
        result_parts.append(f"{s_upper}_{s_lower}_{s_reversed}_{s_stripped}|")
    
    return "".join(result_parts)


@profile
def comprehensive_function(data):
    """
    综合函数 - 包含多个不同类型的操作
    便于观察哪一行真正耗时
    """
    result = []
    
    # 操作1: 数学计算
    math_results = []
    for x in data:
        val = math.sqrt(x) + math.sin(x) + math.cos(x)
        math_results.append(val)
    
    # 操作2: 条件过滤
    filtered = [x for x in math_results if x > 10]
    
    # 操作3: 排序
    sorted_data = sorted(filtered, reverse=True)
    
    # 操作4: 格式化输出
    for i, x in enumerate(sorted_data[:100]):
        result.append(f"Rank_{i:03d}: {x:.4f}")
    
    return result


def demonstrate_optimization():
    """
    演示优化前后的对比
    """
    print("=" * 60)
    print("line_profiler 示例")
    print("运行: kernprof -l -v 03_line_profiler_example.py")
    print("=" * 60)
    
    # 生成测试数据
    print("\n生成测试数据...")
    data = [random.randint(1, 1000) for _ in range(1000)]
    strings = [f"test_string_{i}_abcdefghijklmnop" for i in range(200)]
    small_matrix = [[random.random() for _ in range(30)] for _ in range(30)]
    
    # 运行各个被 @profile 标记的函数
    print("\n运行 slow_data_processing...")
    result1 = slow_data_processing(data[:500])
    print(f"  处理了 {len(result1)} 条数据")
    
    print("\n运行 faster_data_processing...")
    result2 = faster_data_processing(data[:500])
    print(f"  处理了 {len(result2)} 条数据")
    
    print("\n运行 string_manipulation_slow...")
    result3 = string_manipulation_slow(strings)
    print(f"  生成了 {len(result3)} 字符的字符串")
    
    print("\n运行 string_manipulation_fast...")
    result4 = string_manipulation_fast(strings)
    print(f"  生成了 {len(result4)} 字符的字符串")
    
    print("\n运行 matrix_operation_naive (小矩阵)...")
    result5 = matrix_operation_naive(small_matrix)
    print(f"  处理了 {len(result5)}x{len(result5[0])} 的矩阵")
    
    print("\n运行 comprehensive_function...")
    result6 = comprehensive_function(data)
    print(f"  生成了 {len(result6)} 条结果")
    
    print("\n" + "=" * 60)
    print("所有函数执行完成，请查看上面的逐行分析结果")
    print("=" * 60)


def show_optimization_tips():
    """
    显示优化建议
    """
    print("\n\n" + "=" * 60)
    print("优化建议")
    print("=" * 60)
    print("""
根据 line_profiler 的输出，你应该关注：
1. % Time 列数值高的行 - 这是主要瓶颈
2. Hits 列数值异常高的行 - 可能存在不必要的重复计算
3. Per Hit 列数值高的行 - 单次执行就很慢

常见优化方法：
- 循环内的重复计算移到循环外
- 使用列表推导式替代 for + append
- 使用 ''.join() 替代字符串 += 拼接
- 使用内置函数和库（通常是C实现的）
- 考虑使用生成器而不是列表
    """)


if __name__ == "__main__":
    # 设置随机种子确保可复现
    random.seed(42)
    
    # 运行演示
    demonstrate_optimization()
    
    # 显示优化建议
    show_optimization_tips()
