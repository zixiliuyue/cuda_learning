#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行所有benchmark并输出美观的汇总表格
"""

import sys
import os
from typing import List, Dict

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入各个benchmark模块
import importlib
benchmark_modules = [
    ("01_gil_benchmark", "GIL性能对比"),
    ("02_data_structure_benchmark", "数据结构性能对比"),
    ("03_loop_benchmark", "循环优化对比"),
    ("04_numba_benchmark", "Numba加速对比"),
    ("05_numpy_benchmark", "NumPy向量化对比"),
    ("06_concurrency_benchmark", "并发性能对比"),
]


def print_header(title: str, width: int = 80):
    """打印标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_table(data: List[Dict], category: str):
    """打印美观的表格"""
    if not data:
        return
    
    # 表头
    col1_width = max(30, max(len(item["name"]) for item in data) + 2)
    col2_width = 20
    col3_width = 15
    
    print(f"+{'-' * col1_width}+{'-' * col2_width}+{'-' * col3_width}+")
    print(f"| {'测试项目':<{col1_width-2}} | {'耗时 (秒)':<{col2_width-2}} | {'加速比':<{col3_width-2}} |")
    print(f"+{'=' * col1_width}+{'=' * col2_width}+{'=' * col3_width}+")
    
    # 数据行
    for item in data:
        time_str = f"{item['time']:.6f}"
        speedup_str = f"{item['speedup']:.2f}x" if item['speedup'] >= 1 else f"{item['speedup']:.2f}x"
        print(f"| {item['name']:<{col1_width-2}} | {time_str:>{col2_width-2}} | {speedup_str:>{col3_width-2}} |")
    
    print(f"+{'-' * col1_width}+{'-' * col2_width}+{'-' * col3_width}+")
    print()


def print_summary(all_results: List[Dict]):
    """打印所有benchmark的汇总"""
    print_header("性能测试汇总报告")
    
    total_categories = 0
    total_tests = 0
    max_speedup = 0
    max_speedup_name = ""
    
    # 统计信息
    for result in all_results:
        if result["tests"]:
            total_categories += 1
            total_tests += len(result["tests"])
            for test in result["tests"]:
                if test["speedup"] > max_speedup:
                    max_speedup = test["speedup"]
                    max_speedup_name = f"{result['category']} - {test['name']}"
    
    # 统计表格
    col1_width = 50
    col2_width = 25
    
    print(f"+{'-' * col1_width}+{'-' * col2_width}+")
    print(f"| {'统计项':<{col1_width-2}} | {'数值':<{col2_width-2}} |")
    print(f"+{'=' * col1_width}+{'=' * col2_width}+")
    print(f"| {'测试类别总数':<{col1_width-2}} | {total_categories:>{col2_width-2}} |")
    print(f"| {'测试项目总数':<{col1_width-2}} | {total_tests:>{col2_width-2}} |")
    print(f"| {'最大加速比':<{col1_width-2}} | {max_speedup:.2f}x |")
    print(f"+{'-' * col1_width}+{'-' * col2_width}+")
    
    # 分类别详细结果
    for result in all_results:
        if result["tests"]:
            print_table(result["tests"], result["category"])


def main():
    """主函数"""
    print_header("Python 性能基准测试套件", 80)
    print("开始运行所有基准测试...\n")
    
    all_results = []
    
    for module_name, display_name in benchmark_modules:
        try:
            module = importlib.import_module(module_name)
            result = module.run_benchmark()
            all_results.append(result)
        except Exception as e:
            print(f"运行 {display_name} 时出错: {e}")
            print()
    
    # 打印汇总
    print_summary(all_results)
    
    print("\n所有基准测试完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
