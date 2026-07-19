#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory_profiler 内存分析示例
运行方式：
    1. 逐行分析：python -m memory_profiler 05_memory_profiler_example.py
    2. 时间线可视化：
        mprof run 05_memory_profiler_example.py
        mprof plot
"""

import gc
import time


@profile
def memory_intensive_function():
    """内存密集型函数，演示逐行内存分析"""
    result = []
    
    for i in range(10000):
        data = [i] * 100
        result.append(data)
    
    return result


@profile
def nested_memory_usage():
    """演示嵌套数据结构的内存使用"""
    outer_list = []
    
    for i in range(100):
        inner_dict = {
            'id': i,
            'data': [f'item_{j}' for j in range(50)],
            'metadata': {'type': 'test', 'index': i}
        }
        outer_list.append(inner_dict)
    
    time.sleep(0.1)
    return outer_list


@profile
def memory_with_release():
    """演示内存释放的时机"""
    big_data1 = list(range(100000))
    time.sleep(0.1)
    
    big_data2 = list(range(50000))
    time.sleep(0.1)
    
    del big_data1
    gc.collect()
    time.sleep(0.1)
    
    big_data3 = list(range(30000))
    time.sleep(0.1)
    
    return big_data2, big_data3


@profile
def string_memory_usage():
    """演示字符串的内存使用"""
    short_strings = []
    long_strings = []
    
    for i in range(5000):
        short_strings.append(f'short_{i}')
    
    for i in range(1000):
        long_str = 'long_string_' * 100 + str(i)
        long_strings.append(long_str)
    
    return short_strings, long_strings


if __name__ == '__main__':
    print("=" * 60)
    print("memory_profiler 逐行内存分析示例")
    print("=" * 60)
    
    print("\n--- 1. 内存密集型函数 ---")
    result1 = memory_intensive_function()
    print(f"创建了 {len(result1)} 个元素")
    del result1
    gc.collect()
    
    print("\n--- 2. 嵌套数据结构 ---")
    result2 = nested_memory_usage()
    print(f"创建了 {len(result2)} 个字典")
    del result2
    gc.collect()
    
    print("\n--- 3. 内存释放演示 ---")
    result3 = memory_with_release()
    print(f"剩余数据大小: {len(result3[0])}, {len(result3[1])}")
    del result3
    gc.collect()
    
    print("\n--- 4. 字符串内存使用 ---")
    result4 = string_memory_usage()
    print(f"短字符串: {len(result4[0])} 个, 长字符串: {len(result4[1])} 个")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("提示：使用 'mprof plot' 查看内存使用时间线图")
    print("=" * 60)
