#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracemalloc 内存快照对比示例
运行方式：python 06_tracemalloc_example.py
"""

import tracemalloc
import gc


def allocate_large_list():
    """分配大列表"""
    return [i for i in range(100000)]


def allocate_large_dict():
    """分配大字典"""
    return {f'key_{i}': f'value_{i}' for i in range(50000)}


def allocate_nested_structure():
    """分配嵌套结构"""
    result = []
    for i in range(1000):
        result.append({
            'id': i,
            'data': [j for j in range(100)],
            'metadata': {'type': 'test', 'index': i}
        })
    return result


def filter_internal_frames(stat):
    """过滤内部调用栈"""
    traceback = stat.traceback
    if traceback:
        frame = traceback[0]
        filename = frame.filename
        return 'tracemalloc' not in filename and '<frozen' not in filename
    return True


def demo_simple_tracking():
    """演示：简单跟踪内存分配"""
    print("\n" + "=" * 60)
    print("演示1：简单跟踪内存分配")
    print("=" * 60)
    
    tracemalloc.start()
    
    allocate_large_list()
    allocate_large_dict()
    
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("\n[Top 10 内存分配位置]")
    for i, stat in enumerate(top_stats[:10]):
        print(f"{i+1}. {stat}")
    
    tracemalloc.stop()


def demo_snapshot_comparison():
    """演示：快照对比"""
    print("\n" + "=" * 60)
    print("演示2：快照对比")
    print("=" * 60)
    
    tracemalloc.start()
    
    snapshot1 = tracemalloc.take_snapshot()
    print(f"\n快照1已拍摄")
    
    data1 = allocate_large_list()
    print(f"已分配大列表")
    
    snapshot2 = tracemalloc.take_snapshot()
    print(f"快照2已拍摄")
    
    data2 = allocate_large_dict()
    print(f"已分配大字典")
    
    snapshot3 = tracemalloc.take_snapshot()
    print(f"快照3已拍摄")
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("\n[快照1 -> 快照2 内存增长 Top 5]")
    for stat in top_stats[:5]:
        print(f"  {stat}")
    
    top_stats = snapshot3.compare_to(snapshot2, 'lineno')
    print("\n[快照2 -> 快照3 内存增长 Top 5]")
    for stat in top_stats[:5]:
        print(f"  {stat}")
    
    del data1, data2
    gc.collect()
    
    snapshot4 = tracemalloc.take_snapshot()
    top_stats = snapshot4.compare_to(snapshot3, 'lineno')
    print("\n[释放内存后 快照3 -> 快照4 变化 Top 5]")
    for stat in top_stats[:5]:
        print(f"  {stat}")
    
    tracemalloc.stop()


def demo_traceback():
    """演示：追踪调用栈"""
    print("\n" + "=" * 60)
    print("演示3：追踪调用栈")
    print("=" * 60)
    
    tracemalloc.start()
    
    allocate_nested_structure()
    
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('traceback')
    
    print("\n[Top 3 内存分配调用栈]")
    for i, stat in enumerate(top_stats[:3]):
        print(f"\n{i+1}. 大小: {stat.size / 1024:.1f} KiB, 计数: {stat.count}")
        for j, line in enumerate(stat.traceback.format()):
            if j < 5:
                print(f"   {line}")
        if len(stat.traceback) > 5:
            print(f"   ... (共 {len(stat.traceback)} 帧)")
    
    tracemalloc.stop()


def demo_memory_leak_simulation():
    """演示：模拟内存泄漏检测"""
    print("\n" + "=" * 60)
    print("演示4：模拟内存泄漏检测")
    print("=" * 60)
    
    leaky_cache = []
    
    tracemalloc.start()
    snapshots = []
    
    for iteration in range(5):
        # 模拟泄漏：每次向全局列表添加数据
        leaky_cache.extend([f'item_{iteration}_{i}' for i in range(2000)])
        
        snapshot = tracemalloc.take_snapshot()
        snapshots.append(snapshot)
        
        print(f"\n迭代 {iteration + 1}: 缓存大小 = {len(leaky_cache)}")
    
    print("\n[每次迭代的内存增长对比]")
    for i in range(1, len(snapshots)):
        stats = snapshots[i].compare_to(snapshots[i-1], 'lineno')
        if stats:
            print(f"\n  迭代 {i} -> {i+1}:")
            for stat in stats[:3]:
                if stat.size_diff > 0:
                    print(f"    +{stat.size_diff / 1024:.1f} KiB at {stat.traceback[0]}")
    
    tracemalloc.stop()


def demo_peak_memory():
    """演示：获取峰值内存"""
    print("\n" + "=" * 60)
    print("演示5：峰值内存统计")
    print("=" * 60)
    
    tracemalloc.start()
    
    allocate_large_list()
    current, peak = tracemalloc.get_traced_memory()
    
    print(f"\n当前内存使用: {current / 1024 / 1024:.2f} MB")
    print(f"峰值内存使用: {peak / 1024 / 1024:.2f} MB")
    
    allocate_large_dict()
    current, peak = tracemalloc.get_traced_memory()
    
    print(f"\n分配字典后:")
    print(f"当前内存使用: {current / 1024 / 1024:.2f} MB")
    print(f"峰值内存使用: {peak / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()


if __name__ == '__main__':
    print("=" * 60)
    print("tracemalloc 内存快照对比示例")
    print("=" * 60)
    
    demo_simple_tracking()
    demo_snapshot_comparison()
    demo_traceback()
    demo_memory_leak_simulation()
    demo_peak_memory()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
