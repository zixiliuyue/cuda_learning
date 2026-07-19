#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pympler 对象大小分析示例
运行方式：python 08_pympler_example.py
"""

from pympler import asizeof
from pympler.tracker import SummaryTracker
from pympler import muppy, summary
import gc


class ComplexObject:
    """复杂对象，包含多种属性"""
    
    def __init__(self, obj_id):
        self.id = obj_id
        self.name = f'object_{obj_id}'
        self.data = [i for i in range(100)]
        self.metadata = {
            'created': 'now',
            'version': 1,
            'tags': ['tag1', 'tag2', 'tag3']
        }


def demo_asizeof_basic():
    """演示：基础对象大小计算"""
    print("\n" + "=" * 60)
    print("演示1：基础对象大小计算")
    print("=" * 60)
    
    test_cases = [
        ("整数", 42),
        ("浮点数", 3.14159),
        ("短字符串", "hello"),
        ("长字符串", "a" * 1000),
        ("空列表", []),
        ("小列表", [1, 2, 3]),
        ("大列表", list(range(10000))),
        ("空字典", {}),
        ("小字典", {'a': 1, 'b': 2, 'c': 3}),
        ("大字典", {f'key_{i}': i for i in range(1000)}),
        ("元组", (1, 2, 3, 4, 5)),
        ("集合", {1, 2, 3, 4, 5}),
    ]
    
    print("\n[各类型对象大小]:")
    for name, obj in test_cases:
        size = asizeof.asizeof(obj)
        print(f"  {name:12s}: {size:8d} bytes ({size / 1024:.2f} KB)")


def demo_asizeof_detail():
    """演示：详细的对象大小分析"""
    print("\n" + "=" * 60)
    print("演示2：详细对象大小分析")
    print("=" * 60)
    
    complex_obj = ComplexObject(1)
    
    print("\n[复杂对象详细大小分析]:")
    sized = asizeof.asized(complex_obj, detail=2)
    print(sized.format())
    
    print(f"\n[递归总大小]: {asizeof.asizeof(complex_obj)} bytes")
    print(f"[只算自身大小]: {asizeof.asizeof(complex_obj, detail=0)} bytes")
    
    nested_data = {
        'list1': [1, 2, 3],
        'list2': [[1, 2], [3, 4]],
        'dict1': {'a': [1, 2, 3], 'b': {'c': 4}},
        'objects': [ComplexObject(i) for i in range(3)]
    }
    
    print(f"\n[嵌套数据结构总大小]: {asizeof.asizeof(nested_data) / 1024:.2f} KB")


def demo_tracker():
    """演示：SummaryTracker 跟踪内存变化"""
    print("\n" + "=" * 60)
    print("演示3：SummaryTracker 跟踪内存变化")
    print("=" * 60)
    
    tracker = SummaryTracker()
    
    print("\n[初始状态]:")
    tracker.print_diff()
    
    print("\n[分配大列表后]:")
    big_list = [i for i in range(100000)]
    tracker.print_diff()
    
    print("\n[分配大字典后]:")
    big_dict = {f'key_{i}': f'value_{i}' for i in range(50000)}
    tracker.print_diff()
    
    print("\n[释放大列表后]:")
    del big_list
    gc.collect()
    tracker.print_diff()
    
    print("\n[释放大字典后]:")
    del big_dict
    gc.collect()
    tracker.print_diff()


def demo_muppy():
    """演示：muppy 获取所有对象"""
    print("\n" + "=" * 60)
    print("演示4：muppy 内存使用摘要")
    print("=" * 60)
    
    all_objects = muppy.get_objects()
    print(f"\n[总对象数]: {len(all_objects)}")
    
    sum1 = summary.summarize(all_objects)
    print("\n[内存使用摘要 Top 10]:")
    summary.print_(sum1, limit=10)
    
    print("\n[创建一些对象...]")
    objects = [ComplexObject(i) for i in range(100)]
    
    all_objects2 = muppy.get_objects()
    sum2 = summary.summarize(all_objects2)
    
    print("\n[创建对象后的增长]:")
    diff = summary.get_diff(sum1, sum2)
    summary.print_(diff, limit=10)
    
    del objects
    gc.collect()


def demo_size_comparison():
    """演示：不同数据结构大小对比"""
    print("\n" + "=" * 60)
    print("演示5：不同数据结构大小对比")
    print("=" * 60)
    
    n = 10000
    
    list_of_ints = list(range(n))
    list_of_strings = [f'item_{i}' for i in range(n)]
    list_of_tuples = [(i, f'item_{i}') for i in range(n)]
    dict_of_values = {f'key_{i}': i for i in range(n)}
    set_of_values = set(range(n))
    
    data_structures = [
        ("整数列表", list_of_ints),
        ("字符串列表", list_of_strings),
        ("元组列表", list_of_tuples),
        ("字典", dict_of_values),
        ("集合", set_of_values),
    ]
    
    print(f"\n[各数据结构大小对比 (n={n})]:")
    print(f"{'类型':15s} {'总大小 (KB)':>12s} {'每个元素 (bytes)':>18s}")
    print("-" * 50)
    
    for name, obj in data_structures:
        size = asizeof.asizeof(obj)
        per_item = size / n
        print(f"{name:15s} {size / 1024:12.2f} {per_item:18.2f}")


def demo_object_overhead():
    """演示：对象开销分析"""
    print("\n" + "=" * 60)
    print("演示6：对象开销分析")
    print("=" * 60)
    
    class EmptyClass:
        pass
    
    class SimpleClass:
        def __init__(self):
            self.x = 1
    
    class SlotClass:
        __slots__ = ['x', 'y', 'z']
        
        def __init__(self):
            self.x = 1
            self.y = 2
            self.z = 3
    
    class AttrClass:
        def __init__(self):
            self.x = 1
            self.y = 2
            self.z = 3
    
    test_objects = [
        ("空类实例", EmptyClass()),
        ("单个属性类", SimpleClass()),
        ("__slots__ 类", SlotClass()),
        ("三个属性类", AttrClass()),
    ]
    
    print("\n[类实例开销对比]:")
    print(f"{'类型':20s} {'大小 (bytes)':>12s}")
    print("-" * 35)
    
    for name, obj in test_objects:
        size = asizeof.asizeof(obj)
        print(f"{name:20s} {size:12d}")
    
    n = 1000
    normal_instances = [AttrClass() for _ in range(n)]
    slot_instances = [SlotClass() for _ in range(n)]
    
    normal_size = asizeof.asizeof(normal_instances)
    slot_size = asizeof.asizeof(slot_instances)
    
    print(f"\n[批量创建 {n} 个实例对比]:")
    print(f"普通类: {normal_size / 1024:.2f} KB (每个 {normal_size / n:.2f} bytes)")
    print(f"__slots__: {slot_size / 1024:.2f} KB (每个 {slot_size / n:.2f} bytes)")
    print(f"节省: {(normal_size - slot_size) / normal_size * 100:.1f}%")


if __name__ == '__main__':
    print("=" * 60)
    print("pympler 对象大小分析示例")
    print("=" * 60)
    
    demo_asizeof_basic()
    demo_asizeof_detail()
    demo_tracker()
    demo_muppy()
    demo_size_comparison()
    demo_object_overhead()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
