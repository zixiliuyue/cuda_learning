#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
objgraph 对象引用图示例
运行方式：python 07_objgraph_example.py
"""

import objgraph
import gc


class UserData:
    """示例数据类"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.data = []


class Node:
    """节点类，用于演示循环引用"""
    
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None


def demo_most_common_types():
    """演示：显示最常见的对象类型"""
    print("\n" + "=" * 60)
    print("演示1：最常见的对象类型")
    print("=" * 60)
    
    print("\n[执行前 Top 10 对象类型]")
    objgraph.show_most_common_types(limit=10)
    
    test_list = []
    test_dict = {}
    for i in range(1000):
        test_list.append(f'item_{i}')
        test_dict[f'key_{i}'] = f'value_{i}'
    
    print("\n[创建列表和字典后 Top 10 对象类型]")
    objgraph.show_most_common_types(limit=10)
    
    return test_list, test_dict


def demo_show_growth():
    """演示：显示对象增长"""
    print("\n" + "=" * 60)
    print("演示2：显示对象增长")
    print("=" * 60)
    
    initial_stats = objgraph.typestats()
    print("\n[记录初始状态...]")
    
    created_objects = []
    for i in range(100):
        created_objects.append(UserData(i))
    
    print(f"\n[创建了 {len(created_objects)} 个 UserData 对象]")
    print("\n[对象增长 Top 10]:")
    objgraph.show_growth(limit=10, shortnames=False)
    
    del created_objects
    gc.collect()
    
    print("\n[删除并 GC 后的增长]:")
    objgraph.show_growth(initial=initial_stats, limit=10, shortnames=False)


def demo_circular_reference():
    """演示：循环引用检测"""
    print("\n" + "=" * 60)
    print("演示3：循环引用检测")
    print("=" * 60)
    
    gc.collect()
    initial_count = len(gc.get_objects())
    
    print(f"\n初始对象数: {initial_count}")
    
    node_a = Node("A")
    node_b = Node("B")
    node_a.children.append(node_b)
    node_b.parent = node_a
    
    print(f"\n创建循环引用后对象数: {len(gc.get_objects())}")
    
    del node_a, node_b
    gc.collect()
    
    print(f"删除后对象数: {len(gc.get_objects())}")
    
    user_a = UserData(1)
    user_b = UserData(2)
    user_a.data.append(user_b)
    user_b.data.append(user_a)
    
    print(f"\n创建循环引用后的 UserData 对象数: {objgraph.count('UserData')}")
    
    del user_a, user_b
    gc.collect()
    
    print(f"删除后 UserData 对象数: {objgraph.count('UserData')}")


def demo_back_references():
    """演示：反向引用追踪"""
    print("\n" + "=" * 60)
    print("演示4：反向引用追踪")
    print("=" * 60)
    
    global_ref = UserData(999)
    
    local_list = []
    local_list.append(global_ref)
    
    print(f"\n当前 UserData 对象数: {objgraph.count('UserData')}")
    
    user_objects = objgraph.by_type('UserData')
    if user_objects:
        obj = user_objects[0]
        
        print(f"\n对象: {obj}")
        print(f"引用计数: {objgraph.get_objects()[objgraph.get_objects().index(obj)]}")
        
        print(f"\n正在引用此对象的对象:")
        backrefs = objgraph.find_backref_chain(obj, objgraph.is_proper_module)
        if backrefs:
            for ref in backrefs:
                print(f"  - {type(ref).__name__}: {ref}")


def demo_leak_detection():
    """演示：检测泄漏对象"""
    print("\n" + "=" * 60)
    print("演示5：泄漏对象检测")
    print("=" * 60)
    
    leaky_container = []
    
    print("\n[第1轮分配...]")
    for i in range(10):
        leaky_container.append(UserData(i))
    
    print(f"UserData 对象数: {objgraph.count('UserData')}")
    
    print("\n[第2轮分配...]")
    for i in range(10, 20):
        leaky_container.append(UserData(i))
    
    print(f"UserData 对象数: {objgraph.count('UserData')}")
    
    print("\n[第3轮分配...]")
    for i in range(20, 30):
        leaky_container.append(UserData(i))
    
    print(f"UserData 对象数: {objgraph.count('UserData')}")
    
    print("\n[增长分析]:")
    objgraph.show_growth(limit=5)
    
    print("\n提示：UserData 对象持续增长，可能存在泄漏！")
    print(f"泄漏源: leaky_container 持有 {len(leaky_container)} 个对象")


def demo_type_comparison():
    """演示：自定义类型分析"""
    print("\n" + "=" * 60)
    print("演示6：自定义类型分析")
    print("=" * 60)
    
    print("\n[创建对象前]:")
    print(f"UserData: {objgraph.count('UserData')}")
    print(f"Node: {objgraph.count('Node')}")
    
    users = [UserData(i) for i in range(50)]
    nodes = [Node(f'node_{i}') for i in range(30)]
    
    print("\n[创建对象后]:")
    print(f"UserData: {objgraph.count('UserData')}")
    print(f"Node: {objgraph.count('Node')}")
    
    print("\n[只显示自定义类型增长]:")
    objgraph.show_growth(limit=10, shortnames=False)
    
    del users[:25]
    del nodes[:15]
    gc.collect()
    
    print("\n[删除部分对象后]:")
    print(f"UserData: {objgraph.count('UserData')}")
    print(f"Node: {objgraph.count('Node')}")


if __name__ == '__main__':
    print("=" * 60)
    print("objgraph 对象引用图示例")
    print("=" * 60)
    
    demo_most_common_types()
    demo_show_growth()
    demo_circular_reference()
    demo_back_references()
    demo_leak_detection()
    demo_type_comparison()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
