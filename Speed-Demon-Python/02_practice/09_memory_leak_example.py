#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
常见内存泄漏模式和修复方法
运行方式：python 09_memory_leak_example.py
"""

import gc
import weakref
from collections import OrderedDict
import tracemalloc


class LRUCache:
    """使用 OrderedDict 实现的 LRU 缓存（修复方案）"""
    
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def size(self):
        return len(self.cache)


class EventManagerSafe:
    """安全的事件管理器，支持注销监听器"""
    
    def __init__(self):
        self.listeners = []
    
    def register(self, listener):
        self.listeners.append(listener)
    
    def unregister(self, listener):
        if listener in self.listeners:
            self.listeners.remove(listener)
    
    def count(self):
        return len(self.listeners)


class WeakEventManager:
    """使用弱引用的事件管理器"""
    
    def __init__(self):
        self.listeners = []
    
    def register(self, listener):
        self.listeners.append(weakref.ref(listener))
    
    def get_active_listeners(self):
        active = []
        for ref in self.listeners:
            listener = ref()
            if listener is not None:
                active.append(listener)
        return active
    
    def count(self):
        return len(self.get_active_listeners())


class Resource:
    """资源类，使用上下文管理器"""
    
    def __init__(self, name):
        self.name = name
        self.data = [i for i in range(1000)]
        print(f"资源 {name} 已分配")
    
    def cleanup(self):
        print(f"资源 {self.name} 已释放")
        self.data.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def demo_global_container_leak():
    """演示1：全局容器累积泄漏"""
    print("\n" + "=" * 60)
    print("模式1：全局容器累积泄漏")
    print("=" * 60)
    
    leaky_cache = {}
    
    print("\n[泄漏模式 - 只添加不清理]:")
    for i in range(5):
        leaky_cache[f'key_{i}'] = [j for j in range(1000)]
        print(f"  迭代 {i+1}: 缓存大小 = {len(leaky_cache)}")
    
    print(f"\n泄漏结果: 共有 {len(leaky_cache)} 项数据在内存中")
    del leaky_cache
    gc.collect()
    
    print("\n[修复方案 - 使用 LRU 缓存]:")
    safe_cache = LRUCache(max_size=3)
    for i in range(5):
        safe_cache.put(f'key_{i}', [j for j in range(1000)])
        print(f"  迭代 {i+1}: 缓存大小 = {safe_cache.size()}")
    
    print("修复结果: 缓存大小被限制在 max_size 以内，旧数据自动被清理")


def demo_closure_leak():
    """演示2：闭包持有外部引用泄漏"""
    print("\n" + "=" * 60)
    print("模式2：闭包持有外部引用泄漏")
    print("=" * 60)
    
    def create_big_data_callback_leaky():
        """泄漏模式 - 闭包持有整个大对象"""
        big_data = {
            'id': 1,
            'name': 'test',
            'data': [i for i in range(10000)],
            'metadata': {'version': 1, 'author': 'test'}
        }
        
        def callback(x):
            return big_data['id'] * x
        
        return callback
    
    print("\n[泄漏模式 - 闭包持有整个大对象]:")
    callback_leaky = create_big_data_callback_leaky()
    print(f"  回调函数结果: {callback_leaky(10)}")
    print("  问题: big_data 整个对象仍在内存中，虽然只用到了 'id'")
    
    def create_big_data_callback_safe():
        """修复方案 - 只提取需要的值，显式删除大对象"""
        big_data = {
            'id': 1,
            'name': 'test',
            'data': [i for i in range(10000)],
            'metadata': {'version': 1, 'author': 'test'}
        }
        
        result_id = big_data['id']
        del big_data
        gc.collect()
        
        def callback(x):
            return result_id * x
        
        return callback
    
    print("\n[修复方案 - 只提取需要的值]:")
    callback_safe = create_big_data_callback_safe()
    print(f"  回调函数结果: {callback_safe(10)}")
    print("  修复: big_data 已被删除，内存已释放")


def demo_event_listener_leak():
    """演示3：未注销的事件监听器泄漏"""
    print("\n" + "=" * 60)
    print("模式3：未注销的事件监听器泄漏")
    print("=" * 60)
    
    class Component:
        def __init__(self, name):
            self.name = name
            self.data = [i for i in range(1000)]
        
        def on_event(self, event):
            pass
    
    leaky_manager = EventManagerSafe()
    
    print("\n[泄漏模式 - 只注册不注销]:")
    for i in range(5):
        comp = Component(f'comp_{i}')
        leaky_manager.register(comp.on_event)
        print(f"  注册组件 {i}，监听器数: {leaky_manager.count()}")
    
    print(f"泄漏结果: {leaky_manager.count()} 个监听器仍被持有")
    del leaky_manager
    gc.collect()
    
    print("\n[修复方案1 - 支持注销]:")
    safe_manager = EventManagerSafe()
    components = []
    for i in range(5):
        comp = Component(f'comp_{i}')
        components.append(comp)
        safe_manager.register(comp.on_event)
        print(f"  注册组件 {i}，监听器数: {safe_manager.count()}")
    
    for comp in components:
        safe_manager.unregister(comp.on_event)
        print(f"  注销组件，监听器数: {safe_manager.count()}")
    
    print("\n[修复方案2 - 使用弱引用]:")
    weak_manager = WeakEventManager()
    for i in range(5):
        comp = Component(f'weak_comp_{i}')
        weak_manager.register(comp)
        print(f"  注册组件 {i}，活跃监听器数: {weak_manager.count()}")
    
    del comp
    gc.collect()
    print(f"  删除组件后，活跃监听器数: {weak_manager.count()}")
    print("  修复: 弱引用不会阻止对象被垃圾回收")


def demo_circular_reference_leak():
    """演示4：循环引用 + __del__ 泄漏"""
    print("\n" + "=" * 60)
    print("模式4：循环引用 + __del__ 泄漏")
    print("=" * 60)
    
    class NodeLeaky:
        """泄漏模式 - 有 __del__ 方法 + 循环引用"""
        
        def __init__(self, name):
            self.name = name
            self.children = []
            self.parent = None
        
        def __del__(self):
            print(f"    NodeLeaky {self.name} 被清理")
    
    print("\n[泄漏模式 - __del__ + 循环引用]:")
    gc.collect()
    gc.set_debug(gc.DEBUG_SAVEALL)
    
    a = NodeLeaky("A")
    b = NodeLeaky("B")
    a.children.append(b)
    b.parent = a
    
    print(f"  创建循环引用: A <-> B")
    
    del a, b
    gc.collect()
    
    uncollectable = len(gc.garbage)
    print(f"  不可回收对象数: {uncollectable}")
    print("  问题: __del__ 方法导致循环引用无法被 GC 处理")
    
    gc.set_debug(0)
    gc.garbage.clear()
    gc.collect()
    
    class NodeSafe:
        """修复方案 - 使用 cleanup 方法替代 __del__"""
        
        def __init__(self, name):
            self.name = name
            self.children = []
            self.parent = None
        
        def cleanup(self):
            print(f"    NodeSafe {self.name} 被清理")
            self.children.clear()
            if self.parent:
                self.parent = None
    
    print("\n[修复方案1 - 手动打破循环]:")
    a_safe = NodeSafe("A_safe")
    b_safe = NodeSafe("B_safe")
    a_safe.children.append(b_safe)
    b_safe.parent = a_safe
    
    print(f"  创建循环引用: A_safe <-> B_safe")
    
    a_safe.cleanup()
    b_safe.cleanup()
    
    del a_safe, b_safe
    gc.collect()
    print("  修复: 手动清理引用后，对象可被正常回收")
    
    print("\n[修复方案2 - 使用上下文管理器]:")
    with NodeSafe("WithContext") as node:
        print(f"  在上下文中使用: {node.name}")
    print("  修复: 上下文自动调用 cleanup，确保资源释放")


def demo_context_manager():
    """演示5：使用上下文管理器确保资源释放"""
    print("\n" + "=" * 60)
    print("模式5：资源未正确释放")
    print("=" * 60)
    
    print("\n[泄漏模式 - 忘记释放资源]:")
    try:
        res = Resource("leaky")
        raise Exception("发生错误")
        res.cleanup()
    except Exception as e:
        print(f"  异常: {e}")
    print("  问题: 异常导致 cleanup() 未被调用")
    
    print("\n[修复方案 - 使用上下文管理器]:")
    try:
        with Resource("safe") as res:
            raise Exception("发生错误")
    except Exception as e:
        print(f"  异常: {e}")
    print("  修复: __exit__ 确保 cleanup() 总是被调用")


def demo_memory_leak_detection():
    """演示6：使用 tracemalloc 检测内存泄漏"""
    print("\n" + "=" * 60)
    print("实战：使用 tracemalloc 检测内存泄漏")
    print("=" * 60)
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    leaky_list = []
    
    print("\n[模拟多次调用...]")
    for iteration in range(5):
        for i in range(100):
            leaky_list.append(f'item_{iteration}_{i}' * 10)
        
        snapshot2 = tracemalloc.take_snapshot()
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
        print(f"  迭代 {iteration + 1}: 内存增长 = {growth / 1024:.1f} KB")
        
        snapshot1 = snapshot2
    
    print("\n[泄漏分析]:")
    print("  观察: 内存持续线性增长，没有下降")
    print("  结论: 存在内存泄漏 - leaky_list 不断累积数据")
    print("  修复: 设置列表最大长度或定期清理")
    
    tracemalloc.stop()


def demo_best_practices():
    """演示7：预防内存泄漏的最佳实践"""
    print("\n" + "=" * 60)
    print("预防内存泄漏的最佳实践总结")
    print("=" * 60)
    
    practices = [
        ("使用上下文管理器 (with 语句)", "确保资源总是被释放"),
        ("限制容器大小", "使用 LRU 缓存或设置最大容量"),
        ("及时注销监听器", "注册和注销要成对出现"),
        ("考虑使用弱引用", "避免不必要的强引用持有"),
        ("避免 __del__ 方法", "使用显式的 cleanup 方法"),
        ("定期做内存检查", "长时间运行的服务要监控内存趋势"),
        ("慎用全局变量", "全局容器最容易累积泄漏"),
        ("注意闭包引用", "只捕获真正需要的变量"),
    ]
    
    print("\n[最佳实践清单]:")
    for i, (practice, description) in enumerate(practices, 1):
        print(f"  {i}. {practice}")
        print(f"     - {description}")


if __name__ == '__main__':
    print("=" * 60)
    print("常见内存泄漏模式和修复方法")
    print("=" * 60)
    
    demo_global_container_leak()
    demo_closure_leak()
    demo_event_listener_leak()
    demo_circular_reference_leak()
    demo_context_manager()
    demo_memory_leak_detection()
    demo_best_practices()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
