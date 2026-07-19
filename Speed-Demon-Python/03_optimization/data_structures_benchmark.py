#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 数据结构性能基准测试
涵盖: list, tuple, array, dict, OrderedDict, defaultdict, set, deque, heapq, bisect
"""

import time
import random
import sys
from collections import OrderedDict, defaultdict, deque
import heapq
import bisect
from array import array

# 测试数据规模
SMALL_SIZE = 1000
MEDIUM_SIZE = 10000
LARGE_SIZE = 100000

# 生成测试数据
random.seed(42)

class Benchmark:
    """基准测试基类"""
    def __init__(self, name):
        self.name = name
        self.results = {}
    
    def run(self):
        raise NotImplementedError
    
    @staticmethod
    def timeit(func, *args, iterations=1000):
        """计时函数"""
        start = time.perf_counter()
        for _ in range(iterations):
            func(*args)
        end = time.perf_counter()
        return (end - start) * 1000  # 返回毫秒


class ListTupleArrayBenchmark(Benchmark):
    """list vs tuple vs array 性能对比"""
    def __init__(self):
        super().__init__("list vs tuple vs array")
    
    def run(self):
        size = LARGE_SIZE
        test_data = [random.randint(0, 1000000) for _ in range(size)]
        
        # 创建性能测试
        print("\n=== 1. list vs tuple vs array 性能对比 ===")
        
        # 创建性能
        list_create_time = self.timeit(lambda: list(test_data), iterations=100)
        tuple_create_time = self.timeit(lambda: tuple(test_data), iterations=100)
        array_create_time = self.timeit(lambda: array('i', test_data), iterations=100)
        
        print(f"\n创建性能 (元素数: {size}):")
        print(f"list:   {list_create_time:.3f} ms")
        print(f"tuple:  {tuple_create_time:.3f} ms")
        print(f"array:  {array_create_time:.3f} ms")
        
        self.results['create'] = {
            'list': list_create_time,
            'tuple': tuple_create_time,
            'array': array_create_time
        }
        
        # 访问性能
        lst = list(test_data)
        tpl = tuple(test_data)
        arr = array('i', test_data)
        indices = [random.randint(0, size-1) for _ in range(1000)]
        
        def access_list():
            for idx in indices:
                _ = lst[idx]
        
        def access_tuple():
            for idx in indices:
                _ = tpl[idx]
        
        def access_array():
            for idx in indices:
                _ = arr[idx]
        
        list_access_time = self.timeit(access_list, iterations=100)
        tuple_access_time = self.timeit(access_tuple, iterations=100)
        array_access_time = self.timeit(access_array, iterations=100)
        
        print(f"\n随机访问性能 (1000次访问 * 100轮):")
        print(f"list:   {list_access_time:.3f} ms")
        print(f"tuple:  {tuple_access_time:.3f} ms")
        print(f"array:  {array_access_time:.3f} ms")
        
        self.results['access'] = {
            'list': list_access_time,
            'tuple': tuple_access_time,
            'array': array_access_time
        }
        
        # 迭代性能
        def iterate_list():
            sum(lst)
        
        def iterate_tuple():
            sum(tpl)
        
        def iterate_array():
            sum(arr)
        
        list_iter_time = self.timeit(iterate_list, iterations=100)
        tuple_iter_time = self.timeit(iterate_tuple, iterations=100)
        array_iter_time = self.timeit(iterate_array, iterations=100)
        
        print(f"\n迭代求和性能:")
        print(f"list:   {list_iter_time:.3f} ms")
        print(f"tuple:  {tuple_iter_time:.3f} ms")
        print(f"array:  {array_iter_time:.3f} ms")
        
        self.results['iterate'] = {
            'list': list_iter_time,
            'tuple': tuple_iter_time,
            'array': array_iter_time
        }
        
        # 内存占用
        print(f"\n内存占用:")
        print(f"list:   {sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in lst[:100])} bytes (仅估算)")
        print(f"tuple:  {sys.getsizeof(tpl) + sum(sys.getsizeof(x) for x in tpl[:100])} bytes (仅估算)")
        print(f"array:  {sys.getsizeof(arr)} bytes")
        
        return self.results


class DictBenchmark(Benchmark):
    """dict vs OrderedDict vs defaultdict 性能对比"""
    def __init__(self):
        super().__init__("dict vs OrderedDict vs defaultdict")
    
    def run(self):
        size = MEDIUM_SIZE
        keys = [f"key_{i}" for i in range(size)]
        values = [random.randint(0, 1000000) for _ in range(size)]
        items = list(zip(keys, values))
        
        print("\n=== 2. dict vs OrderedDict vs defaultdict 性能对比 ===")
        
        # 创建性能
        def create_dict():
            return dict(items)
        
        def create_ordered():
            return OrderedDict(items)
        
        def create_default():
            d = defaultdict(int)
            for k, v in items:
                d[k] = v
            return d
        
        dict_create_time = self.timeit(create_dict, iterations=100)
        ordered_create_time = self.timeit(create_ordered, iterations=100)
        default_create_time = self.timeit(create_default, iterations=100)
        
        print(f"\n创建性能 (键值对: {size}):")
        print(f"dict:         {dict_create_time:.3f} ms")
        print(f"OrderedDict:  {ordered_create_time:.3f} ms")
        print(f"defaultdict:  {default_create_time:.3f} ms")
        
        self.results['create'] = {
            'dict': dict_create_time,
            'OrderedDict': ordered_create_time,
            'defaultdict': default_create_time
        }
        
        # 查找性能
        d = dict(items)
        od = OrderedDict(items)
        dd = defaultdict(int, items)
        test_keys = random.sample(keys, 1000) + ['nonexistent'] * 100
        
        def lookup_dict():
            for k in test_keys:
                _ = d.get(k, -1)
        
        def lookup_ordered():
            for k in test_keys:
                _ = od.get(k, -1)
        
        def lookup_default():
            for k in test_keys:
                _ = dd.get(k, -1)
        
        dict_lookup_time = self.timeit(lookup_dict, iterations=100)
        ordered_lookup_time = self.timeit(lookup_ordered, iterations=100)
        default_lookup_time = self.timeit(lookup_default, iterations=100)
        
        print(f"\n查找性能 (1100次查找 * 100轮):")
        print(f"dict:         {dict_lookup_time:.3f} ms")
        print(f"OrderedDict:  {ordered_lookup_time:.3f} ms")
        print(f"defaultdict:  {default_lookup_time:.3f} ms")
        
        self.results['lookup'] = {
            'dict': dict_lookup_time,
            'OrderedDict': ordered_lookup_time,
            'defaultdict': default_lookup_time
        }
        
        # 插入和删除性能
        def insert_delete_dict():
            d = {}
            for i in range(1000):
                d[f'new_{i}'] = i
            for i in range(1000):
                del d[f'new_{i}']
        
        def insert_delete_ordered():
            od = OrderedDict()
            for i in range(1000):
                od[f'new_{i}'] = i
            for i in range(1000):
                del od[f'new_{i}']
        
        def insert_delete_default():
            dd = defaultdict(int)
            for i in range(1000):
                dd[f'new_{i}'] = i
            for i in range(1000):
                del dd[f'new_{i}']
        
        dict_insert_time = self.timeit(insert_delete_dict, iterations=50)
        ordered_insert_time = self.timeit(insert_delete_ordered, iterations=50)
        default_insert_time = self.timeit(insert_delete_default, iterations=50)
        
        print(f"\n插入+删除性能 (1000次操作 * 50轮):")
        print(f"dict:         {dict_insert_time:.3f} ms")
        print(f"OrderedDict:  {ordered_insert_time:.3f} ms")
        print(f"defaultdict:  {default_insert_time:.3f} ms")
        
        self.results['insert_delete'] = {
            'dict': dict_insert_time,
            'OrderedDict': ordered_insert_time,
            'defaultdict': default_insert_time
        }
        
        return self.results


class SetBenchmark(Benchmark):
    """set 高效操作测试"""
    def __init__(self):
        super().__init__("set 操作性能")
    
    def run(self):
        size = LARGE_SIZE
        print("\n=== 3. set 高效操作性能 ===")
        
        set1 = set(range(size))
        set2 = set(range(size // 2, size + size // 2))
        lst1 = list(range(size))
        lst2 = list(range(size // 2, size + size // 2))
        
        # 成员检查性能
        test_values = [random.randint(0, size * 2) for _ in range(1000)]
        
        def set_membership():
            for v in test_values:
                _ = v in set1
        
        def list_membership():
            for v in test_values:
                _ = v in lst1
        
        set_memb_time = self.timeit(set_membership, iterations=100)
        list_memb_time = self.timeit(list_membership, iterations=100)
        
        print(f"\n成员检查性能 (1000次检查 * 100轮):")
        print(f"set:   {set_memb_time:.3f} ms")
        print(f"list:  {list_memb_time:.3f} ms")
        print(f"set 比 list 快 {list_memb_time / set_memb_time:.1f} 倍")
        
        self.results['membership'] = {
            'set': set_memb_time,
            'list': list_memb_time
        }
        
        # 集合运算性能
        def set_union():
            _ = set1 | set2
        
        def set_intersection():
            _ = set1 & set2
        
        def set_difference():
            _ = set1 - set2
        
        union_time = self.timeit(set_union, iterations=100)
        intersection_time = self.timeit(set_intersection, iterations=100)
        difference_time = self.timeit(set_difference, iterations=100)
        
        print(f"\n集合运算性能:")
        print(f"并集:    {union_time:.3f} ms")
        print(f"交集:    {intersection_time:.3f} ms")
        print(f"差集:    {difference_time:.3f} ms")
        
        self.results['operations'] = {
            'union': union_time,
            'intersection': intersection_time,
            'difference': difference_time
        }
        
        # 去重性能
        duplicates = [random.randint(0, size // 10) for _ in range(size)]
        
        def set_dedup():
            return set(duplicates)
        
        def list_dedup():
            result = []
            seen = set()
            for item in duplicates:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        
        set_dedup_time = self.timeit(set_dedup, iterations=50)
        list_dedup_time = self.timeit(list_dedup, iterations=50)
        
        print(f"\n去重性能:")
        print(f"set:     {set_dedup_time:.3f} ms")
        print(f"list:    {list_dedup_time:.3f} ms")
        
        self.results['dedup'] = {
            'set': set_dedup_time,
            'list': list_dedup_time
        }
        
        return self.results


class DequeBenchmark(Benchmark):
    """deque vs list 两端操作性能对比"""
    def __init__(self):
        super().__init__("deque vs list 两端操作")
    
    def run(self):
        size = MEDIUM_SIZE
        print("\n=== 4. deque vs list 两端操作性能对比 ===")
        
        # 左端添加性能
        def deque_append_left():
            dq = deque()
            for i in range(size // 10):
                dq.appendleft(i)
        
        def list_insert_0():
            lst = []
            for i in range(size // 10):
                lst.insert(0, i)
        
        deque_append_left_time = self.timeit(deque_append_left, iterations=10)
        list_insert_0_time = self.timeit(list_insert_0, iterations=10)
        
        print(f"\n左端添加性能 (10000次添加 * 10轮):")
        print(f"deque.appendleft:  {deque_append_left_time:.3f} ms")
        print(f"list.insert(0):    {list_insert_0_time:.3f} ms")
        print(f"deque 比 list 快 {list_insert_0_time / deque_append_left_time:.1f} 倍")
        
        self.results['append_left'] = {
            'deque': deque_append_left_time,
            'list': list_insert_0_time
        }
        
        # 左端弹出性能
        dq = deque(range(size))
        lst = list(range(size))
        
        def deque_popleft():
            for _ in range(size // 10):
                dq.popleft()
        
        def list_pop0():
            for _ in range(size // 10):
                lst.pop(0)
        
        deque_pop_time = self.timeit(deque_popleft, iterations=10)
        list_pop_time = self.timeit(list_pop0, iterations=10)
        
        print(f"\n左端弹出性能 (10000次弹出 * 10轮):")
        print(f"deque.popleft:     {deque_pop_time:.3f} ms")
        print(f"list.pop(0):       {list_pop_time:.3f} ms")
        print(f"deque 比 list 快 {list_pop_time / deque_pop_time:.1f} 倍")
        
        self.results['popleft'] = {
            'deque': deque_pop_time,
            'list': list_pop_time
        }
        
        # 右端添加性能
        def deque_append():
            dq = deque()
            for i in range(size):
                dq.append(i)
        
        def list_append():
            lst = []
            for i in range(size):
                lst.append(i)
        
        deque_append_time = self.timeit(deque_append, iterations=10)
        list_append_time = self.timeit(list_append, iterations=10)
        
        print(f"\n右端添加性能:")
        print(f"deque.append:      {deque_append_time:.3f} ms")
        print(f"list.append:       {list_append_time:.3f} ms")
        
        self.results['append'] = {
            'deque': deque_append_time,
            'list': list_append_time
        }
        
        return self.results


class HeapqBisectBenchmark(Benchmark):
    """heapq 与 bisect 使用场景测试"""
    def __init__(self):
        super().__init__("heapq 与 bisect")
    
    def run(self):
        size = MEDIUM_SIZE
        print("\n=== 5. heapq 与 bisect 使用场景 ===")
        
        data = [random.randint(0, 1000000) for _ in range(size)]
        
        # 构建堆 vs 排序列表
        def build_heap():
            heap = data.copy()
            heapq.heapify(heap)
            return heap
        
        def build_sorted():
            return sorted(data)
        
        heap_build_time = self.timeit(build_heap, iterations=10)
        sorted_build_time = self.timeit(build_sorted, iterations=10)
        
        print(f"\n构建性能 (数据量: {size}):")
        print(f"heapq.heapify:  {heap_build_time:.3f} ms")
        print(f"sorted:         {sorted_build_time:.3f} ms")
        
        self.results['build'] = {
            'heapq': heap_build_time,
            'sorted': sorted_build_time
        }
        
        # 获取最小元素
        heap = data.copy()
        heapq.heapify(heap)
        sorted_list = sorted(data)
        
        def heap_pop():
            h = heap.copy()
            for _ in range(100):
                heapq.heappop(h)
        
        def list_pop0():
            lst = sorted_list.copy()
            for _ in range(100):
                lst.pop(0)
        
        heap_pop_time = self.timeit(heap_pop, iterations=10)
        list_pop0_time = self.timeit(list_pop0, iterations=10)
        
        print(f"\n获取前100个最小元素:")
        print(f"heapq.heappop:  {heap_pop_time:.3f} ms")
        print(f"list.pop(0):    {list_pop0_time:.3f} ms")
        
        self.results['pop_smallest'] = {
            'heapq': heap_pop_time,
            'list': list_pop0_time
        }
        
        # bisect 二分查找
        search_values = [random.randint(0, 1000000) for _ in range(1000)]
        
        def bisect_search():
            for v in search_values:
                idx = bisect.bisect_left(sorted_list, v)
                if idx < len(sorted_list) and sorted_list[idx] == v:
                    _ = idx
        
        def linear_search():
            for v in search_values:
                for i, item in enumerate(data):
                    if item == v:
                        break
        
        bisect_time = self.timeit(bisect_search, iterations=50)
        linear_time = self.timeit(linear_search, iterations=1)
        
        print(f"\n查找性能 (1000次查找):")
        print(f"bisect (二分):   {bisect_time:.3f} ms")
        print(f"线性查找:        {linear_time:.3f} ms")
        print(f"bisect 比线性查找快 {linear_time / bisect_time:.1f} 倍")
        
        self.results['search'] = {
            'bisect': bisect_time,
            'linear': linear_time
        }
        
        # 有序插入性能
        def bisect_insert():
            lst = sorted_list.copy()
            for i in range(100):
                bisect.insort(lst, random.randint(0, 1000000))
        
        def naive_insert():
            lst = sorted_list.copy()
            for i in range(100):
                val = random.randint(0, 1000000)
                lst.append(val)
                lst.sort()
        
        bisect_insert_time = self.timeit(bisect_insert, iterations=10)
        naive_insert_time = self.timeit(naive_insert, iterations=10)
        
        print(f"\n有序插入性能 (100次插入):")
        print(f"bisect.insort:   {bisect_insert_time:.3f} ms")
        print(f"append + sort:   {naive_insert_time:.3f} ms")
        
        self.results['sorted_insert'] = {
            'bisect': bisect_insert_time,
            'naive': naive_insert_time
        }
        
        return self.results


def print_summary_table(all_results):
    """打印汇总性能对比表格"""
    print("\n" + "=" * 80)
    print("性能对比汇总表 (单位: ms)")
    print("=" * 80)
    
    print("\n【创建性能对比】")
    print(f"{'结构':<15} {'创建':>10} {'访问':>10} {'迭代':>10}")
    print("-" * 45)
    for name in ['list', 'tuple', 'array']:
        create = all_results['list_tuple_array']['create'][name]
        access = all_results['list_tuple_array']['access'][name]
        iterate = all_results['list_tuple_array']['iterate'][name]
        star = " ⭐" if name == 'array' and iterate < all_results['list_tuple_array']['iterate']['list'] else ""
        print(f"{name:<15} {create:>10.3f} {access:>10.3f} {iterate:>10.3f}{star}")
    
    print("\n【字典类型性能对比】")
    print(f"{'结构':<15} {'创建':>10} {'查找':>10} {'插入+删除':>12}")
    print("-" * 47)
    for name in ['dict', 'OrderedDict', 'defaultdict']:
        create = all_results['dict']['create'][name]
        lookup = all_results['dict']['lookup'][name]
        insert = all_results['dict']['insert_delete'][name]
        star = " ⭐" if name == 'dict' else ""
        print(f"{name:<15} {create:>10.3f} {lookup:>10.3f} {insert:>12.3f}{star}")
    
    print("\n【成员检查性能对比】")
    print(f"{'结构':<15} {'成员检查':>12} {'去重':>10}")
    print("-" * 37)
    for name in ['set', 'list']:
        memb = all_results['set']['membership'][name]
        dedup = all_results['set']['dedup'][name]
        star = " ⭐" if name == 'set' else ""
        print(f"{name:<15} {memb:>12.3f} {dedup:>10.3f}{star}")
    
    print("\n【两端操作性能对比】")
    print(f"{'结构':<15} {'左端添加':>10} {'左端弹出':>10} {'右端添加':>10}")
    print("-" * 45)
    for name in ['deque', 'list']:
        append_left = all_results['deque']['append_left'][name]
        popleft = all_results['deque']['popleft'][name]
        append = all_results['deque']['append'][name]
        star = " ⭐" if name == 'deque' else ""
        print(f"{name:<15} {append_left:>10.3f} {popleft:>10.3f} {append:>10.3f}{star}")
    
    print("\n【堆与二分查找性能】")
    print(f"{'操作':<15} {'heapq':>10} {'sorted':>10}")
    print("-" * 35)
    print(f"{'构建':<15} {all_results['heapq_bisect']['build']['heapq']:>10.3f} {all_results['heapq_bisect']['build']['sorted']:>10.3f}")
    print(f"{'弹出100最小':<15} {all_results['heapq_bisect']['pop_smallest']['heapq']:>10.3f} {all_results['heapq_bisect']['pop_smallest']['list']:>10.3f}")
    
    print("\n【查找性能对比】")
    print(f"{'方法':<15} {'时间(ms)':>12}")
    print("-" * 27)
    print(f"{'bisect(二分)':<15} {all_results['heapq_bisect']['search']['bisect']:>12.3f}")
    print(f"{'线性查找':<15} {all_results['heapq_bisect']['search']['linear']:>12.3f}")
    
    print("\n" + "=" * 80)


def main():
    print("Python 数据结构性能基准测试")
    print("=" * 50)
    
    all_results = {}
    
    benchmarks = [
        (ListTupleArrayBenchmark(), 'list_tuple_array'),
        (DictBenchmark(), 'dict'),
        (SetBenchmark(), 'set'),
        (DequeBenchmark(), 'deque'),
        (HeapqBisectBenchmark(), 'heapq_bisect')
    ]
    
    for benchmark, key in benchmarks:
        all_results[key] = benchmark.run()
    
    print_summary_table(all_results)
    
    print("\n所有基准测试完成！")
    return all_results


if __name__ == "__main__":
    main()
