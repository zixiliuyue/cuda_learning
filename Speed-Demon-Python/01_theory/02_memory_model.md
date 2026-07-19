# Python 内存模型深度解析

## 目录
1. [引用计数原理和循环引用](#1-引用计数原理和循环引用)
2. [垃圾回收机制](#2-垃圾回收机制)
3. [__slots__ 减少内存占用](#3-__slots__-减少内存占用)
4. [内存视图 memoryview](#4-内存视图-memoryview)
5. [大数据处理内存优化](#5-大数据处理内存优化)
6. [内存泄漏检测](#6-内存泄漏检测)

---

## 1. 引用计数原理和循环引用

### 1.1 引用计数原理

Python中每个对象都维护一个引用计数，当引用计数为0时，对象被立即销毁。

```python
import sys

a = []
print(sys.getrefcount(a))  # 2 (a本身和getrefcount的参数)

b = a
print(sys.getrefcount(a))  # 3

del b
print(sys.getrefcount(a))  # 2
```

**引用计数增加场景：**
- 对象被创建：`x = 10`
- 对象被赋值：`y = x`
- 对象作为参数传递：`func(x)`
- 对象存储在容器中：`lst = [x]`

**引用计数减少场景：**
- 引用被显式删除：`del x`
- 引用超出作用域（函数返回）
- 引用被重新赋值：`x = None`

### 1.2 循环引用问题

引用计数无法处理循环引用场景：

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# 创建循环引用
a = Node(1)
b = Node(2)
a.next = b
b.next = a  # a和b互相引用

del a
del b
# 此时引用计数都不为0，但这两个对象已无法访问
```

**性能对比：引用计数 vs 标记清除**

```python
import time
import gc

def create_objects(n):
    objects = []
    for i in range(n):
        obj = {"data": i}
        objects.append(obj)
    return objects

# 测试引用计数（自动回收）
start = time.time()
objs = create_objects(1_000_000)
del objs  # 引用计数立即回收
ref_count_time = time.time() - start

# 测试垃圾回收
gc.disable()
start = time.time()
objs = create_objects(1_000_000)
del objs
gc.collect()  # 强制垃圾回收
gc_time = time.time() - start
gc.enable()

print(f"引用计数回收时间: {ref_count_time:.4f}s")
print(f"垃圾回收时间: {gc_time:.4f}s")
# 引用计数通常更快，但无法处理循环引用
```

---

## 2. 垃圾回收机制

### 2.1 分代回收原理

Python采用分代回收策略，将对象分为3代（0, 1, 2）：

- **第0代**：新创建的对象，存活时间短
- **第1代**：经过一次GC后存活的对象
- **第2代**：经过多次GC后存活的对象，存活时间长

```python
import gc

# 查看各代阈值
print(f"各代回收阈值: {gc.get_threshold()}")  # 默认 (700, 10, 10)

# 设置回收阈值
gc.set_threshold(1000, 15, 15)
print(f"新的回收阈值: {gc.get_threshold()}")
```

**分代回收触发条件：**
- 第0代：分配次数 - 释放次数 > 阈值（默认700）
- 第1代：第0代回收次数 > 阈值（默认10）
- 第2代：第1代回收次数 > 阈值（默认10）

### 2.2 触发时机

```python
import gc
import time

# 1. 自动触发
print("GC自动触发演示:")
for i in range(1000):
    # 创建大量临时对象
    _ = [f"item_{j}" for j in range(100)]

# 2. 手动触发
gc.collect()  # 完整GC
gc.collect(generation=0)  # 只回收第0代

# 3. 退出时自动触发
def test_gc_trigger():
    large_list = [object() for _ in range(10000)]
    return

start = time.time()
test_gc_trigger()
# 函数返回后，large_list引用计数变为0，立即回收
auto_time = time.time() - start

start = time.time()
large_list = [object() for _ in range(10000)]
del large_list
gc.collect()
manual_time = time.time() - start

print(f"自动回收时间: {auto_time:.6f}s")
print(f"手动回收时间: {manual_time:.6f}s")
```

### 2.3 垃圾回收性能对比

```python
import gc
import time

def benchmark_gc(enabled=True, iterations=100):
    if enabled:
        gc.enable()
    else:
        gc.disable()
    
    total_time = 0
    for _ in range(iterations):
        start = time.time()
        # 创建大量对象
        objects = [{"data": i} for i in range(10000)]
        for i in range(1000):
            objects[i] = objects[i+1]  # 创建一些引用链
        del objects
        if not enabled:
            gc.collect()
        total_time += time.time() - start
    
    gc.enable()
    return total_time / iterations

gc_enabled_avg = benchmark_gc(enabled=True)
gc_disabled_avg = benchmark_gc(enabled=False)

print(f"GC启用平均时间: {gc_enabled_avg:.4f}s")
print(f"GC禁用+手动回收平均时间: {gc_disabled_avg:.4f}s")
print(f"性能差异: {(gc_disabled_avg/gc_enabled_avg - 1) * 100:.1f}%")
```

---

## 3. __slots__ 减少内存占用

### 3.1 原理

默认情况下，Python对象使用`__dict__`存储属性，这会消耗大量内存。`__slots__`告诉Python不使用`__dict__`，只使用固定的属性集合。

```python
import sys

class RegularPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SlottedPoint:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 单个对象内存对比
p1 = RegularPoint(1, 2)
p2 = SlottedPoint(1, 2)

print(f"普通对象大小: {sys.getsizeof(p1) + sys.getsizeof(p1.__dict__)} bytes")
print(f"__slots__对象大小: {sys.getsizeof(p2)} bytes")
# 通常节省 50%+ 内存
```

### 3.2 大规模性能对比

```python
import sys
import time

class RegularUser:
    def __init__(self, id, name, email, age):
        self.id = id
        self.name = name
        self.email = email
        self.age = age

class SlottedUser:
    __slots__ = ('id', 'name', 'email', 'age')
    def __init__(self, id, name, email, age):
        self.id = id
        self.name = name
        self.email = email
        self.age = age

def measure_memory(cls, count=100000):
    start = time.time()
    users = [cls(i, f"User{i}", f"user{i}@test.com", 20 + i % 50) 
             for i in range(count)]
    elapsed = time.time() - start
    
    # 粗略估算内存使用
    sample = users[0]
    if hasattr(sample, '__dict__'):
        per_object = sys.getsizeof(sample) + sys.getsizeof(sample.__dict__)
    else:
        per_object = sys.getsizeof(sample)
    
    total_memory = per_object * count
    return total_memory, elapsed, users

count = 100000
regular_mem, regular_time, regular_users = measure_memory(RegularUser, count)
slotted_mem, slotted_time, slotted_users = measure_memory(SlottedUser, count)

print(f"普通类 - 内存: {regular_mem / 1024 / 1024:.1f}MB, 时间: {regular_time:.3f}s")
print(f"__slots__ - 内存: {slotted_mem / 1024 / 1024:.1f}MB, 时间: {slotted_time:.3f}s")
print(f"内存节省: {(1 - slotted_mem/regular_mem) * 100:.1f}%")
print(f"创建速度提升: {(1 - slotted_time/regular_time) * 100:.1f}%")

# 属性访问速度测试
def access_test(users):
    start = time.time()
    for user in users:
        _ = user.name
        _ = user.email
        user.age += 1
    return time.time() - start

regular_access = access_test(regular_users)
slotted_access = access_test(slotted_users)

print(f"属性访问 - 普通: {regular_access:.4f}s, __slots__: {slotted_access:.4f}s")
print(f"访问速度提升: {(1 - slotted_access/regular_access) * 100:.1f}%")
```

### 3.3 继承中的__slots__

```python
class Base:
    __slots__ = ('x',)

class Derived(Base):
    __slots__ = ('y',)  # 子类需要再次声明__slots__
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

d = Derived(1, 2)
print(d.x, d.y)  # 1 2
# d.__dict__  # AttributeError: 没有__dict__
```

---

## 4. 内存视图 memoryview

### 4.1 原理

`memoryview`允许在不拷贝的情况下访问对象的内存，支持切片操作但不复制数据。

```python
# 常规切片（拷贝）
data = bytearray(b'x' * 1000000)
start = time.time()
sliced = data[100:100000]
copy_time = time.time() - start

# memoryview切片（不拷贝）
mv = memoryview(data)
start = time.time()
sliced_mv = mv[100:100000]
view_time = time.time() - start

print(f"常规切片时间: {copy_time:.6f}s")
print(f"memoryview切片时间: {view_time:.6f}s")
print(f"速度提升: {(copy_time / view_time):.1f}x")
```

### 4.2 内存使用对比

```python
import sys

data = bytearray(b'x' * 10_000_000)  # 10MB数据

# 常规方式
sliced = data[1_000_000:2_000_000]
print(f"常规切片内存: {sys.getsizeof(sliced)} bytes")

# memoryview方式
mv = memoryview(data)
sliced_mv = mv[1_000_000:2_000_000]
print(f"memoryview内存: {sys.getsizeof(sliced_mv)} bytes")
print(f"内存节省: {(1 - sys.getsizeof(sliced_mv)/sys.getsizeof(sliced)) * 100:.1f}%")
```

### 4.3 实际应用：处理大型二进制数据

```python
import struct
import time

def process_without_memoryview(data):
    """不使用memoryview，每次切片都拷贝"""
    results = []
    for i in range(0, len(data) - 8, 8):
        chunk = data[i:i+8]
        num = struct.unpack('Q', chunk)[0]
        results.append(num)
    return results

def process_with_memoryview(data):
    """使用memoryview，零拷贝"""
    mv = memoryview(data)
    results = []
    for i in range(0, len(data) - 8, 8):
        chunk = mv[i:i+8]
        num = struct.unpack('Q', chunk)[0]
        results.append(num)
    return results

# 生成测试数据
data_size = 10_000_000
data = bytearray(data_size)
for i in range(0, data_size, 8):
    struct.pack_into('Q', data, i, i)

# 性能测试
start = time.time()
r1 = process_without_memoryview(data)
no_mv_time = time.time() - start

start = time.time()
r2 = process_with_memoryview(data)
mv_time = time.time() - start

print(f"无memoryview时间: {no_mv_time:.3f}s")
print(f"有memoryview时间: {mv_time:.3f}s")
print(f"性能提升: {(no_mv_time / mv_time):.1f}x")
```

---

## 5. 大数据处理内存优化

### 5.1 生成器 vs 列表

```python
import sys
import time

def create_list(n):
    return [i * 2 for i in range(n)]

def create_generator(n):
    return (i * 2 for i in range(n))

# 内存对比
count = 10_000_000
lst = create_list(count)
gen = create_generator(count)

print(f"列表内存: {sys.getsizeof(lst) / 1024 / 1024:.1f}MB")
print(f"生成器内存: {sys.getsizeof(gen)} bytes")

# 迭代性能对比
start = time.time()
total = sum(lst)
list_time = time.time() - start

start = time.time()
total = sum(gen)
gen_time = time.time() - start

print(f"列表迭代时间: {list_time:.3f}s")
print(f"生成器迭代时间: {gen_time:.3f}s")
```

### 5.2 迭代器处理大文件

```python
def read_file_generator(file_path, chunk_size=8192):
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def process_large_file(file_path):
    total_lines = 0
    for chunk in read_file_generator(file_path):
        total_lines += chunk.count(b'\n')
    return total_lines

# 对比：一次性读取 vs 分块读取
import os

def read_entire_file(file_path):
    with open(file_path, 'rb') as f:
        return f.read().count(b'\n')

# 创建测试文件
test_file = 'large_test.txt'
with open(test_file, 'w') as f:
    for i in range(1_000_000):
        f.write(f"Line {i}: this is a test line\n")

start = time.time()
lines1 = read_entire_file(test_file)
entire_time = time.time() - start

start = time.time()
lines2 = process_large_file(test_file)
chunk_time = time.time() - start

os.remove(test_file)

print(f"一次性读取时间: {entire_time:.3f}s")
print(f"分块读取时间: {chunk_time:.3f}s")
print(f"内存差异显著，分块读取内存占用更小")
```

### 5.3 使用array模块优化数值数组

```python
import array
import sys

# Python列表存储整数
py_list = list(range(1_000_000))

# array模块存储整数
int_array = array.array('i', range(1_000_000))

print(f"Python列表内存: {sys.getsizeof(py_list) + sum(sys.getsizeof(x) for x in py_list[:1000]) / 1000 * 1_000_000 / 1024 / 1024:.1f}MB")
print(f"array数组内存: {int_array.itemsize * len(int_array) / 1024 / 1024:.1f}MB")
print(f"内存节省超过80%!")
```

### 5.4 pandas优化技巧

```python
import pandas as pd
import numpy as np

def optimize_dataframe(df):
    """优化DataFrame内存使用"""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    # 优化object类型为category
    for col in df.select_dtypes(include=['object']):
        num_unique = df[col].nunique()
        if num_unique / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    # 优化数值类型
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"内存从 {start_mem:.2f}MB 优化到 {end_mem:.2f}MB, 节省 {(1 - end_mem/start_mem)*100:.1f}%")
    return df

# 创建测试数据
df = pd.DataFrame({
    'id': range(100_000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 100_000),
    'value': np.random.randn(100_000),
    'status': np.random.choice(['active', 'inactive'], 100_000)
})

optimized_df = optimize_dataframe(df.copy())
```

---

## 6. 内存泄漏检测

### 6.1 tracemalloc 内置模块

```python
import tracemalloc
import time

def memory_leak_example():
    """模拟内存泄漏"""
    cache = []
    
    def add_to_cache(item):
        cache.append(item)  # 缓存不断增长，可能导致内存泄漏
    
    return add_to_cache

def detect_leak():
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    # 执行可能泄漏的代码
    add_item = memory_leak_example()
    for i in range(10000):
        add_item({"data": i})
    
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("[Top 10 memory differences]")
    for stat in top_stats[:10]:
        print(stat)
    
    # 获取内存使用
    current, peak = tracemalloc.get_traced_memory()
    print(f"\n当前内存使用: {current / 1024 / 1024:.1f}MB")
    print(f"峰值内存使用: {peak / 1024 / 1024:.1f}MB")
    
    tracemalloc.stop()

detect_leak()
```

### 6.2 objgraph 分析对象引用

```python
import objgraph
import gc

class LeakyClass:
    instances = []
    
    def __init__(self, name):
        self.name = name
        LeakyClass.instances.append(self)  # 这里造成泄漏

def simulate_leak():
    for i in range(100):
        LeakyClass(f"obj_{i}")

simulate_leak()
gc.collect()

# 查看对象数量
print(f"LeakyClass对象数量: {objgraph.count('LeakyClass')}")
print(f"dict对象数量: {objgraph.count('dict')}")

# 查看增长最多的对象类型
print("\n增长最多的对象:")
objgraph.show_growth(limit=10)

# 查找引用链
leaked = objgraph.by_type('LeakyClass')[0]
print("\n引用链:")
objgraph.show_backrefs([leaked], filename='backrefs.png')
```

### 6.3 内存泄漏常见模式

```python
# 1. 全局缓存导致的泄漏
global_cache = {}

def bad_cache_func(key, value):
    global_cache[key] = value  # 缓存无限增长

# 修复：使用LRU缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def good_cache_func(key):
    return some_expensive_computation(key)

# 2. 监听器/回调未注销
class EventEmitter:
    def __init__(self):
        self.listeners = []
    
    def add_listener(self, callback):
        self.listeners.append(callback)
    
    def remove_listener(self, callback):
        self.listeners.remove(callback)

# 3. 循环引用中的__del__
class CircularA:
    def __init__(self):
        self.other = None
    
    def __del__(self):
        pass  # 有__del__的对象在循环引用中无法被GC回收

class CircularB:
    def __init__(self):
        self.other = None

a = CircularA()
b = CircularB()
a.other = b
b.other = a
del a, b  # 由于__del__，这两个对象会泄漏
```

### 6.4 内存泄漏检测综合示例

```python
import tracemalloc
import gc
import time

class MemoryMonitor:
    def __init__(self):
        self.snapshots = []
    
    def start(self):
        tracemalloc.start()
    
    def snapshot(self, label):
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot))
    
    def report(self):
        print("\n=== 内存使用报告 ===")
        for i in range(1, len(self.snapshots)):
            prev_label, prev_snap = self.snapshots[i-1]
            curr_label, curr_snap = self.snapshots[i]
            
            stats = curr_snap.compare_to(prev_snap, 'lineno')
            print(f"\n{prev_label} -> {curr_label}:")
            for stat in stats[:5]:
                print(f"  {stat}")
    
    def stop(self):
        tracemalloc.stop()

def simulate_possible_leak():
    monitor = MemoryMonitor()
    monitor.start()
    
    # 初始状态
    monitor.snapshot("初始状态")
    
    # 创建一些对象
    cache = []
    for i in range(10000):
        cache.append({"data": i})
    
    monitor.snapshot("添加10k对象")
    
    # 清空缓存（应该回收）
    cache.clear()
    gc.collect()
    
    monitor.snapshot("清空缓存后")
    
    monitor.report()
    monitor.stop()

simulate_possible_leak()
```

---

## 总结：最佳实践

| 技术 | 适用场景 | 预期收益 |
|------|---------|---------|
| 引用计数 | 所有Python代码 | 自动即时回收 |
| `__slots__` | 创建大量对象时 | 节省50%+内存 |
| `memoryview` | 处理二进制数据、大数组 | 零拷贝，内存占用极低 |
| 生成器/迭代器 | 处理大数据集、大文件 | 按需加载，内存可控 |
| `array`模块 | 数值数组 | 内存节省80%+ |
| `tracemalloc` | 内存泄漏检测 | 精确定位泄漏位置 |
| `objgraph` | 对象引用分析 | 查找循环引用 |

### 关键要点：
1. 理解引用计数机制，但不要完全依赖它
2. 对于大量小对象，始终考虑使用`__slots__`
3. 处理二进制数据时优先使用`memoryview`
4. 大数据处理始终使用生成器/迭代器模式
5. 定期使用`tracemalloc`检查内存使用情况
6. 注意全局缓存、事件监听器等常见泄漏点
