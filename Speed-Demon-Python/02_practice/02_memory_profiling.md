# Python 内存分析工具详解

内存优化是Python性能优化的重要组成部分。本文将介绍四种常用的内存分析工具，帮助你准确定位内存使用热点、发现和修复内存泄漏。

## 目录
1. [memory_profiler - 逐行内存分析](#1-memory_profiler---逐行内存分析)
2. [tracemalloc - 内存快照对比](#2-tracemalloc---内存快照对比)
3. [objgraph - 对象引用图](#3-objgraph---对象引用图)
4. [pympler - 对象大小分析](#4-pympler---对象大小分析)
5. [常见内存泄漏模式和修复方法](#5-常见内存泄漏模式和修复方法)

---

## 1. memory_profiler - 逐行内存分析

memory_profiler 可以逐行显示代码的内存使用情况，类似于 line_profiler，但针对内存而非CPU时间。

### 1.1 安装

```bash
pip install memory-profiler
```

### 1.2 使用方式

#### 方式一：使用装饰器

在代码中使用 `@profile` 装饰器标记需要分析的函数：

```python
@profile
def process_large_data():
    data = []
    for i in range(100000):
        data.append(i * i)
    return data
```

然后运行：
```bash
python -m memory_profiler 05_memory_profiler_example.py
```

#### 方式二：时间线可视化

```bash
# 生成内存使用时间线图
mprof run 05_memory_profiler_example.py
mprof plot
```

### 1.3 结果分析

运行：
```bash
python -m memory_profiler 05_memory_profiler_example.py
```

典型输出：
```
Filename: 05_memory_profiler_example.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     5   40.234 MiB   40.234 MiB           1   @profile
     6                                         def memory_intensive_function():
     7   40.234 MiB    0.000 MiB           1       result = []
     8   47.871 MiB    0.016 MiB       10001       for i in range(10000):
     9   47.871 MiB    0.180 MiB       10000           data = [i] * 100
    10   47.871 MiB    7.441 MiB       10000           result.append(data)
    11   47.871 MiB    0.000 MiB           1       return result
```

#### 字段说明：

| 字段 | 说明 |
|------|------|
| `Line #` | 行号 |
| `Mem usage` | 执行到该行时的总内存使用量（MiB） |
| `Increment` | 该行执行导致的内存增量 |
| `Occurrences` | 该行被执行的次数 |
| `Line Contents` | 代码内容 |

#### 关键分析技巧：

1. **关注 `Increment` 大的行**：这些行分配了大量内存
2. **查看内存使用曲线**：使用 `mprof plot` 观察内存随时间的变化趋势
3. **对比峰值内存**：了解程序最大内存占用

### 1.4 内存时间线可视化

```bash
# 运行并记录内存使用
mprof run 05_memory_profiler_example.py

# 查看所有记录
mprof list

# 绘制最新记录的图表
mprof plot
```

这会生成一个内存使用时间线图，可以清晰地看到：
- 内存分配的时间点
- 内存峰值
- 内存释放的时机
- 是否存在内存持续增长的情况

---

## 2. tracemalloc - 内存快照对比

tracemalloc 是 Python 3.4+ 内置的内存分析工具，可以跟踪内存分配、对比两个时间点的内存快照，找出内存增长的来源。

### 2.1 安装

tracemalloc 是 Python 标准库的一部分，无需额外安装。

### 2.2 基本使用

#### 方式一：简单跟踪

```python
import tracemalloc

# 开始跟踪
tracemalloc.start()

# 执行代码
allocate_some_memory()

# 获取当前内存统计
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

# 打印前10个内存分配最多的位置
print("[Top 10 memory usage]")
for stat in top_stats[:10]:
    print(stat)
```

#### 方式二：快照对比（推荐）

```python
import tracemalloc

tracemalloc.start()

# 拍摄第一个快照
snapshot1 = tracemalloc.take_snapshot()

# 执行业务代码
run_suspicious_code()

# 拍摄第二个快照
snapshot2 = tracemalloc.take_snapshot()

# 对比两个快照，找出内存增长
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
```

### 2.3 结果分析

运行：
```bash
python 06_tracemalloc_example.py
```

典型输出：
```
=== 内存分配最多的位置 ===
06_tracemalloc_example.py:15: size=5120 KiB (+5120 KiB), count=100000, average=52 B
06_tracemalloc_example.py:23: size=2048 KiB (+2048 KiB), count=50000, average=42 B

=== 内存增长对比 ===
06_tracemalloc_example.py:38: size=1536 KiB (+1536 KiB), count=30000, average=52 B
  .../python3.9/linecache.py:137: size=0 B (-856 KiB), count=0 (-21)
```

#### 关键字段说明：

- `size`：分配的总内存大小
- `count`：分配的对象数量
- `average`：平均每个对象的大小
- `+/-`：相对于前一个快照的变化量

### 2.4 高级用法：追踪调用栈

```python
# 获取调用栈信息
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('traceback')

# 显示完整调用栈
for stat in top_stats[:5]:
    print(f"Size: {stat.size / 1024:.1f} KiB")
    for line in stat.traceback.format():
        print(f"  {line}")
```

### 2.5 最佳实践

1. **关键节点拍照**：在程序初始化前后、业务逻辑执行前后拍照
2. **多次对比**：不要只对比一次，多次对比确认趋势
3. **过滤无关信息**：使用 `filter_traces()` 过滤掉系统库的分配

---

## 3. objgraph - 对象引用图

objgraph 可以生成对象引用关系图，帮助你理解为什么某些对象没有被垃圾回收，是排查内存泄漏的利器。

### 3.1 安装

```bash
pip install objgraph

# 如果需要生成图片（需要安装 graphviz）
# Windows: choco install graphviz
# Linux: sudo apt-get install graphviz
# macOS: brew install graphviz
```

### 3.2 基本使用

#### 方式一：统计对象数量

```python
import objgraph

# 显示最常见的对象类型
objgraph.show_most_common_types(limit=20)

# 只显示特定类型
objgraph.show_most_common_types(objects=[list, dict, str])
```

#### 方式二：查看对象增长

```python
# 记录初始状态
initial_counts = objgraph.typestats()

# 执行可疑代码
run_leaky_code()

# 对比增长
objgraph.show_growth(initial=initial_counts, limit=10)
```

#### 方式三：生成引用关系图

```python
# 查找某个对象的引用链
leaky_objects = objgraph.by_type('MyClass')
if leaky_objects:
    # 生成引用图
    objgraph.show_refs(
        leaky_objects[0],
        filename='refs.png',
        refcounts=True,
        max_depth=5
    )
    
    # 生成被引用图（谁在引用这个对象）
    objgraph.show_backrefs(
        leaky_objects[0],
        filename='backrefs.png',
        refcounts=True,
        max_depth=5
    )
```

### 3.3 结果分析

运行：
```bash
python 07_objgraph_example.py
```

典型输出：
```
=== 最常见的对象类型 ===
function                   19832
dict                       11245
list                        8762
tuple                       6543

=== 执行后的增长 ===
list       9234    +472
dict       11897   +652
MyClass    1500    +1500  # 这个很可疑！
```

#### 图形解读：

- **show_refs.png**：显示这个对象引用了哪些其他对象
- **show_backrefs.png**：显示哪些对象在引用这个对象
- **红色箭头**：表示增加的引用计数
- **节点大小**：对应引用计数的多少

### 3.4 实战技巧

1. **找增长最快的类型**：`show_growth()` 是发现泄漏的第一步
2. **关注自定义类**：如果自定义类的实例只增不减，很可能是泄漏
3. **回溯引用链**：用 `show_backrefs()` 找到谁在持有引用
4. **注意循环引用**：虽然Python有GC，但循环引用+__del__会导致泄漏

---

## 4. pympler - 对象大小分析

pympler 提供了更精确的对象大小计算，可以递归计算对象及其引用的所有对象的总大小。

### 4.1 安装

```bash
pip install pympler
```

### 4.2 核心功能

#### 4.2.1 asizeof - 精确计算对象大小

```python
from pympler import asizeof

# 计算单个对象的大小
my_list = [1, 2, 3, 'hello', {'key': 'value'}]
size = asizeof.asizeof(my_list)
print(f"Total size: {size / 1024:.2f} KiB")

# 详细报告
print(asizeof.asized(my_list, detail=1).format())
```

#### 4.2.2 tracker - 跟踪内存变化

```python
from pympler.tracker import SummaryTracker

tracker = SummaryTracker()

# ... 执行代码 ...

tracker.print_diff()
```

#### 4.2.3 muppy - 内存使用分析

```python
from pympler import muppy, summary

# 获取所有对象
all_objects = muppy.get_objects()

# 生成摘要
sum1 = summary.summarize(all_objects)
summary.print_(sum1)

# 对比两个时间点
sum2 = summary.summarize(muppy.get_objects())
diff = summary.get_diff(sum1, sum2)
summary.print_(diff)
```

### 4.3 结果分析

运行：
```bash
python 08_pympler_example.py
```

典型输出：
```
=== asizeof 详细报告 ===
[1, 2, 3, 'hello', {'key': 'value'}] size=864 bytes
  +-- referenced objs: 6
  |   [1] 28 bytes
  |   [2] 28 bytes
  |   [3] 28 bytes
  |   'hello' 56 bytes
  |   {'key': 'value'} 232 bytes

=== tracker 差异 ===
                          types |   # objects |   total size
=============================== | =========== | ============
                           list |        1234 |    512.34 KB
                            str |        5678 |    256.78 KB
                           dict |         890 |    128.56 KB
```

### 4.4 高级用法：内存趋势图

```python
from pympler.classtracker import ClassTracker

tracker = ClassTracker()
tracker.track_class(MyClass)

# 执行操作
for i in range(10):
    create_objects()
    tracker.create_snapshot()

# 生成统计报告
stats = tracker.stats
stats.print_summary()
stats.plot()  # 需要 matplotlib
```

---

## 5. 常见内存泄漏模式和修复方法

内存泄漏是指程序中已不再使用的内存没有被释放，导致内存占用持续增长。

### 5.1 模式一：全局容器累积

**问题代码：**
```python
cache = {}

def process_data(item):
    result = expensive_computation(item)
    # 问题：只添加不清理
    cache[item.id] = result
    return result
```

**特征：**
- `dict` 或 `list` 的数量持续增长
- 每次调用后内存都有增加且不回落

**修复方法：**
```python
from collections import OrderedDict

# 使用LRU缓存，自动清理旧数据
class LRUCache:
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
```

### 5.2 模式二：闭包持有外部引用

**问题代码：**
```python
def create_callback():
    big_data = load_huge_dataset()
    
    def callback(x):
        # 虽然只用到了 big_data 的一个属性
        # 但整个 big_data 都被闭包持有
        return big_data.some_attribute * x
    
    return callback
```

**修复方法：**
```python
def create_callback():
    big_data = load_huge_dataset()
    result = big_data.some_attribute
    # 显式删除不再需要的大对象
    del big_data
    
    def callback(x):
        return result * x
    
    return callback
```

### 5.3 模式三：未注销的事件监听器

**问题代码：**
```python
class EventManager:
    def __init__(self):
        self.listeners = []
    
    def register(self, listener):
        self.listeners.append(listener)
    
    # 缺少 unregister 方法！

manager = EventManager()

class MyComponent:
    def __init__(self):
        manager.register(self.on_event)
        # 问题：即使组件被销毁，manager 仍持有引用
    
    def on_event(self, event):
        pass
```

**修复方法：**
```python
class EventManager:
    def __init__(self):
        self.listeners = []
    
    def register(self, listener):
        self.listeners.append(listener)
    
    def unregister(self, listener):
        self.listeners.remove(listener)

# 使用弱引用
import weakref

class SafeEventManager:
    def __init__(self):
        self.listeners = []
    
    def register(self, listener):
        # 使用弱引用，不阻止对象被回收
        self.listeners.append(weakref.ref(listener))
```

### 5.4 模式四：循环引用 + __del__

**问题代码：**
```python
class Node:
    def __init__(self):
        self.children = []
    
    def __del__(self):
        # 存在 __del__ 时，循环引用无法被GC
        print("Cleaning up")

a = Node()
b = Node()
a.children.append(b)
b.children.append(a)  # 循环引用
del a, b  # 无法被回收！
```

**修复方法：**
```python
# 方案一：手动打破循环
a.children.remove(b)
b.children.remove(a)
del a, b

# 方案二：避免使用 __del__，改用上下文管理器
class Node:
    def __init__(self):
        self.children = []
    
    def cleanup(self):
        print("Cleaning up")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
```

### 5.5 模式五：C扩展的内存泄漏

**特征：**
- Python层面的工具检测不到明显泄漏
- 但RSS（常驻内存）持续增长

**排查方法：**
```python
import os
import psutil

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# 对比调用前后的内存
before = get_memory_usage()
call_c_extension()
after = get_memory_usage()
print(f"Growth: {after - before:.2f} MB")
```

### 5.6 内存泄漏排查流程

1. **确认存在泄漏**
   - 长时间运行后内存显著增长
   - 使用 `mprof` 或 `psutil` 监控趋势

2. **定位泄漏类型**
   - 使用 `tracemalloc` 对比快照
   - 使用 `objgraph.show_growth()` 看哪种对象在增长

3. **找到引用来源**
   - 用 `objgraph.show_backrefs()` 追踪引用链
   - 检查全局变量、缓存、事件监听器

4. **修复并验证**
   - 实施修复
   - 再次运行验证内存是否稳定

### 5.7 预防内存泄漏的最佳实践

1. **使用上下文管理器** (`with` 语句) 确保资源释放
2. **限制缓存大小**，使用 LRU 或 TTL 策略
3. **及时注销监听器**，考虑使用弱引用
4. **避免循环引用**，或确保循环能被打破
5. **定期做内存检查**，特别是长时间运行的服务
6. **慎用全局变量**，全局容器最容易累积泄漏

---

## 工具选择指南

| 场景 | 推荐工具 | 原因 |
|------|----------|------|
| 想知道哪一行代码分配了最多内存 | memory_profiler | 逐行显示，精确到行 |
| 想对比两个时间点的内存变化 | tracemalloc | 内置工具，快照对比 |
| 怀疑有内存泄漏，想找引用来源 | objgraph | 引用关系可视化 |
| 需要精确计算对象的实际大小 | pympler | 递归计算，包含所有引用 |
| 监控长时间运行的服务内存趋势 | memory_profiler + mprof | 生成时间线图表 |
| 生产环境快速排查 | tracemalloc | 无需额外安装依赖 |

## 内存分析流程

1. **先用 memory_profiler 做初步分析**，了解哪些代码分配内存多
2. **用 tracemalloc 对比快照**，确认哪些部分在持续增长
3. **用 objgraph 深入分析**，找到泄漏对象的引用链
4. **用 pympler 精确计算**，验证修复效果
5. **持续监控**，确保修复后内存稳定

## 注意事项

1. **分析器本身也用内存**：在生产环境使用时注意影响
2. **采样会有误差**：多次运行取平均值
3. **内存碎片也会导致增长**：不一定是泄漏，可能是碎片
4. **Python GC 不是实时的**：可以用 `gc.collect()` 强制回收后再测量
5. **小泄漏可能不重要**：短生命周期程序不用过度纠结
