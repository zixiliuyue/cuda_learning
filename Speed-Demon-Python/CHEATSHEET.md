# Python 性能优化速查表

## 一、性能分析工具

### 1.1 时间测量

```python
import time
import timeit

# 简单计时
start = time.time()
# 代码块
elapsed = time.time() - start

# timeit
timeit.timeit('func()', setup='from __main__ import func', number=1000)
```

### 1.2 cProfile 性能分析

```python
import cProfile

# 直接运行
cProfile.run('my_function()')

# 保存到文件
cProfile.run('my_function()', 'profile_stats')

# 分析结果
import pstats
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
stats.sort_stats('time').print_stats(10)
```

### 1.3 line_profiler 逐行分析

```python
# 安装：pip install line_profiler

# 使用装饰器
@profile
def slow_function():
    pass

# 命令行运行
# kernprof -l -v script.py
```

### 1.4 memory_profiler 内存分析

```python
# 安装：pip install memory_profiler

# 使用装饰器
@profile
def memory_heavy():
    pass

# 命令行运行
# python -m memory_profiler script.py
```

## 二、基础优化技巧

### 2.1 循环优化

```python
# 不好的写法
result = []
for i in range(1000):
    result.append(i * 2)

# 好的写法 - 列表推导
result = [i * 2 for i in range(1000)]

# 更好 - 向量化
import numpy as np
result = np.arange(1000) * 2
```

### 2.2 字符串拼接

```python
# 不好的写法
result = ''
for s in strings:
    result += s

# 好的写法
result = ''.join(strings)
```

### 2.3 字典查找

```python
# 使用 get 方法设置默认值
value = my_dict.get(key, default_value)

# 批量查找
keys = ['a', 'b', 'c']
values = [my_dict[k] for k in keys if k in my_dict]
```

### 2.4 集合操作

```python
# 成员检查 - set 比 list 快
my_set = set(items)
if item in my_set:
    pass

# 集合运算
common = set1 & set2
union = set1 | set2
diff = set1 - set2
```

## 三、内存优化

### 3.1 生成器 vs 列表

```python
# 列表 - 立即计算全部
result = [x for x in big_data if x > 0]

# 生成器 - 惰性计算
result = (x for x in big_data if x > 0)

# 生成器函数
def process_data(data):
    for item in data:
        yield process(item)
```

### 3.2 __slots__ 减少内存

```python
class MyClass:
    __slots__ = ['name', 'value']
    
    def __init__(self, name, value):
        self.name = name
        self.value = value
```

### 3.3 数组代替列表

```python
import array

# 数值数组
arr = array.array('i', [1, 2, 3, 4, 5])

# NumPy 数组
import numpy as np
np_arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
```

### 3.4 弱引用

```python
import weakref

# 创建弱引用
ref = weakref.ref(obj)

# 弱引用字典
cache = weakref.WeakKeyDictionary()
cache[obj] = value
```

## 四、缓存与记忆化

### 4.1 lru_cache

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### 4.2 自定义缓存

```python
cache = {}

def expensive_func(key):
    if key in cache:
        return cache[key]
    
    result = compute(key)
    cache[key] = result
    return result
```

## 五、并发编程

### 5.1 多线程（I/O 密集）

```python
import threading

def worker():
    pass

threads = []
for _ in range(10):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

### 5.2 多进程（CPU 密集）

```python
import multiprocessing

def worker():
    pass

processes = []
for _ in range(4):
    p = multiprocessing.Process(target=worker)
    p.start()
    processes.append(p)

for p in processes:
    p.join()
```

### 5.3 线程池/进程池

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# 线程池
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(func, items))

# 进程池
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(func, items))
```

### 5.4 asyncio 异步

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return data

async def main():
    tasks = [fetch_data() for _ in range(10)]
    results = await asyncio.gather(*tasks)

asyncio.run(main())
```

## 六、Numba 加速

### 6.1 JIT 编译

```python
from numba import jit

@jit(nopython=True)
def fast_function(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result
```

### 6.2 并行循环

```python
from numba import jit, prange

@jit(nopython=True, parallel=True)
def parallel_sum(arr):
    result = 0
    for i in prange(len(arr)):
        result += arr[i]
    return result
```

## 七、NumPy 向量化

### 7.1 避免循环

```python
import numpy as np

# 不好的写法
result = []
for x in arr:
    result.append(x * 2)

# 好的写法
result = arr * 2
```

### 7.2 广播机制

```python
# 自动扩展维度
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
result = a + b
```

### 7.3 内置函数

```python
# 使用内置函数，不要自己写
total = np.sum(arr)
mean = np.mean(arr)
std = np.std(arr)
```

## 八、I/O 优化

### 8.1 文件读写

```python
# 批量读取
with open('file.txt', 'r') as f:
    lines = f.readlines()

# 使用缓冲区
with open('file.txt', 'r', buffering=1024*1024) as f:
    pass
```

### 8.2 批量数据库操作

```python
# 使用 executemany
cursor.executemany(
    'INSERT INTO table VALUES (?, ?)',
    data
)
```

## 九、常见性能瓶颈

| 瓶颈类型 | 常见原因 | 优化方向 |
|---------|---------|---------|
| CPU 密集 | 复杂循环、大量计算 | Numba、多进程、向量化 |
| I/O 密集 | 文件、网络、数据库 | 异步、批量、缓存 |
| 内存密集 | 大数据集合、重复创建 | 生成器、__slots__、数组 |
| 算法低效 | O(n²) 算法 | 优化算法、使用合适数据结构 |

## 十、优化原则

1. **先测量，后优化** - 不要猜测瓶颈
2. **正确性优先** - 先保证代码正确
3. **可读性重要** - 不要为了性能牺牲可维护性
4. **渐进式优化** - 从最明显的瓶颈开始
5. **验证效果** - 优化后一定要验证性能提升

## 十一、常用命令

```bash
# 性能分析
python -m cProfile -s cumulative script.py

# 内存分析
python -m memory_profiler script.py

# line_profiler
kernprof -l -v script.py

# 查看字节码
python -m dis script.py
```
