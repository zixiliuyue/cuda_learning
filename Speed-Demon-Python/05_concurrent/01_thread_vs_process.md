# 多线程 vs 多进程：Python 并发编程深度解析

## 目录

1. [多线程适用场景（I/O 密集）](#1-多线程适用场景io-密集)
2. [多进程适用场景（CPU 密集）](#2-多进程适用场景cpu-密集)
3. [ThreadPoolExecutor vs ProcessPoolExecutor](#3-threadpoolexecutor-vs-processpoolexecutor)
4. [GIL 对多线程的影响](#4-gil对多线程的影响)
5. [进程间通信（队列、管道）](#5-进程间通信队列管道)
6. [线程安全和锁](#6-线程安全和锁)
7. [并发数选择策略](#7-并发数选择策略)
8. [性能对比测试](#8-性能对比测试)

---

## 1. 多线程适用场景（I/O 密集）

### 什么是 I/O 密集型任务？

I/O 密集型任务的特点是：
- 程序大部分时间在等待 I/O 操作完成
- CPU 利用率相对较低
- 常见场景：网络请求、文件读写、数据库查询、API 调用

### 为什么多线程适合 I/O 密集型任务？

在 I/O 等待期间，线程会被阻塞，此时 GIL（全局解释器锁）会被释放，允许其他线程执行。这样可以充分利用等待时间，提高整体效率。

**线程在等待 I/O 时释放 GIL → 其他线程可以执行**

### 典型应用场景

```python
# 场景1：并发网络请求
import requests
urls = ["https://api.example.com/data1", "https://api.example.com/data2"]

# 场景2：并发文件读写
files = ["large_file1.txt", "large_file2.txt", "large_file3.txt"]

# 场景3：并发数据库查询
queries = ["SELECT * FROM users", "SELECT * FROM orders", "SELECT * FROM products"]
```

### 多线程 I/O 密集型优势

| 指标 | 单线程 | 多线程 |
|------|--------|--------|
| 执行时间 | 长（串行等待） | 短（并行等待） |
| CPU 利用率 | 低 | 中高 |
| 内存开销 | 小 | 较小 |
| 编程复杂度 | 低 | 中等 |

---

## 2. 多进程适用场景（CPU 密集）

### 什么是 CPU 密集型任务？

CPU 密集型任务的特点是：
- 程序大部分时间在进行计算
- CPU 利用率很高
- 常见场景：数学计算、图像处理、数据压缩、机器学习训练

### 为什么多进程适合 CPU 密集型任务？

由于 GIL 的存在，Python 的多线程无法真正利用多核 CPU。而多进程每个进程都有独立的 Python 解释器和 GIL，可以真正并行执行。

**每个进程有独立的 GIL → 可以利用多核 CPU**

### 典型应用场景

```python
# 场景1：大规模数学计算
def calculate_pi(n):
    s = 0.0
    for i in range(n):
        s += (-1)**i / (2*i + 1)
    return 4 * s

# 场景2：图像处理
from PIL import Image
def process_image(image_path):
    img = Image.open(image_path)
    # 复杂的图像处理操作
    return img.resize((1000, 1000)).rotate(90)

# 场景3：数据处理
import pandas as pd
def process_dataframe(df):
    # 复杂的数据转换和聚合操作
    return df.groupby('category').agg({'value': 'sum'})
```

### 多进程 CPU 密集型优势

| 指标 | 多线程 | 多进程 |
|------|--------|--------|
| 执行时间 | 长（GIL 限制） | 短（真正并行） |
| CPU 利用率 | 单核 | 多核 |
| 内存开销 | 较小 | 较大 |
| 编程复杂度 | 中等 | 较高 |

---

## 3. ThreadPoolExecutor vs ProcessPoolExecutor

### ThreadPoolExecutor

**特点：**
- 基于线程池实现
- 线程之间共享内存空间
- 创建开销小
- 受 GIL 限制

**适用场景：**
- I/O 密集型任务
- 任务之间需要共享数据
- 任务创建频繁

**基本用法：**
```python
from concurrent.futures import ThreadPoolExecutor

def task(url):
    return requests.get(url).json()

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(task, urls))
```

### ProcessPoolExecutor

**特点：**
- 基于进程池实现
- 每个进程有独立的内存空间
- 创建开销大
- 不受 GIL 限制

**适用场景：**
- CPU 密集型任务
- 任务之间独立性高
- 需要利用多核 CPU

**基本用法：**
```python
from concurrent.futures import ProcessPoolExecutor

def heavy_calculation(n):
    return sum(i*i for i in range(n))

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(heavy_calculation, [10**7]*4))
```

### 核心对比表

| 特性 | ThreadPoolExecutor | ProcessPoolExecutor |
|------|-------------------|---------------------|
| **并行方式** | 并发（受 GIL 限制） | 真正并行 |
| **内存共享** | 共享内存空间 | 独立内存空间 |
| **创建开销** | 小 | 大 |
| **数据传输** | 快（直接共享） | 慢（序列化） |
| **GIL 影响** | 受影响 | 不受影响 |
| **适用任务类型** | I/O 密集型 | CPU 密集型 |
| **最大推荐 workers** | 通常 20-100 | CPU 核心数 × 2 |

---

## 4. GIL 对多线程的影响

### 什么是 GIL？

GIL（Global Interpreter Lock，全局解释器锁）是 Python 解释器中的一个互斥锁，用于保护对 Python 对象的访问，防止多线程同时执行 Python 字节码。

### GIL 的工作原理

1. **获取 GIL**：线程开始执行 Python 字节码前必须获取 GIL
2. **执行字节码**：持有 GIL 的线程执行
3. **释放 GIL**：
   - 遇到 I/O 操作时释放
   - 执行一定数量字节码后释放（默认 100 条）
   - 线程结束时释放

### GIL 对多线程的具体影响

#### 场景 1：CPU 密集型任务

```
单线程：
[线程1] 计算计算计算计算计算计算计算计算计算计算 → 100% CPU（单核）

多线程（2个线程）：
[线程1] 计算计算 GIL释放 → 等待 → 计算计算 GIL释放 → ...
[线程2] 等待 → 计算计算 GIL释放 → 等待 → 计算计算 ...
→ 依然 ~100% CPU（单核），但有上下文切换开销
```

**结论**：CPU 密集型任务，多线程反而可能比单线程慢！

#### 场景 2：I/O 密集型任务

```
单线程：
[线程1] 请求 → 等待等待等待等待 → 处理 → 请求 → 等待等待等待等待 → 处理

多线程（2个线程）：
[线程1] 请求 → GIL释放 → 等待等待 → GIL获取 → 处理 → 请求 → ...
[线程2] 等待 → 请求 → GIL释放 → 等待等待 → GIL获取 → 处理 → ...
→ I/O 等待时间重叠，整体效率提升
```

**结论**：I/O 密集型任务，多线程可以显著提升效率！

### GIL 释放时机

| 操作类型 | 是否释放 GIL | 说明 |
|---------|-------------|------|
| Python 字节码执行 | ❌ 否 | 执行期间持有 GIL |
| I/O 操作（网络、文件） | ✅ 是 | 等待期间释放 |
| time.sleep() | ✅ 是 | 睡眠期间释放 |
| C 扩展计算 | ✅ 是 | 可手动释放 |
| numpy 运算 | ✅ 是 | numpy 内部会释放 GIL |

### 如何绕过 GIL？

1. **使用多进程**：每个进程有独立的 GIL
2. **使用 C 扩展**：在 C 代码中手动释放 GIL
3. **使用 numba**：JIT 编译，可以绕过 GIL
4. **使用 multiprocessing**：进程级并行

---

## 5. 进程间通信（队列、管道）

由于进程之间内存空间独立，需要特殊的机制进行数据交换。

### 5.1 队列（Queue）

**特点：**
- 线程安全、进程安全
- FIFO（先进先出）
- 支持多种数据类型（需要可序列化）

**基本用法：**
```python
from multiprocessing import Process, Queue

def producer(q):
    for i in range(5):
        q.put(f"数据 {i}")
        print(f"生产者放入: 数据 {i}")

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"消费者取出: {item}")

if __name__ == "__main__":
    q = Queue()
    p1 = Process(target=producer, args=(q,))
    p2 = Process(target=consumer, args=(q,))
    
    p1.start()
    p2.start()
    
    p1.join()
    q.put(None)  # 发送结束信号
    p2.join()
```

### 5.2 管道（Pipe）

**特点：**
- 双向通信
- 适合两个进程之间通信
- 性能比 Queue 好

**基本用法：**
```python
from multiprocessing import Process, Pipe

def sender(conn):
    messages = ["Hello", "World", "END"]
    for msg in messages:
        conn.send(msg)
        print(f"发送: {msg}")
    conn.close()

def receiver(conn):
    while True:
        msg = conn.recv()
        if msg == "END":
            break
        print(f"接收: {msg}")
    conn.close()

if __name__ == "__main__":
    parent_conn, child_conn = Pipe()
    p1 = Process(target=sender, args=(parent_conn,))
    p2 = Process(target=receiver, args=(child_conn,))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
```

### 5.3 共享内存（Manager）

**特点：**
- 可以在多个进程间共享数据
- 支持 list、dict、Value、Array 等

**基本用法：**
```python
from multiprocessing import Process, Manager

def worker(d, l, v):
    d['key'] = 'value'
    l.append(1)
    v.value += 1

if __name__ == "__main__":
    with Manager() as manager:
        d = manager.dict()
        l = manager.list()
        v = manager.Value('i', 0)
        
        processes = [Process(target=worker, args=(d, l, v)) for _ in range(4)]
        
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        
        print(f"Dict: {dict(d)}")
        print(f"List: {list(l)}")
        print(f"Value: {v.value}")
```

### 5.4 IPC 方式对比

| 方式 | 适用场景 | 性能 | 数据大小 | 使用复杂度 |
|------|---------|------|---------|-----------|
| Queue | 多生产者多消费者 | 中 | 中等 | 低 |
| Pipe | 两个进程通信 | 高 | 中等 | 低 |
| Manager | 共享复杂数据结构 | 低 | 较大 | 中 |
| Array/Value | 共享简单数据 | 高 | 小 | 中 |

---

## 6. 线程安全和锁

### 6.1 什么是线程安全？

线程安全是指多个线程同时访问共享资源时，不会出现数据不一致或程序异常的情况。

### 6.2 竞态条件示例

```python
import threading

counter = 0

def increment():
    global counter
    for _ in range(1000000):
        counter += 1  # 这不是原子操作！

threads = [threading.Thread(target=increment) for _ in range(4)]

for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"期望结果: 4000000")
print(f"实际结果: {counter}")  # 结果通常小于 4000000！
```

**问题原因**：`counter += 1` 实际上包含三步操作：
1. 读取 counter 的值
2. 计算加 1
3. 写回 counter

这三步之间可能被其他线程打断。

### 6.3 使用锁（Lock）

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(1000000):
        with lock:  # 获取锁，自动释放
            counter += 1

threads = [threading.Thread(target=increment) for _ in range(4)]

for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"结果: {counter}")  # 结果正确：4000000
```

### 6.4 RLock（可重入锁）

允许同一个线程多次获取同一个锁。

```python
import threading

rlock = threading.RLock()

def nested_function():
    with rlock:
        print("嵌套函数")

def outer_function():
    with rlock:
        print("外层函数")
        nested_function()  # 同一个线程可以再次获取锁

thread = threading.Thread(target=outer_function)
thread.start()
thread.join()
```

### 6.5 其他同步机制

#### Semaphore（信号量）
控制同时访问资源的线程数量：

```python
import threading
import time

semaphore = threading.Semaphore(2)  # 最多2个线程同时执行

def worker(id):
    with semaphore:
        print(f"线程 {id} 开始执行")
        time.sleep(1)
        print(f"线程 {id} 执行结束")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
```

#### Condition（条件变量）
用于线程间的协作：

```python
import threading

condition = threading.Condition()
items = []

def producer():
    with condition:
        print("生产者生产数据")
        items.append("data")
        condition.notify()  # 通知消费者

def consumer():
    with condition:
        condition.wait_for(lambda: len(items) > 0)  # 等待数据
        print(f"消费者消费数据: {items.pop()}")
```

### 6.6 死锁避免原则

1. **按固定顺序获取锁
2. **避免嵌套锁**
3. **使用超时机制**
4. **使用 context manager（with 语句）**

---

## 7. 并发数选择策略

### 7.1 线程池大小选择

#### I/O 密集型任务

**公式：**
```
线程数 = CPU 核心数 × (1 + 等待时间 / 计算时间)
```

**经验值：
- 网络请求：20-100 个线程
- 文件 I/O：10-50 个线程
- 数据库查询：5-20 个线程（受连接池限制）

#### CPU 密集型任务

**公式：**
```
线程数 = CPU 核心数 或 CPU 核心数 + 1
```

**原因**：超过 CPU 核心数后，上下文切换开销会超过并行收益。

### 7.2 进程池大小选择

**通用规则：**
```
进程数 = CPU 核心数 或 CPU 核心数 × 2
```

**考虑因素：**
- 内存占用：每个进程有独立内存空间
- 任务粒度：任务越大，进程数可以少一些
- 通信开销：进程间通信频繁的话，减少进程数

### 7.3 选择决策树

```
开始
  ↓
任务类型是什么？
  ├─→ I/O 密集型
  │    ├─→ 等待时间长？
  │    │    ├─→ 是：线程数 = CPU × 5 ~ CPU × 10
  │    │    └─→ 否：线程数 = CPU × 2 ~ CPU × 5
  │    └─→ 是否有连接限制？
  │         ├─→ 是：线程数 = 连接数限制
  │         └─→ 否：同上
  │
  └─→ CPU 密集型
       ├─→ 任务是否独立？
       │    ├─→ 是：进程数 = CPU 核心数
       │    └─→ 否：进程数 = CPU 核心数 × 2
       └─→ 内存是否足够？
            ├─→ 是：同上
            └─→ 否：减少进程数
```

### 7.4 动态调整策略

```python
import os
import time
from concurrent.futures import ThreadPoolExecutor

def get_optimal_thread_count(task_type='io'):
    cpu_count = os.cpu_count()
    
    if task_type == 'io':
        # I/O 密集型：CPU × 5
        return min(cpu_count * 5, 100)
    elif task_type == 'cpu':
        # CPU 密集型：CPU 核心数
        return cpu_count
    elif task_type == 'mixed':
        # 混合型：CPU × 2
        return cpu_count * 2
    else:
        return cpu_count

# 使用示例
optimal_threads = get_optimal_thread_count('io')
print(f"推荐线程数: {optimal_threads}")
```

---

## 8. 性能对比测试

### 8.1 CPU 密集型任务对比

```bash
python 01_cpu_benchmark.py
```

**预期结果：**
- 单线程：最慢
- 多线程：比单线程稍慢（GIL 开销）
- 多进程：最快（真正并行）

### 8.2 I/O 密集型任务对比

```bash
python 02_io_benchmark.py
```

**预期结果：**
- 单线程：最慢
- 多线程：显著快于单线程
- 多进程：比多线程稍慢（进程创建开销）

### 8.3 并发数影响测试

```bash
python 03_concurrency_strategy.py
```

---

## 总结

| 维度 | 多线程 | 多进程 |
|------|--------|--------|
| **最佳场景** | I/O 密集型 | CPU 密集型 |
| **GIL 影响** | 受影响 | 不受影响 |
| **内存开销** | 小 | 大 |
| **创建开销** | 小 | 大 |
| **数据共享** | 容易 | 困难 |
| **编程复杂度** | 中等 | 较高 |
| **推荐并发数** | CPU × 5 ~ CPU × 10 | CPU 核心数 |

### 选择建议

1. **I/O 密集型任务** → 使用 **ThreadPoolExecutor**
2. **CPU 密集型任务** → 使用 **ProcessPoolExecutor**
3. **需要共享数据** → 使用 **线程 + 锁
4. **任务独立性高** → 使用 **进程**
5. **不确定类型** → 先测试，再选择
