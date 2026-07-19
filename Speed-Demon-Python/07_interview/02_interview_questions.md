# Python 性能优化面试题集

## 一、GIL和并发基础（15题）

### 1. 什么是Python的GIL？它对Python程序性能有什么影响？

**考察点**：GIL基本概念及其对并发性能的影响

**标准答案**：
GIL（全局解释器锁，Global Interpreter Lock）是CPython解释器中的一个互斥锁，用于保护对Python对象的访问，防止多个线程同时执行Python字节码。

**影响**：
- CPU密集型任务：多线程无法真正并行执行，因为GIL的存在使得同一时刻只有一个线程能执行Python代码
- I/O密集型任务：GIL在I/O等待时会被释放，多线程仍然可以带来性能提升
- 多进程可以绕过GIL限制，每个进程有独立的GIL

---

### 2. 为什么Python需要GIL？有没有办法绕过GIL？

**考察点**：GIL的设计原因和解决方案

**标准答案**：
**需要GIL的原因**：
- CPython的内存管理不是线程安全的，引用计数需要保护
- 简化C扩展开发，C扩展不需要处理线程安全问题
- 历史原因：早期设计决策，现在难以移除

**绕过GIL的方法**：
- 使用多进程（multiprocessing）而非多线程
- 使用C扩展在C层面释放GIL
- 使用无GIL的Python实现（如PyPy、Jython、IronPython）
- 使用Numba、Cython等将代码编译为机器码
- 使用concurrent.futures.ProcessPoolExecutor

---

### 3. Python多线程在什么场景下能提升性能？请举例说明

**考察点**：GIL对不同类型任务的影响

**标准答案**：
**能提升性能的场景**：
- I/O密集型任务：网络请求、文件读写、数据库操作等
- 等待外部资源的任务：调用外部API、等待用户输入
- 调用C扩展且C扩展释放了GIL的任务：如NumPy、Pandas操作

**示例**：
```python
import threading
import requests

def fetch_url(url):
    return requests.get(url)

# 多线程下载多个网页（I/O密集型，GIL在等待时释放）
urls = ['http://example.com'] * 10
threads = [threading.Thread(target=fetch_url, args=(url,)) for url in urls]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

### 4. multiprocessing和threading的区别是什么？各自的适用场景？

**考察点**：多进程与多线程的对比

**标准答案**：
**核心区别**：
| 特性 | multiprocessing | threading |
|------|-----------------|-----------|
| GIL影响 | 每个进程独立GIL，不受影响 | 受GIL限制，同一时刻单线程执行 |
| 内存空间 | 独立内存空间，数据共享复杂 | 共享内存空间，数据共享简单 |
| 创建开销 | 进程创建开销大 | 线程创建开销小 |
| 数据传递 | 需要IPC（管道、队列、共享内存） | 直接共享变量 |
| 稳定性 | 进程崩溃不影响其他进程 | 线程崩溃可能导致整个进程崩溃 |

**适用场景**：
- multiprocessing：CPU密集型任务、需要真正并行计算
- threading：I/O密集型任务、需要频繁数据共享的任务

---

### 5. 解释GIL的释放机制，什么时候会释放GIL？

**考察点**：GIL的工作原理和调度策略

**标准答案**：
**GIL的释放时机**：
1. **I/O操作时**：文件读写、网络请求、sleep等阻塞操作会释放GIL
2. **定期检查**：Python 3.2+每5毫秒（100个tick）检查一次，可能释放GIL
3. **C扩展主动释放**：C扩展可以手动调用`Py_BEGIN_ALLOW_THREADS`释放GIL
4. **线程结束时**：线程执行完成自然释放GIL

**释放机制**：
- GIL使用协作式抢占
- 当线程释放GIL后，其他线程竞争获取GIL
- 没有公平性保证，可能出现线程饥饿

---

### 6. concurrent.futures模块中的ThreadPoolExecutor和ProcessPoolExecutor有何不同？

**考察点**：并发futures模块的理解

**标准答案**：
**ThreadPoolExecutor**：
- 基于线程池实现
- 受GIL限制，适合I/O密集型任务
- 线程间共享内存，数据传递开销小
- 创建和管理开销较小

**ProcessPoolExecutor**：
- 基于进程池实现
- 每个进程有独立GIL，适合CPU密集型任务
- 进程间数据传递需要序列化（pickle）
- 创建和管理开销较大

**使用建议**：
```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# I/O密集型使用线程池
with ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(fetch_url, urls)

# CPU密集型使用进程池
with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(cpu_intensive_task, data)
```

---

### 7. 什么是协程？Python的asyncio如何工作？为什么它比多线程更高效？

**考察点**：异步编程和协程的理解

**标准答案**：
**协程（Coroutine）**：
- 是一种用户态的轻量级线程
- 可以在执行过程中暂停和恢复
- 由程序员控制切换时机，而非操作系统

**asyncio工作原理**：
- 基于事件循环（Event Loop）
- 单线程执行，通过`await`挂起耗时操作
- 任务之间协作式调度，没有线程切换开销

**比多线程高效的原因**：
1. 没有线程上下文切换的开销
2. 不需要GIL竞争
3. 内存占用更小（一个线程约1MB栈空间，协程只有KB级）
4. 可以处理上万并发连接，线程做不到

```python
import asyncio

async def hello():
    await asyncio.sleep(1)
    print("Hello")

async def main():
    await asyncio.gather(hello(), hello(), hello())

asyncio.run(main())
```

---

### 8. 解释Python中的GIL对多线程CPU密集型任务的具体影响，并给出性能提升方案

**考察点**：GIL对实际性能的影响和优化策略

**标准答案**：
**具体影响**：
- 对于CPU密集型任务，多线程不仅不能加速，反而可能变慢
- 因为线程频繁竞争GIL，产生额外的上下文切换开销
- 性能甚至可能比单线程还差

**性能提升方案**：
1. **使用多进程**：multiprocessing模块，每个进程独立GIL
2. **使用C扩展**：如NumPy、Pandas在C层面释放GIL
3. **使用Numba**：JIT编译为机器码，释放GIL
4. **使用PyPy**：无GIL的Python实现
5. **任务拆分**：将CPU密集型和I/O密集型分离

```python
from multiprocessing import Pool

def cpu_heavy_task(n):
    return sum(i * i for i in range(n))

if __name__ == '__main__':
    with Pool() as pool:
        results = pool.map(cpu_heavy_task, [10**7] * 4)
```

---

### 9. 什么是守护线程？守护线程和普通线程有什么区别？

**考察点**：线程类型的理解

**标准答案**：
**守护线程（Daemon Thread）**：
- 是一种后台运行的线程
- 当所有非守护线程结束时，守护线程会被强制终止

**区别**：
- 生命周期：主线程结束时，守护线程立即终止；普通线程会继续执行直到完成
- 用途：守护线程用于后台服务（如日志、监控）
- 资源清理：守护线程不会执行finally块，不适合做资源清理

```python
import threading
import time

def daemon_worker():
    while True:
        time.sleep(1)
        print("Daemon running")

t = threading.Thread(target=daemon_worker, daemon=True)
t.start()
# 主线程结束后，守护线程自动终止
```

---

### 10. 解释进程间通信（IPC）的几种方式，在Python中如何实现？

**考察点**：多进程通信机制

**标准答案**：
**常见IPC方式**：

1. **Queue（队列）**：
```python
from multiprocessing import Process, Queue

def worker(q):
    q.put("Hello from worker")

q = Queue()
p = Process(target=worker, args=(q,))
p.start()
print(q.get())  # 接收数据
p.join()
```

2. **Pipe（管道）**：双向通信，性能更好
```python
from multiprocessing import Process, Pipe

parent_conn, child_conn = Pipe()

def worker(conn):
    conn.send("Hello")
    conn.close()
```

3. **Shared Memory（共享内存）**：
```python
from multiprocessing import Array, Value

# 共享数值
count = Value('i', 0)
# 共享数组
arr = Array('d', [1.0, 2.0, 3.0])
```

4. **Manager**：支持更复杂的数据结构
```python
from multiprocessing import Manager

manager = Manager()
shared_dict = manager.dict()
shared_list = manager.list()
```

---

### 11. Python中的锁有哪些类型？它们的使用场景是什么？

**考察点**：并发同步机制

**标准答案**：
**主要锁类型**：

1. **Lock（互斥锁）**：最基础的锁，防止竞态条件
```python
from threading import Lock
lock = Lock()
with lock:
    # 临界区代码
    pass
```

2. **RLock（可重入锁）**：允许同一线程多次获取
```python
from threading import RLock
lock = RLock()
# 同一个线程可以多次acquire
lock.acquire()
lock.acquire()
lock.release()
lock.release()
```

3. **Semaphore（信号量）**：控制并发数量
```python
from threading import Semaphore
# 最多允许5个线程同时访问
sem = Semaphore(5)
```

4. **Condition（条件变量）**：复杂同步场景
```python
from threading import Condition
cond = Condition()
# wait/notify机制
```

5. **Event（事件）**：线程间信号传递

**使用场景**：
- Lock：简单的临界区保护
- RLock：递归函数中需要多次加锁
- Semaphore：连接池等资源限制
- Condition：生产者-消费者模式

---

### 12. 什么是死锁？如何避免死锁？

**考察点**：并发问题识别和预防

**标准答案**：
**死锁定义**：两个或多个线程相互等待对方释放资源，导致所有线程都无法继续执行。

**死锁产生的四个必要条件**：
1. 互斥条件：资源只能被一个线程占用
2. 持有并等待：持有资源同时等待其他资源
3. 不可剥夺：资源不能被强制释放
4. 循环等待：形成等待环路

**避免死锁的方法**：
1. **固定加锁顺序**：所有线程按相同顺序获取锁
2. **超时机制**：设置锁的超时时间
3. **死锁检测**：定期检测并中断死锁线程
4. **资源分级**：按资源等级获取

```python
# 固定加锁顺序避免死锁
def transfer(from_account, to_account, amount):
    # 按id顺序获取锁，避免循环等待
    lock1 = from_account.lock if id(from_account) < id(to_account) else to_account.lock
    lock2 = to_account.lock if id(from_account) < id(to_account) else from_account.lock
    
    with lock1, lock2:
        from_account.balance -= amount
        to_account.balance += amount
```

---

### 13. asyncio中的事件循环是什么？它是如何工作的？

**考察点**：异步编程核心概念

**标准答案**：
**事件循环（Event Loop）**：
- 是asyncio的核心，负责调度和执行异步任务
- 单线程运行，不断轮询就绪的任务

**工作原理**：
1. 注册异步任务到事件循环
2. 事件循环不断检查任务状态
3. 当任务遇到`await`时挂起，控制权交回事件循环
4. 事件循环调度下一个就绪的任务
5. 当挂起任务的等待条件满足时，恢复执行

**关键组件**：
```python
import asyncio

# 获取事件循环
loop = asyncio.get_event_loop()

# 创建任务
task = loop.create_task(coroutine())

# 运行直到完成
loop.run_until_complete(task)

# Python 3.7+简化写法
asyncio.run(main())
```

---

### 14. 解释Python中的线程安全和非线程安全，举例说明哪些操作是线程安全的

**考察点**：多线程安全性理解

**标准答案**：
**线程安全**：多线程环境下执行结果正确，不会出现意外行为。

**Python中的线程安全操作**：
- 单个字节码操作（如`list.append()`）
- `queue.Queue`的所有操作
- `threading`模块中的锁机制保护的操作

**非线程安全操作**：
- 读-改-写操作（如`count += 1`）
- 字典/列表的复合操作
- 多个原子操作的组合

**示例**：
```python
import threading

# 非线程安全
count = 0
def unsafe_increment():
    global count
    for _ in range(100000):
        count += 1  # 不是原子操作！

# 线程安全版本
count = 0
lock = threading.Lock()
def safe_increment():
    global count
    for _ in range(100000):
        with lock:
            count += 1
```

---

### 15. 什么是任务队列？在Python中如何实现分布式任务队列？

**考察点**：分布式并发处理

**标准答案**：
**任务队列**：将任务分发到多个工作进程/线程执行的系统，用于解耦任务提交和执行。

**Python中常见的任务队列实现**：

1. **Celery（最常用）**：
```python
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def add(x, y):
    return x + y

# 提交异步任务
result = add.delay(4, 4)
print(result.get(timeout=1))
```

2. **RQ（Redis Queue）**：轻量级，基于Redis
3. **APScheduler**：定时任务调度
4. **Dramatiq**：现代化任务队列

**分布式架构**：
- Broker：消息中间件（Redis、RabbitMQ）
- Worker：执行任务的进程
- Result Backend：存储任务结果

---

## 二、内存优化（10题）

### 16. Python中的内存管理机制是怎样的？

**考察点**：Python内存管理基础

**标准答案**：
**Python内存管理三层机制**：

1. **引用计数（Reference Counting）**：
- 每个对象维护引用计数
- 计数为0时立即释放内存
- 优点：实时释放，简单高效
- 缺点：无法处理循环引用

2. **垃圾回收（Garbage Collection）**：
- 处理循环引用
- 分代回收：年轻代（0代）、中年代（1代）、老年代（2代）
- 阈值触发：达到一定对象数量自动回收

3. **内存池（Memory Pool）**：
- 小对象（<256KB）使用内存池分配
- 避免频繁向OS申请释放内存
- Pymalloc机制优化小内存分配

```python
import sys

# 查看引用计数
sys.getrefcount(obj)

# 手动触发垃圾回收
import gc
gc.collect()
```

---

### 17. 什么是内存泄漏？在Python中常见的内存泄漏场景有哪些？

**考察点**：内存泄漏识别和诊断

**标准答案**：
**内存泄漏**：不再使用的内存无法被释放，导致内存占用持续增长。

**常见场景**：
1. **全局变量引用**：全局列表/字典持续追加不清理
2. **缓存无限增长**：LRU缓存未设置maxsize
3. **闭包中的外部变量引用**
4. **未关闭的文件/网络连接**
5. **循环引用且包含__del__方法**
6. **C扩展中的内存泄漏**

**示例**：
```python
# 内存泄漏示例
def create_leak():
    data = []
    def inner():
        return data  # 闭包持有引用
    return inner

# 缓存泄漏
from functools import lru_cache

@lru_cache(maxsize=None)  # 无限增长！
def expensive_func(x):
    return x * 2
```

---

### 18. 如何检测和分析Python程序的内存泄漏？介绍常用的工具

**考察点**：内存分析工具使用

**标准答案**：
**常用内存分析工具**：

1. **tracemalloc**（Python 3.4+内置）：
```python
import tracemalloc

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()
# ... 执行业务代码 ...
snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
for stat in top_stats[:10]:
    print(stat)
```

2. **memory_profiler**：逐行内存分析
```bash
pip install memory-profiler
python -m memory_profiler script.py
```

3. **objgraph**：对象引用图分析
```python
import objgraph
# 显示增长最多的对象类型
objgraph.show_growth()
# 显示引用链
objgraph.show_backrefs([obj], filename='backrefs.png')
```

4. **pympler**：内存使用量化

5. **gc模块**：检查无法回收的对象
```python
import gc
gc.set_debug(gc.DEBUG_LEAK)
gc.collect()
print(gc.garbage)
```

---

### 19. __slots__是什么？它如何节省内存？使用场景和限制是什么？

**考察点**：内存优化高级技巧

**标准答案**：
**__slots__**：用于限制类实例的属性，固定实例内存布局。

**内存节省原理**：
- 默认每个实例有`__dict__`字典（约40-80字节开销）
- 使用`__slots__`后，属性存储在固定数组中
- 大量实例时内存节省显著（可达50-80%）

**使用示例**：
```python
class Point:
    __slots__ = ['x', 'y']  # 只允许这两个属性
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 对比内存占用
import sys
class NormalPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p1 = NormalPoint(1, 2)  # 约 56 + 40(__dict__) = 96+ 字节
p2 = Point(1, 2)         # 约 40 字节
```

**限制**：
- 子类必须重新声明`__slots__`
- 不能动态添加属性
- 失去`__dict__`带来的灵活性

---

### 20. 生成器（generator）为什么比列表更节省内存？适用场景是什么？

**考察点**：惰性求值和内存优化

**标准答案**：
**生成器的内存优势**：
- 惰性求值：一次只生成一个元素，不预先生成全部
- 不需要存储所有元素在内存中
- 处理大数据时可以避免内存溢出

**对比示例**：
```python
# 列表：一次性加载所有数据到内存
def get_big_list():
    return [i for i in range(10**7)]  # 占用约 80MB+ 内存

# 生成器：按需生成，内存占用稳定
def get_big_generator():
    for i in range(10**7):
        yield i

# 或者使用生成器表达式
gen = (i for i in range(10**7))  # 几乎不占内存
```

**适用场景**：
- 处理大数据集/文件
- 数据流式处理
- 不需要随机访问，只需顺序迭代
- 无限序列（如斐波那契数列）

---

### 21. 什么是弱引用（weakref）？它解决什么问题？

**考察点**：高级内存管理技巧

**标准答案**：
**弱引用**：不会增加对象的引用计数，不阻止对象被垃圾回收。

**解决的问题**：
- 避免循环引用导致的内存泄漏
- 缓存场景：缓存的对象可以在内存不足时被回收
- 观察者模式：观察者不持有被观察者的强引用

**使用示例**：
```python
import weakref

class Observer:
    def __init__(self, subject):
        self.subject_ref = weakref.ref(subject)  # 弱引用
    
    def update(self):
        subject = self.subject_ref()
        if subject is not None:  # 检查对象是否还存在
            # 处理
            pass

# WeakValueDictionary：值为弱引用的字典
cache = weakref.WeakValueDictionary()

def get_data(key):
    if key in cache:
        return cache[key]
    data = load_from_db(key)
    cache[key] = data  # 内存不足时自动清理
    return data
```

---

### 22. 如何优化大型列表的内存占用？列举至少3种方法

**考察点**：集合类型内存优化

**标准答案**：
**优化方法**：

1. **使用数组（array）模块**：
```python
import array
# 存储类型化数据，节省内存
arr = array.array('i', [1, 2, 3, 4, 5])  # 'i' = int，每个4字节
```

2. **使用numpy数组**：
```python
import numpy as np
# 高效数值数组
arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
```

3. **使用生成器代替列表**：
```python
# 不创建完整列表，按需生成
for item in (x * 2 for x in big_data):
    process(item)
```

4. **分块处理**：
```python
def process_in_chunks(file_path, chunk_size=10000):
    with open(file_path) as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                process_chunk(chunk)
                chunk = []  # 释放内存
```

5. **使用pandas处理数据**：适合结构化数据，有优化的存储

---

### 23. 解释Python中的字符串驻留（string interning）机制，它对性能有什么影响？

**考察点**：Python内部优化机制

**标准答案**：
**字符串驻留**：将相同的字符串对象只保存一份，避免重复创建。

**驻留规则**：
- 编译时的字符串字面量自动驻留
- 长度为0或1的字符串自动驻留
- 只包含字母、数字、下划线的字符串会被驻留
- 可以手动调用`sys.intern()`驻留字符串

**性能影响**：
- 优点：节省内存，字符串比较更快（O(1)指针比较）
- 缺点：驻留本身有开销，不适合大量唯一字符串

**示例**：
```python
import sys

# 自动驻留
a = "hello"
b = "hello"
print(a is b)  # True，同一对象

# 不自动驻留
c = "hello!"
d = "hello!"
print(c is d)  # False

# 手动驻留
e = sys.intern("hello!")
f = sys.intern("hello!")
print(e is f)  # True
```

---

### 24. 对于字典和集合，Python是如何优化内存使用的？

**考察点**：内置数据结构的优化

**标准答案**：
**字典的内存优化**：

1. **哈希表实现**：使用开放寻址法（Python 3.7+）
2. **负载因子**：达到2/3容量时自动扩容，避免哈希冲突过多
3. **共享键字典**（Python 3.3+）：类实例的字典可以共享键
4. **紧凑字典**（Python 3.6+）：更紧凑的内存布局

**集合的优化**：
- 基于哈希表实现，类似于只有键的字典
- 元素必须是可哈希的

**内存优化建议**：
```python
# 1. 使用字典推导式而非逐步添加
data = {k: v for k, v in items}  # 一次性分配

# 2. 预估计容量
from collections import defaultdict
# 如果知道大致大小，可以避免多次扩容

# 3. 使用namedtuple或dataclass代替小字典
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)  # 比 {'x':1, 'y':2} 更省内存

# 4. Python 3.7+字典按插入顺序存储，更高效
```

---

### 25. 什么是内存池？Python如何通过内存池优化小对象分配？

**考察点**：Python底层内存管理

**标准答案**：
**内存池**：预先分配一块内存，用于频繁分配和释放的小对象，避免系统调用开销。

**Python的Pymalloc机制**：
- 管理小于等于512字节的内存分配
- 按大小分级（8, 16, ..., 512字节）
- 每级维护一个空闲链表
- 分配时直接从对应大小的链表取

**工作原理**：
```
申请内存：
    if 大小 > 512字节:
        使用系统malloc
    else:
        找到对应大小的Arena
        从Pool中分配Block
        分配失败则创建新的Arena
```

**优化效果**：
- 减少系统调用次数
- 减少内存碎片
- 小内存分配速度提升显著

**手动优化**：
```python
# 避免频繁创建小对象
# 不好的写法
result = []
for x in data:
    result.append(str(x))  # 每次创建新字符串

# 更好的写法：使用join减少中间对象
result = [str(x) for x in data]
final = ''.join(result)
```

---

## 三、性能分析工具（10题）

### 26. cProfile和profile的区别是什么？如何使用cProfile分析性能？

**考察点**：Python性能分析基础工具

**标准答案**：
**区别**：
- cProfile：C实现，开销小，适合生产环境
- profile：Python实现，可扩展，但开销大（约10倍）
- 推荐使用cProfile

**使用方法**：

1. **命令行使用**：
```bash
python -m cProfile -o output.prof script.py
python -m cProfile -s cumulative script.py  # 按累积时间排序
```

2. **代码中使用**：
```python
import cProfile

def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

# 直接运行
cProfile.run('slow_function()')

# 保存到文件
cProfile.run('slow_function()', 'slow_function.prof')
```

3. **分析结果**：
```python
import pstats

stats = pstats.Stats('output.prof')
stats.sort_stats('cumulative').print_stats(20)  # 显示前20个耗时函数
stats.sort_stats('time').print_stats(10)        # 按自身时间排序
```

**关注指标**：
- ncalls：调用次数
- tottime：函数自身时间（不含子函数）
- cumtime：累积时间（含子函数）

---

### 27. line_profiler是什么？如何使用它进行逐行性能分析？

**考察点**：行级性能分析工具

**标准答案**：
**line_profiler**：可以显示函数中每行代码的执行时间和占用比例。

**使用步骤**：

1. **安装**：
```bash
pip install line-profiler
```

2. **使用装饰器标记需要分析的函数**：
```python
@profile  # 添加此装饰器
def process_data(data):
    result = []
    for item in data:
        processed = heavy_computation(item)
        result.append(processed)
    return result
```

3. **运行分析**：
```bash
kernprof -l -v script.py
# -l: 使用line_profiler
# -v: 立即显示结果
```

**输出示例**：
```
Total time: 2.5s
File: script.py
Function: process_data at line 1

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     1                                           @profile
     2                                           def process_data(data):
     3         1            2      2.0      0.0      result = []
     4      1000          520      0.5      0.0      for item in data:
     5      1000      2499480   2499.5     99.9          processed = heavy_computation(item)
     6      1000           80      0.1      0.0          result.append(processed)
     7         1            2      2.0      0.0      return result
```

---

### 28. 如何使用timeit进行基准测试？有哪些注意事项？

**考察点**：基准测试工具使用

**标准答案**：
**timeit**：用于测量小段Python代码的执行时间，自动处理多次运行统计。

**使用方法**：

1. **命令行使用**：
```bash
python -m timeit 'sum(range(1000))'
python -m timeit -s 'x = list(range(1000))' 'sum(x)'  # -s: 设置执行环境
python -m timeit -n 10000 -r 5 'sum(range(100))'     # -n: 循环次数, -r: 重复次数
```

2. **代码中使用**：
```python
import timeit

# 测试单个语句
time = timeit.timeit('sum(range(1000))', number=10000)
print(f"Average: {time/10000:.6f}s")

# 测试函数
def test_func():
    return sum(range(1000))

time = timeit.timeit(test_func, number=10000)

# repeat测试，取最小值
times = timeit.repeat(test_func, number=10000, repeat=5)
print(f"Best time: {min(times)/10000:.6f}s")
```

**注意事项**：
- 取多次运行的最小值，而非平均值
- 排除系统噪声影响
- 确保测试环境稳定
- 足够的运行次数让结果收敛
- 避免测试受GC影响，可禁用GC

---

### 29. 介绍py-spy这个工具，它有什么优势？如何使用？

**考察点**：采样性能分析工具

**标准答案**：
**py-spy**：非侵入式的Python性能采样分析器，不需要修改代码。

**优势**：
- 非侵入式：不需要添加装饰器或修改代码
- 低开销：对目标程序性能影响小
- 可以分析正在运行的程序
- 支持火焰图生成

**使用方法**：

1. **安装**：
```bash
pip install py-spy
```

2. **分析运行中的进程**：
```bash
py-spy top --pid 12345  # 实时查看函数耗时排行
```

3. **启动并分析程序**：
```bash
py-spy top -- python script.py
```

4. **生成火焰图**：
```bash
py-spy record -o profile.svg -- python script.py
py-spy record -o profile.svg --pid 12345 -d 10  # 采样10秒
```

5. **dump调用栈**：
```bash
py-spy dump --pid 12345
```

**适用场景**：
- 生产环境性能问题排查
- 分析长时间运行的程序
- 无法修改代码的场景
- 快速定位性能瓶颈

---

### 30. memory_profiler的使用方法是什么？如何解读它的输出？

**考察点**：内存分析工具

**标准答案**：
**memory_profiler**：逐行分析Python程序的内存使用情况。

**使用方法**：

1. **安装**：
```bash
pip install memory-profiler
```

2. **使用装饰器**：
```python
from memory_profiler import profile

@profile
def my_func():
    a = [1] * (10**6)
    b = [2] * (2 * 10**7)
    del b
    return a

if __name__ == '__main__':
    my_func()
```

3. **运行**：
```bash
python -m memory_profiler script.py
```

**输出解读**：
```
Line #    Mem usage    Increment  Occurrences   Line Contents
==============================================================
     4   45.684 MiB   45.684 MiB           1   @profile
     5                                         def my_func():
     6   53.316 MiB    7.633 MiB           1       a = [1] * (10**6)
     7  206.270 MiB  152.953 MiB           1       b = [2] * (2 * 10**7)
     8   53.316 MiB -152.953 MiB           1       del b
     9   53.316 MiB    0.000 MiB           1       return a
```

**关键列解释**：
- Mem usage：当前内存使用量
- Increment：相对上一行的内存增量
- Occurrences：执行次数

---

### 31. 什么是火焰图（Flame Graph）？如何生成和解读Python程序的火焰图？

**考察点**：可视化性能分析

**标准答案**：
**火焰图**：可视化函数调用栈和执行时间的图表，便于快速定位性能瓶颈。

**生成方法**：

1. **使用py-spy**：
```bash
py-spy record -o flame.svg -- python script.py
```

2. **使用cProfile + flameprof**：
```bash
pip install flameprof
python -m cProfile -o profile.prof script.py
flameprof profile.prof > flame.svg
```

**解读方法**：
- **Y轴**：调用栈深度，每一层是一个函数
- **X轴**：采样数（代表执行时间），越宽表示耗时越长
- **颜色**：随机区分不同函数
- **顶层宽的函数**：最可能是瓶颈

**常见模式**：
- 单宽峰：某个函数占用大量时间
- 平项：多个函数耗时相当
- 窄底宽顶：深层函数是瓶颈

**优化方向**：
- 优化最宽的函数
- 减少调用次数
- 缓存重复计算结果

---

### 32. 如何使用dis模块分析Python字节码？这对性能优化有什么帮助？

**考察点**：Python底层执行分析

**标准答案**：
**dis模块**：反汇编Python代码为字节码，了解底层执行过程。

**使用方法**：
```python
import dis

def sum_numbers(n):
    total = 0
    for i in range(n):
        total += i
    return total

dis.dis(sum_numbers)
```

**输出示例**：
```
  2           0 LOAD_CONST               1 (0)
              2 STORE_FAST               1 (total)

  3           4 LOAD_GLOBAL              0 (range)
              6 LOAD_FAST                0 (n)
              8 CALL_FUNCTION            1
             10 GET_ITER
        >>   12 FOR_ITER                 6 (to 22)
             14 STORE_FAST               2 (i)

  4          16 LOAD_FAST                1 (total)
             18 LOAD_FAST                2 (i)
             20 INPLACE_ADD
             22 STORE_FAST               1 (total)
             24 JUMP_ABSOLUTE            6 (to 12)

  5     >>   26 LOAD_FAST                1 (total)
             28 RETURN_VALUE
```

**性能优化帮助**：
- 理解循环展开的效果
- 比较局部变量vs全局变量访问（LOAD_FAST比LOAD_GLOBAL快）
- 了解属性访问的开销
- 发现不必要的字节码操作
- 优化条件判断顺序

---

### 33. 介绍Py-Spy、cProfile、line_profiler的适用场景和差异

**考察点**：性能工具选型能力

**标准答案**：
**工具对比表**：

| 特性 | py-spy | cProfile | line_profiler |
|------|--------|----------|---------------|
| 侵入性 | 非侵入式，无需修改代码 | 轻微侵入 | 需要添加装饰器 |
| 性能开销 | 低（<5%） | 中等 | 较高 |
| 分析粒度 | 函数级采样 | 函数级统计 | 行级精确统计 |
| 可分析运行中进程 | 是 | 否 | 否 |
| 火焰图支持 | 原生支持 | 需要第三方工具 | 不支持 |
| 适合分析 | CPU瓶颈，长时间运行程序 | 整体性能瓶颈 | 热点函数详细分析 |
| 生产环境 | 适合 | 谨慎使用 | 不建议 |

**选型建议**：
- **快速排查生产问题**：py-spy
- **首次整体性能分析**：cProfile
- **已知热点函数优化**：line_profiler
- **可视化调用栈**：py-spy + 火焰图
- **内存问题**：memory_profiler

---

### 34. 如何分析和优化Python程序的启动时间？

**考察点**：启动性能优化

**标准答案**：
**分析工具**：

1. **strace（Linux）**：跟踪系统调用
```bash
strace -c python script.py  # 统计系统调用耗时
```

2. **python -X importtime**：分析导入时间（Python 3.7+）
```bash
python -X importtime script.py 2> import.log
# 使用tuna可视化
pip install tuna
tuna import.log
```

3. **cProfile**：
```bash
python -m cProfile -s cumulative script.py
```

**常见优化方法**：

1. **延迟导入**：
```python
# 不好：顶层导入
import pandas as pd
import numpy as np

# 好：函数内按需导入
def process_data():
    import pandas as pd  # 延迟到真正需要时
    return pd.read_csv('data.csv')
```

2. **避免模块级计算**：
```python
# 不好：导入时执行计算
DEFAULT_CONFIG = load_config_from_file()

# 好：懒加载
_DEFAULT_CONFIG = None
def get_config():
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = load_config_from_file()
    return _DEFAULT_CONFIG
```

3. **使用编译后的模块（.pyc）**
4. **减少依赖，只导入必要的模块**
5. **使用PyInstaller等工具预编译**

---

### 35. 什么是基准测试（Benchmark）？如何编写可靠的Python基准测试？

**考察点**：性能测试方法论

**标准答案**：
**基准测试**：在标准条件下测量程序性能，用于比较不同实现的性能差异。

**编写可靠基准测试的要点**：

1. **使用timeit而非time.time**：
```python
import timeit

def benchmark():
    # setup只执行一次
    setup = 'from __main__ import my_func; data = list(range(1000))'
    stmt = 'my_func(data)'
    
    # 多次运行取最小值
    times = timeit.repeat(stmt, setup=setup, number=1000, repeat=5)
    best_time = min(times) / 1000
    print(f"Best time: {best_time:.6f}s")
```

2. **控制变量**：
- 相同硬件环境
- 相同Python版本
- 关闭其他占用CPU的程序
- 禁用GC或控制GC时机

3. **统计方法**：
- 取多次运行的最小值（而非平均值）
- 足够的样本量
- 排除热身影响

4. **使用pytest-benchmark**：
```python
# test_benchmark.py
def test_my_func(benchmark):
    data = list(range(1000))
    result = benchmark(my_func, data)
    assert result is not None
```

```bash
pytest test_benchmark.py
```

5. **A/B测试对比**：
```python
# 比较两种实现
def old_implementation(): pass
def new_implementation(): pass

# 分别测试，对比性能差异
```

---

## 四、代码优化技巧（10题）

### 36. 为什么局部变量访问比全局变量快？如何利用这一点优化性能？

**考察点**：变量查找机制和优化

**标准答案**：
**原因**：
- Python变量查找顺序：局部(L) → 闭包(E) → 全局(G) → 内置(B)
- 局部变量存储在固定大小的数组中，通过索引访问
- 全局变量存储在字典中，通过哈希查找
- 字节码差异：`LOAD_FAST`（局部）比`LOAD_GLOBAL`（全局）快约30%

**优化方法**：

1. **在函数内使用局部变量**：
```python
# 不好：循环中访问全局变量
import math
global_list = list(range(1000000))

def bad_calculate():
    result = []
    for x in global_list:  # 全局访问
        result.append(math.sqrt(x))  # 每次都查找math和sqrt
    return result

# 好：将全局变量转为局部变量
def good_calculate():
    sqrt = math.sqrt  # 缓存到局部
    local_list = global_list  # 缓存到局部
    append = result.append  # 缓存方法
    
    result = []
    for x in local_list:
        append(sqrt(x))
    return result
```

2. **在循环前缓存常用函数**

3. **使用`locals()`检查局部变量**

**性能提升**：通常可带来10-30%的性能提升

---

### 37. 列表推导式为什么比for循环+append更快？还有哪些类似的优化？

**考察点**：Python内置优化机制

**标准答案**：
**列表推导式更快的原因**：
- 底层使用C代码实现，避免了Python层的循环开销
- 预分配列表大小，减少扩容次数
- 减少了属性查找和方法调用开销

**性能对比**：
```python
import timeit

# for循环 + append
def loop_append():
    result = []
    for i in range(10000):
        result.append(i * 2)
    return result

# 列表推导式
def list_comp():
    return [i * 2 for i in range(10000)]

# 性能差异：列表推导式快约2倍
# 列表推导式约0.3ms，循环append约0.6ms
```

**类似的优化技巧**：

1. **生成器表达式**：比列表推导式更省内存
2. **字典/集合推导式**：`{k:v for k,v in items}`
3. **使用map/filter**：对于内置函数，有时比列表推导式更快
```python
# map对于内置函数更快
result = list(map(str, range(1000)))
```

4. **使用内置方法**：如`str.join()`比字符串拼接快
5. **使用itertools模块**：C实现的迭代器工具
6. **避免在循环中调用函数**

---

### 38. 什么是循环展开（Loop Unrolling）？在Python中如何应用？

**考察点**：循环优化技术

**标准答案**：
**循环展开**：减少循环迭代次数，每次迭代处理多个元素，降低循环控制开销。

**优化原理**：
- 减少循环条件检查次数
- 增加指令级并行（ILP）机会
- 减少分支预测失败

**Python中应用**：
```python
# 原始循环
def original_loop(data):
    result = []
    for x in data:
        result.append(x * 2)
    return result

# 循环展开：每次处理2个元素
def unrolled_loop(data):
    result = []
    n = len(data)
    i = 0
    # 每次处理2个元素
    while i < n - 1:
        result.append(data[i] * 2)
        result.append(data[i+1] * 2)
        i += 2
    # 处理剩余元素
    while i < n:
        result.append(data[i] * 2)
        i += 1
    return result

# 更彻底的展开：每次处理4个元素
def fully_unrolled(data):
    result = []
    n = len(data)
    for i in range(0, n - 3, 4):
        result.extend([
            data[i] * 2,
            data[i+1] * 2,
            data[i+2] * 2,
            data[i+3] * 2
        ])
    # 处理剩余
    for i in range(i, n):
        result.append(data[i] * 2)
    return result
```

**注意**：Python中循环展开效果不如编译语言，适度展开（2-4次）通常能带来10-20%提升。

---

### 39. 如何优化函数调用开销？列举至少3种方法

**考察点**：函数调用优化

**标准答案**：
**函数调用开销来源**：栈帧创建、参数传递、返回值处理等。

**优化方法**：

1. **减少函数调用层数，内联小函数**：
```python
# 不好：小函数频繁调用
def add(a, b):
    return a + b

def process(data):
    result = 0
    for x in data:
        result = add(result, x)  # 每次都有函数调用开销

# 好：内联简单逻辑
def process(data):
    result = 0
    for x in data:
        result += x  # 无函数调用开销
```

2. **使用局部函数代替模块级函数**：
```python
def outer():
    def inner_helper(x):  # 局部函数，查找更快
        return x * 2
    
    result = []
    for x in data:
        result.append(inner_helper(x))
```

3. **使用lru_cache缓存纯函数结果**：
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

4. **避免递归，改用迭代**：递归有栈开销且可能栈溢出

5. **使用生成器减少中间函数调用**

6. **Numba JIT编译消除函数调用开销**

---

### 40. 字符串拼接有哪些方式？性能对比如何？为什么`str.join()`最快？

**考察点**：字符串性能优化

**标准答案**：
**字符串拼接方式及性能**：

| 方法 | 性能 | 适用场景 |
|------|------|----------|
| str.join() | 最快 | 大量字符串拼接 |
| f-string | 快 | 格式化字符串 |
| += 或 + | 慢 | 少量拼接（Python 2.5+优化了但仍慢） |
| % 格式化 | 慢 | 旧代码兼容 |
| str.format() | 较慢 | 复杂格式化 |

**`str.join()`最快的原因**：
1. **一次内存分配**：先计算总长度，一次性分配内存
2. **C语言实现**：底层C代码执行，无Python循环开销
3. **避免临时对象**：不需要创建大量中间字符串

**性能对比示例**：
```python
# 最慢：+= 会创建大量临时字符串
result = ''
for s in string_list:
    result += s  # 每次都创建新字符串

# 最快：str.join()
result = ''.join(string_list)  # 一次分配，一次复制
```

**优化建议**：
```python
# 1. 总是使用str.join拼接大量字符串
strings = [str(i) for i in range(10000)]
result = ''.join(strings)

# 2. 小数量拼接可以用f-string，可读性好
result = f"Name: {name}, Age: {age}"

# 3. 混合使用
parts = []
parts.append(f"Name: {name}")
parts.append(f"Age: {age}")
result = '\n'.join(parts)
```

---

### 41. 什么是短路求值？如何利用短路求值优化性能？

**考察点**：逻辑运算优化

**标准答案**：
**短路求值**：逻辑运算符`and`和`or`只要能确定结果，就停止后续表达式的求值。

**规则**：
- `a and b`：a为False时返回a，不计算b
- `a or b`：a为True时返回a，不计算b

**优化应用**：

1. **将快速失败的条件放前面**：
```python
# 好：快速检查放前面
if user is not None and has_permission(user) and check_complex_rule(user):
    # 处理
```

2. **避免不必要的计算**：
```python
# 不好：总是计算两个函数
result = expensive_func1() or expensive_func2()

# 好：如果第一个为True，第二个不执行
def get_value():
    result = get_from_cache()
    if result is not None:
        return result
    return compute_expensive()  # 只有缓存未命中才执行
```

3. **使用`and`提供默认值**：
```python
# 使用or提供默认值（注意：0、''、False会被覆盖）
value = maybe_none or default_value

# 更安全的写法
value = maybe_none if maybe_none is not None else default_value
```

4. **条件链优化**：
```python
# 按失败概率排序，高概率失败放前面
if is_valid_format(data) and is_authorized(user) and process_complex_logic(data):
    pass
```

---

### 42. 如何优化Python中的属性访问？`__slots__`、`property`、`@cached_property`的区别？

**考察点**：属性访问优化

**标准答案**：
**属性访问优化方法**：

1. **使用局部变量缓存属性**：
```python
# 不好：循环中重复访问属性
for item in collection:
    result.append(obj.method(item))  # 每次查找obj和method

# 好：缓存到局部变量
method = obj.method
append = result.append
for item in collection:
    append(method(item))
```

2. **`__slots__`**：固定属性布局，减少内存和加速访问
```python
class Point:
    __slots__ = ['x', 'y']  # 属性访问更快，内存更少
```

3. **`property`**：动态计算属性
```python
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @property
    def area(self):
        return math.pi * self.radius ** 2  # 每次访问都计算
```

4. **`@cached_property`**：缓存计算结果（Python 3.8+）
```python
from functools import cached_property

class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @cached_property
    def area(self):
        return math.pi * self.radius ** 2  # 只计算一次
```

**对比**：
| 特性 | 普通属性 | __slots__ | property | cached_property |
|------|----------|-----------|----------|-----------------|
| 内存占用 | 高（有__dict__） | 低 | 中 | 中 |
| 访问速度 | 快 | 很快 | 慢（每次计算） | 第一次慢，之后快 |
| 动态添加 | 可以 | 不可以 | 不可以 | 不可以 |
| 计算开销 | 无 | 无 | 每次都有 | 一次 |

---

### 43. 什么是鸭子类型？它对性能有什么影响？何时应该使用类型检查？

**考察点**：动态类型与性能

**标准答案**：
**鸭子类型**："如果走起来像鸭子，叫起来像鸭子，那它就是鸭子"。Python不检查类型，只关心对象是否有需要的方法/属性。

**性能影响**：
- 优点：灵活，代码简洁，无需类型转换
- 缺点：
  - 属性查找是动态的，比静态类型慢
  - 无法在编译期优化
  - 需要更多运行时检查

**何时使用类型检查**：

1. **入口参数验证**：
```python
def process_data(data):
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Expected list or tuple, got {type(data)}")
    # 处理
```

2. **性能关键路径**：使用类型注解配合JIT
```python
from numba import jit

@jit(nopython=True)
def fast_sum(arr: list[float]) -> float:
    total = 0.0
    for x in arr:
        total += x
    return total
```

3. **避免过度类型检查**：
```python
# 不好：过度检查，增加开销
def bad_process(x):
    if isinstance(x, int):
        return x + 1
    elif isinstance(x, float):
        return x + 1.0
    else:
        raise TypeError()

# 好：鸭子类型，信任调用者
def good_process(x):
    return x + 1  # 只要支持+操作就可以
```

---

### 44. 如何优化异常处理？try-except对性能有什么影响？

**考察点**：异常处理的性能考量

**标准答案**：
**异常处理的性能影响**：
- 正常情况下（无异常抛出）：try块开销很小，几乎可以忽略
- 异常抛出时：栈展开开销大，应避免用异常做流程控制

**优化原则**：

1. **不要用异常做正常流程控制**：
```python
# 不好：用异常控制流程，预期内的异常性能很差
def bad_parse_int(s):
    try:
        return int(s)
    except ValueError:
        return None  # 如果经常失败，开销很大

# 好：先检查，异常只用于真正的错误
def good_parse_int(s):
    if s.strip().lstrip('-').isdigit():
        return int(s)
    return None
```

2. **缩小try块范围**：
```python
# 不好：范围过大
def bad_func():
    try:
        result = compute()
        log_result(result)
        send_email(result)
    except Exception:
        print("Error")

# 好：只包裹可能出错的部分
def good_func():
    try:
        result = compute()  # 只有这个可能出错
    except ComputeError:
        result = default_value
    log_result(result)
    send_email(result)
```

3. **捕获具体异常，不要except Exception**

4. **使用EAFP vs LBYL原则**：
- EAFP（Easier to Ask for Forgiveness than Permission）：先操作，异常处理（适合Python）
- LBYL（Look Before You Leap）：先检查再操作

5. **性能关键代码避免深层嵌套异常**

---

### 45. 什么是JIT编译？Numba如何加速Python代码？适用场景和限制是什么？

**考察点**：即时编译优化技术

**标准答案**：
**JIT（Just-In-Time）编译**：运行时将热点代码编译为机器码执行。

**Numba工作原理**：
1. 函数第一次调用时，根据参数类型编译为机器码
2. 后续调用直接执行机器码，跳过Python解释器
3. 支持CPU和GPU加速

**使用方法**：
```python
from numba import jit, njit

# 基础用法
@jit
def sum_array(arr):
    total = 0
    for x in arr:
        total += x
    return total

# nopython模式：不使用Python C API，性能更好
@njit  # 等价于 @jit(nopython=True)
def fast_mandelbrot(c, max_iter):
    z = 0j
    for i in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return i
    return max_iter

# 指定类型签名，预编译
@jit('float64(float64[:])')
def typed_sum(arr):
    return arr.sum()
```

**适用场景**：
- 数值计算密集型代码
- 大量循环操作
- NumPy无法直接向量化的操作
- 科学计算、模拟算法

**限制**：
- 只支持Python功能的子集
- 无法加速字符串、字典等操作
- 编译有开销，适合多次调用的函数
- 调试困难

---

## 五、业务场景优化（10题）

### 46. 处理大文件时，如何避免内存溢出？列举至少3种优化策略

**考察点**：大数据处理场景

**标准答案**：
**优化策略**：

1. **逐行读取，不加载整个文件**：
```python
# 不好：一次性加载
with open('big_file.txt') as f:
    lines = f.readlines()  # 全部加载到内存！

# 好：逐行处理
with open('big_file.txt') as f:
    for line in f:  # 迭代器，每次只加载一行
        process_line(line)
```

2. **分块处理**：
```python
def process_in_chunks(file_path, chunk_size=10000):
    with open(file_path) as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                process_chunk(chunk)
                chunk = []  # 释放内存
        # 处理剩余
        if chunk:
            process_chunk(chunk)
```

3. **使用生成器**：
```python
def read_large_file(file_path):
    with open(file_path) as f:
        for line in f:
            yield process_line(line)  # 惰性处理

# 流式处理
for result in read_large_file('big_file.txt'):
    do_something(result)
```

4. **使用pandas的分块读取**：
```python
import pandas as pd

chunk_iter = pd.read_csv('big_data.csv', chunksize=10000)
for chunk in chunk_iter:
    process_chunk(chunk)  # 每次只处理一个chunk
```

5. **使用Dask等外存计算框架**：支持大于内存的数据处理

---

### 47. Web应用中如何优化数据库查询性能？列举至少5种方法

**考察点**：Web服务性能优化

**标准答案**：
**优化方法**：

1. **使用索引**：
```sql
-- 为常用查询字段创建索引
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_order_created ON orders(created_at);
```

2. **N+1查询问题：使用预加载**：
```python
# 不好：N+1查询
users = User.objects.all()
for user in users:
    print(user.profile.name)  # 每个用户都查一次profile

# 好：预加载关联数据
users = User.objects.select_related('profile').all()  # 一次JOIN查询
```

3. **只查询需要的字段**：
```python
# 不好：查询所有字段
users = User.objects.all()

# 好：只查需要的字段
users = User.objects.only('id', 'name', 'email')
users = User.objects.values('id', 'name')  # 返回字典，更省内存
```

4. **分页查询**：
```python
# 不好：查询所有结果
posts = Post.objects.all()

# 好：分页
posts = Post.objects.all()[:20]  # LIMIT 20
```

5. **使用连接池**：
```python
# Django settings
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'OPTIONS': {
            'MAX_CONNS': 20,  # 连接池大小
        }
    }
}
```

6. **数据库读写分离**：主写从读

7. **添加缓存层**：Redis/Memcached缓存热点数据

8. **批量操作减少查询次数**：
```python
# 批量创建
User.objects.bulk_create([User(name=f'user{i}') for i in range(100)])
```

---

### 48. 如何设计和实现一个高效的缓存系统？LRU、LFU等算法的区别？

**考察点**：缓存设计与实现

**标准答案**：
**高效缓存设计要点**：

1. **Python标准库LRU缓存**：
```python
from functools import lru_cache

@lru_cache(maxsize=1024)  # 限制大小，防止内存泄漏
def get_user(user_id):
    return db.query(User).get(user_id)

# 手动管理
get_user.cache_info()  # 查看缓存统计
get_user.cache_clear()  # 清空缓存
```

2. **自定义LRU实现（使用OrderedDict）**：
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)  # 移到末尾表示最近使用
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # 移除最久未使用的
```

**缓存淘汰算法对比**：

| 算法 | 全称 | 淘汰策略 | 优点 | 缺点 | 适用场景 |
|------|------|----------|------|------|----------|
| LRU | Least Recently Used | 最久未使用 | 实现简单，性能好 | 缓存污染 | 通用场景 |
| LFU | Least Frequently Used | 最少使用 | 命中率高 | 实现复杂，新数据易被淘汰 | 热点数据稳定 |
| FIFO | First In First Out | 最早进入 | 最简单 | 命中率低 | 简单场景 |
| ARC | Adaptive Replacement Cache | 自适应LRU+LFU | 性能最佳 | 专利问题 | 高级缓存 |

**最佳实践**：
- 设置合理的过期时间（TTL）
- 监控命中率
- 避免缓存击穿、雪崩、穿透
- 使用多级缓存（内存 + 分布式缓存）

---

### 49. 高并发场景下如何优化Python Web服务？列举至少5种策略

**考察点**：高并发Web服务优化

**标准答案**：
**优化策略**：

1. **选择合适的Web服务器**：
- 不要用开发服务器（runserver）
- 使用Gunicorn + Gevent/Uvicorn
```bash
# Gunicorn + Gevent 部署
gunicorn -w 4 -k gevent -b 0.0.0.0:8000 app:app

# 异步ASGI服务器
uvicorn app:app --workers 4 --port 8000
```

2. **异步处理耗时任务**：
```python
# 不好：同步等待
@app.route('/process')
def process():
    result = long_running_task()  # 阻塞整个线程
    return result

# 好：异步任务队列
@app.route('/process')
def process():
    task = long_running_task.delay()  # Celery异步执行
    return {'task_id': task.id}
```

3. **连接池管理**：
```python
import redis
from redis.connection import ConnectionPool

# 全局连接池，不要每次创建连接
pool = ConnectionPool(host='localhost', port=6379, max_connections=100)
r = redis.Redis(connection_pool=pool)
```

4. **静态资源CDN加速**：
- JS/CSS/图片放到CDN
- Nginx反向代理静态文件
- 启用gzip压缩

5. **数据库优化**：
- 读写分离
- 分库分表
- 添加适当索引
- 避免全表扫描

6. **限流和熔断**：
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=get_remote_address)

@app.route("/api")
@limiter.limit("100/minute")  # 每分钟100次请求
def api():
    return "ok"
```

7. **水平扩展**：多实例部署 + 负载均衡

---

### 50. 数据处理中如何使用NumPy/Pandas优化性能？常用的优化技巧有哪些？

**考察点**：数据处理性能优化

**标准答案**：
**NumPy/Pandas优化技巧**：

1. **使用向量化操作代替循环**：
```python
import numpy as np
import pandas as pd

# 不好：Python循环
def bad_normalize(data):
    result = []
    for x in data:
        result.append((x - data.mean()) / data.std())
    return result

# 好：向量化操作（C层面执行，快100倍+）
def good_normalize(data):
    return (data - data.mean()) / data.std()

# Pandas向量化
df['new_col'] = df['a'] * df['b'] + df['c']  # 不是循环！
```

2. **避免链式索引，使用.loc/.iloc**：
```python
# 不好：链式索引可能产生SettingWithCopyWarning
df[df['age'] > 18]['category'] = 'adult'  # 可能修改的是副本

# 好：使用.loc
df.loc[df['age'] > 18, 'category'] = 'adult'
```

3. **使用内置函数代替apply**：
```python
# 不好：apply是Python循环，很慢
df['double'] = df['value'].apply(lambda x: x * 2)

# 好：使用向量化操作
df['double'] = df['value'] * 2

# 使用map操作映射值（比apply快）
df['grade'] = df['score'].map({90: 'A', 80: 'B', 70: 'C'})
```

4. **选择合适的数据类型**：
```python
# 查看当前数据类型的内存占用
df.memory_usage(deep=True).sum()

# 使用更小的数据类型
df['id'] = df['id'].astype('int32')
df['category'] = df['category'].astype('category')  # 低基数字符串用category
```

5. **使用eval/query进行高效计算**：
```python
# 大量列计算时使用eval
df.eval('''
    c = a + b
    d = c * 2
    e = d ** 2
''', inplace=True)

# 复杂过滤使用query
result = df.query('age > 18 and score > 90 and gender == "M"')
```

6. **合并操作优化**：
```python
# 设置索引后再merge
df1 = df1.set_index('user_id')
df2 = df2.set_index('user_id')
merged = df1.join(df2)  # 比merge快

# 使用指定的join方式，避免笛卡尔积
```

---

### 51. API接口优化有哪些常用策略？请结合Python Web框架举例说明

**考察点**：API性能优化实战

**标准答案**：
**优化策略**：

1. **请求数据压缩**：
```python
# Flask启用压缩
from flask_compress import Compress
Compress(app)

# FastAPI使用gzip中间件
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

2. **分页返回结果**：
```python
# 使用游标分页而不是OFFSET分页
@app.get("/users")
def get_users(limit: int = 20, cursor: int = 0):
    return {
        "data": User.query.filter(User.id > cursor).limit(limit).all(),
        "next_cursor": cursor + limit
    }
```

3. **使用缓存**：
```python
from functools import lru_cache
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route("/hot-products")
@cache.cached(timeout=300, key_prefix=lambda: request.path)
def get_hot_products():
    return Product.query.filter_by(is_hot=True).all()
```

4. **异步处理非核心逻辑**：
```python
# 不好：同步发送邮件
@app.post("/register")
def register(user_data):
    user = create_user(user_data)
    send_welcome_email(user)  # 阻塞！
    return {"status": "ok"}

# 好：异步发送
from celery import Celery

@celery.task
def async_send_email(user_id):
    user = User.query.get(user_id)
    send_welcome_email(user)

@app.post("/register")
def register(user_data):
    user = create_user(user_data)
    async_send_email.delay(user.id)  # 不阻塞
    return {"status": "ok"}
```

5. **批量接口设计**：
```python
# 支持批量操作，减少请求次数
@app.post("/users/batch")
def batch_create_users(users: List[UserSchema]):
    User.objects.bulk_create([User(**u.dict()) for u in users])
    return {"created": len(users)}
```

6. **响应字段筛选**：
```python
@app.get("/users/{user_id}")
def get_user(user_id: int, fields: str = None):
    user = User.query.get(user_id)
    if fields:
        field_list = fields.split(',')
        return {k: v for k, v in user.to_dict().items() if k in field_list}
    return user.to_dict()
```

---

### 52. 机器学习/深度学习推理时如何优化性能？列举至少5种方法

**考察点**：ML推理性能优化

**标准答案**：
**优化方法**：

1. **模型量化**：
```python
# 使用ONNX Runtime量化
import onnxruntime as rt
from onnxruntime.quantization import quantize_dynamic, QuantType

# 动态量化：FP32 -> INT8，速度提升2-4倍
quantize_dynamic("model.onnx", "model_quantized.onnx", weight_type=QuantType.QUInt8)
```

2. **模型剪枝**：
```python
import torch
import torch.nn.utils.prune as prune

# 剪枝：移除不重要的权重
module = model.conv1
prune.l1_unstructured(module, name="weight", amount=0.3)  # 剪枝30%
prune.remove(module, "weight")  # 永久化剪枝
```

3. **使用推理优化框架**：
```python
# TensorRT
import tensorrt as trt

# ONNX Runtime
session = rt.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])

# TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("scripted_model.pt")
```

4. **批处理（Batch Inference）**：
```python
# 合并多个请求一起处理，提高GPU利用率
def batch_infer(requests, batch_size=32):
    results = []
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        batch_tensor = preprocess(batch)
        batch_results = model(batch_tensor)
        results.extend(postprocess(batch_results))
    return results
```

5. **模型蒸馏**：
```python
# 训练小模型模仿大模型
teacher_model = load_big_model()
student_model = SmallModel()

# 知识蒸馏训练
def distillation_loss(student_logits, teacher_logits, labels, T=3, alpha=0.7):
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                        soft_targets, reduction='batchmean') * T * T
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

6. **使用GPU/TPU加速**：
7. **模型集成优化**：减少集成模型数量
8. **缓存频繁请求的预测结果**

---

### 53. 如何优化Python网络请求性能？列举异步、连接池等方案

**考察点**：网络IO优化

**标准答案**：
**优化方案**：

1. **使用连接池复用连接**：
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 创建带连接池的session
session = requests.Session()
retry_strategy = Retry(total=3, backoff_factor=1)
adapter = HTTPAdapter(
    pool_connections=10,  # 连接池大小
    pool_maxsize=50,      # 最大连接数
    max_retries=retry_strategy
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# 使用session发送请求（复用连接）
response = session.get('https://api.example.com/data')
```

2. **异步请求（aiohttp）**：
```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.json()

async def main(urls):
    # 连接池大小
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)  # 并发执行
    return results

# 100个请求并发
urls = [f'https://api.example.com/data/{i}' for i in range(100)]
results = asyncio.run(main(urls))
```

3. **HTTP/2支持**：使用httpx替代requests
```python
import httpx

# HTTP/2客户端
client = httpx.Client(http2=True)
response = client.get('https://api.example.com/data')
```

4. **请求压缩**：
```python
# 开启gzip压缩（requests默认支持）
headers = {'Accept-Encoding': 'gzip, deflate'}
response = session.get(url, headers=headers)
```

5. **缓存GET请求**：
```python
import requests_cache

# 自动缓存GET请求
session = requests_cache.CachedSession(
    'cache_name',
    expire_after=300,  # 5分钟过期
    allowable_methods=['GET']
)
```

6. **批量API调用**：
- 合并多个小请求为一个批量请求
- 减少TCP握手和TLS协商开销

---

### 54. 什么是内存泄漏？如何定位和解决Python应用中的内存泄漏？

**考察点**：内存泄漏排查能力

**标准答案**：
**内存泄漏定义**：程序中已不再使用的内存无法被垃圾回收器释放，导致内存占用持续增长。

**定位步骤**：

1. **使用tracemalloc初步定位**：
```python
import tracemalloc

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()

# 运行可疑代码
run_leaking_code()

snapshot2 = tracemalloc.take_snapshot()
# 比较差异
top_stats = snapshot2.compare_to(snapshot1, 'lineno')

print("[Top 10 memory differences]")
for stat in top_stats[:10]:
    print(stat)
```

2. **使用objgraph分析对象引用**：
```python
import objgraph

# 查看增长最多的类型
objgraph.show_growth()

# 查找特定对象的引用链
leaking_objects = objgraph.by_type('MyClass')
if leaking_objects:
    objgraph.show_backrefs(
        leaking_objects[:1],
        filename='backrefs.png',
        max_depth=10
    )
```

3. **使用memory_profiler逐行分析**：
```python
from memory_profiler import profile

@profile
def suspicious_function():
    # 可疑代码
    data = load_big_data()
    cache_store(data)
    return result
```

**常见泄漏场景和解决方案**：

1. **全局集合无限增长**：
```python
# 不好：全局列表无限追加
_global_cache = []

def add_to_cache(item):
    _global_cache.append(item)  # 永不清理！

# 好：限制大小，LRU淘汰
from collections import deque
_global_cache = deque(maxlen=1000)  # 最多1000个元素
```

2. **闭包持有外部变量引用**：
```python
# 不好：闭包持有大对象引用
def make_handler():
    big_data = load_huge_data()
    
    def handler():
        # 只用到big_data的一小部分
        return big_data[0]
    
    return handler

# 好：只保留需要的数据
def make_handler():
    big_data = load_huge_data()
    first_item = big_data[0]
    del big_data  # 显式删除不需要的引用
    
    def handler():
        return first_item
    
    return handler
```

3. **未关闭的资源**：
```python
# 不好：不关闭文件/连接
def read_file():
    f = open('data.txt')  # 没有关闭！
    return f.read()

# 好：使用with语句
def read_file():
    with open('data.txt') as f:  # 自动关闭
        return f.read()
```

4. **循环引用带__del__方法**：
```python
# 避免：循环引用+__del__导致无法回收
class BadClass:
    def __del__(self):  # 有析构函数
        pass

a = BadClass()
b = BadClass()
a.ref = b
b.ref = a  # 循环引用，GC无法回收
```

---

### 55. 大规模数据处理时，如何选择合适的Python框架？对比Dask、PySpark、Vaex的适用场景

**考察点**：大数据处理技术选型

**标准答案**：
**框架对比**：

| 特性 | Dask | PySpark | Vaex |
|------|------|---------|------|
| 学习曲线 | 低（类Pandas API） | 高（Spark生态） | 中 |
| 单机能力 | 强 | 一般 | 最强（内存映射） |
| 集群能力 | 有 | 成熟强大 | 有限 |
| SQL支持 | 有限 | 完善 | 有 |
| 延迟 | 低 | 高（JVM开销） | 低 |
| 数据格式 | Parquet/CSV等 | 丰富 | HDF5/Parquet/Arrow |

**选型建议**：

1. **Dask适用场景**：
```python
# 适合：Pandas代码需要扩展，不想重写
import dask.dataframe as dd

# API几乎和Pandas一样
df = dd.read_csv('large_data_*.csv')
result = df.groupby('category').value.mean().compute()

# 优势：
# - 最小化代码修改
# - 平滑从小规模过渡到集群
# - 与Python生态无缝集成
```

2. **PySpark适用场景**：
```python
# 适合：TB级数据，已有Hadoop生态，复杂ETL
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("example").getOrCreate()
df = spark.read.parquet("hdfs:///data/large_dataset")
result = df.groupBy("category").agg({"value": "mean"}).collect()

# 优势：
# - 成熟的集群管理
# - SQL查询优化器
# - 内置MLlib流处理等
```

3. **Vaex适用场景**：
```python
# 适合：单机超大数据（100GB+），交互式探索
import vaex

# 内存映射，不加载全部数据
df = vaex.open('huge_data.hdf5')  # 100GB+数据，内存占用几MB
result = df[df.x > 0].groupby(df.category).agg({'value': 'mean'})

# 优势：
# - 极致的内存效率
# - 快速交互式数据探索
# - 支持懒执行和即时计算
```

**决策流程图**：
```
数据大小 < 内存 -> 使用Pandas
内存 < 数据 < 10*内存 -> 使用Dask
数据 > 10*内存 + 已有Spark集群 -> 使用PySpark
单机 + 超大数据 + 交互式分析 -> 使用Vaex
需要复杂SQL/流处理/ML管道 -> 使用PySpark
```

---

## 总结

本文档整理了55道Python性能优化高频面试题，覆盖五大分类：
1. **GIL和并发基础**（15题）：多线程、多进程、协程、异步IO等
2. **内存优化**（10题）：垃圾回收、内存池、生成器、弱引用等
3. **性能分析工具**（10题）：cProfile、py-spy、memory_profiler、火焰图等
4. **代码优化技巧**（10题）：向量化、循环优化、JIT编译、字符串优化等
5. **业务场景优化**（10题）：大文件处理、数据库优化、缓存系统、Web服务优化等

每题包含详细的**考察点说明**和**标准答案**，并配有**示例代码**，帮助深入理解和实际应用。