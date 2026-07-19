# 08 - 无限创建线程

## 问题描述

线程炸弹（Thread Bomb）是指在没有控制的情况下不断创建新线程，导致系统资源耗尽。每个线程都需要占用内存（栈空间）和操作系统资源，创建太多线程会导致系统变慢、响应迟钝甚至崩溃。

## 示例代码

```python
import threading

def process_request(request):
    # 处理请求...
    pass

# 反模式：每个请求创建一个新线程，没有限制
def handle_request(request):
    thread = threading.Thread(target=process_request, args=(request,))
    thread.start()
    # 线程结束后也不做任何清理

# 在高并发下，会创建成千上万个线程
while True:
    request = accept_connection()
    handle_request(request)
```

## 性能对比

运行 `python 08_thread_bomb.py` 会输出类似：

```
=== 无限制创建线程 ===
创建 100 个线程耗时: 0.02 秒, 当前线程数: 101
创建 200 个线程耗时: 0.05 秒, 当前线程数: 201
创建 300 个线程耗时: 0.08 秒, 当前线程数: 301
...
内存使用随线程数线性增长，系统越来越慢！

=== 使用线程池 ===
处理 1000 个任务耗时: 1.23 秒
活跃线程数稳定在: 10
内存使用: 稳定
```

## 为什么会这样？

1. **线程创建开销**：每个线程创建需要时间和内存
2. **栈空间占用**：每个线程默认有 8MB 栈空间（Linux）
3. **上下文切换开销**：太多线程会导致频繁的 CPU 上下文切换
4. **系统限制**：操作系统对最大线程数有限制

## 修复方案

### 方案 1：使用线程池

```python
from concurrent.futures import ThreadPoolExecutor

# 创建固定大小的线程池
executor = ThreadPoolExecutor(max_workers=10)

def handle_request(request):
    # 提交任务到线程池，而不是创建新线程
    executor.submit(process_request, request)
```

### 方案 2：使用信号量限制并发数

```python
import threading

max_concurrent = 10
semaphore = threading.Semaphore(max_concurrent)

def process_request(request):
    with semaphore:
        # 只有获得信号量的线程才能执行这里
        do_work(request)
```

### 方案 3：使用队列 + 工作线程模式

```python
import queue
import threading

work_queue = queue.Queue(maxsize=100)

def worker():
    while True:
        task = work_queue.get()
        if task is None:
            break
        process_task(task)
        work_queue.task_done()

# 启动固定数量的工作线程
for _ in range(10):
    threading.Thread(target=worker, daemon=True).start()

# 提交任务
def handle_request(request):
    try:
        work_queue.put_nowait(request)
    except queue.Full:
        # 队列已满，拒绝请求或降级处理
        return_503_error()
```

### 方案 4：使用异步 I/O 代替多线程

```python
import asyncio

async def handle_request(request):
    # 使用异步 I/O，不需要多线程
    result = await async_process(request)
    return result

# 创建事件循环
async def main():
    server = await asyncio.start_server(handle_request, '0.0.0.0', 8080)
    async with server:
        await server.serve_forever()

asyncio.run(main())
```

## 线程数量的最佳实践

### 1. CPU 密集型任务

```python
import os

# 线程数 = CPU 核心数
num_threads = os.cpu_count() or 4
```

### 2. IO 密集型任务

```python
# IO 等待时间占比高，可以设置更多线程
# 但也不要太多，一般 2-4 倍 CPU 核心数
num_threads = (os.cpu_count() or 4) * 2
```

### 3. 使用有界队列防止过载

```python
from concurrent.futures import ThreadPoolExecutor
import queue

class BoundedExecutor:
    def __init__(self, max_workers=10, queue_size=100):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = threading.Semaphore(queue_size)
    
    def submit(self, func, *args, **kwargs):
        self.semaphore.acquire()
        try:
            future = self.executor.submit(func, *args, **kwargs)
            future.add_done_callback(lambda _: self.semaphore.release())
            return future
        except Exception:
            self.semaphore.release()
            raise
```

## 监控和告警

```python
import threading
import psutil
import logging

def monitor_threads():
    while True:
        current_threads = threading.active_count()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        
        if current_threads > 100:
            logging.warning(f"线程数过高: {current_threads}")
        
        if memory_usage > 1000:  # 1GB
            logging.warning(f"内存使用过高: {memory_usage:.2f} MB")
        
        time.sleep(60)

# 启动监控线程
monitor = threading.Thread(target=monitor_threads, daemon=True)
monitor.start()
```

## 不同并发方案对比

| 方案 | 资源消耗 | 适用场景 | 复杂度 |
|-----|---------|---------|-------|
| 无限制线程 | 极高 | ❌ 不要用 | 低 |
| 线程池 | 可控 | IO 密集型 | 中 |
| 信号量限制 | 可控 | 需要动态调整 | 中 |
| 异步 I/O | 很低 | 高并发 IO | 高 |
| 多进程 | 高 | CPU 密集型 | 中 |

## 总结

1. **永远不要无限制创建线程** - 使用线程池或信号量
2. **合理设置线程池大小** - 根据任务类型（CPU/IO密集）
3. **使用有界队列** - 防止任务堆积导致 OOM
4. **考虑异步 I/O** - 对于高并发 IO 场景
5. **监控线程数量和内存使用** - 设置告警阈值

记住：**更多线程 ≠ 更快**。超过一定数量后，线程越多反而越慢！
