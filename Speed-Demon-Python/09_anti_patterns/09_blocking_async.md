# 09 - 异步代码中调用阻塞 IO

## 问题描述

在 asyncio 异步代码中调用阻塞函数（如同步 IO、sleep 等）会阻塞整个事件循环，导致其他协程无法执行。这会严重降低并发能力，让异步代码变得跟单线程一样慢。

## 示例代码

```python
import asyncio
import requests

# 反模式：在异步函数中调用阻塞 IO
async def fetch_url_bad(url):
    # requests.get 是阻塞的，会卡住整个事件循环
    response = requests.get(url)
    return response.text

async def main_bad():
    urls = ['http://example.com'] * 10
    # 这些请求实际上是串行执行的
    tasks = [fetch_url_bad(url) for url in urls]
    results = await asyncio.gather(*tasks)
```

## 性能对比

运行 `python 09_blocking_async.py` 会输出类似：

```
=== 反模式：异步代码中调用阻塞函数 ===
处理 10 个任务耗时: 10.52 秒
实际上是串行执行！

=== 正确方案 1：使用异步 IO 库 ===
处理 10 个任务耗时: 1.23 秒
真正并发执行！

=== 正确方案 2：使用线程池运行阻塞函数 ===
处理 10 个任务耗时: 2.15 秒
并发执行，但有线程开销
```

## 为什么会这样？

1. **事件循环是单线程的**：同一时间只能执行一个协程
2. **阻塞调用会暂停整个事件循环**：其他所有任务都要等待
3. **失去异步优势**：异步变成了伪并发

## 修复方案

### 方案 1：使用真正的异步 IO 库

```python
import asyncio
import aiohttp  # 异步 HTTP 客户端

async def fetch_url_good(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main_good():
    urls = ['http://example.com'] * 10
    tasks = [fetch_url_good(url) for url in urls]
    results = await asyncio.gather(*tasks)
```

### 方案 2：使用线程池运行阻塞函数

```python
import asyncio
import requests

async def fetch_url_with_threadpool(url):
    loop = asyncio.get_event_loop()
    # 在线程池中运行阻塞函数
    response = await loop.run_in_executor(
        None,  # 使用默认线程池
        requests.get,
        url
    )
    return response.text
```

### 方案 3：使用 asyncio.to_thread（Python 3.9+）

```python
import asyncio
import requests

async def fetch_url_with_to_thread(url):
    # 更简洁的写法
    response = await asyncio.to_thread(requests.get, url)
    return response.text
```

## 常见的阻塞调用和异步替代

| 操作 | 阻塞方式 | 异步替代 |
|-----|---------|---------|
| HTTP 请求 | requests | aiohttp, httpx |
| 文件 IO | open, read | aiofiles, asyncio.to_thread |
| 数据库 | psycopg2 | asyncpg, aiopg |
| Redis | redis-py | aioredis |
| 等待 | time.sleep | asyncio.sleep |
| 进程 | subprocess.Popen | asyncio.create_subprocess_exec |

## 如何检测阻塞调用

### 使用 asyncio 调试模式

```python
import asyncio

# 启用调试模式，检测慢回调
loop = asyncio.get_event_loop()
loop.set_debug(True)
# 设置慢回调阈值（秒）
loop.slow_callback_duration = 0.1
```

### 使用 aiodebug 库

```python
# pip install aiodebug
import aiodebug.log_slow_callbacks
aiodebug.log_slow_callbacks.enable(0.05)  # 50ms 阈值
```

### 自定义检测

```python
import time
import asyncio
import logging

class BlockingDetector:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
    
    def __call__(self, task):
        if task._state != 'PENDING':
            return
        
        start = time.time()
        
        def callback(fut):
            elapsed = time.time() - start
            if elapsed > self.threshold:
                logging.warning(
                    f"任务 {task.get_name()} 阻塞了 {elapsed:.3f} 秒, "
                    f"可能包含阻塞调用"
                )
        
        task.add_done_callback(callback)

# 使用
detector = BlockingDetector(threshold=0.1)
for task in asyncio.all_tasks():
    detector(task)
```

## 混合编程模式

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 创建专用线程池
io_executor = ThreadPoolExecutor(max_workers=20)

async def hybrid_approach():
    # 纯异步任务
    async_result = await async_operation()
    
    # 阻塞任务在线程池运行
    def blocking_func():
        return sync_io_operation()
    
    sync_result = await asyncio.to_thread(blocking_func)
    
    return async_result, sync_result
```

## 总结

1. **永远不要在 async 函数中调用阻塞函数**
2. **使用异步版本的库**：aiohttp, asyncpg, aioredis 等
3. **迫不得已时使用线程池**：`asyncio.to_thread` 或 `loop.run_in_executor`
4. **启用调试模式检测阻塞**：及早发现问题
5. **使用 `asyncio.sleep` 代替 `time.sleep`**

记住：**异步代码的性能优势来自于不阻塞事件循环**。只要有一个地方阻塞了，整个系统的并发能力就会大打折扣！
