# asyncio 异步编程理论

## 1. 协程原理 vs 线程 vs 进程

### 1.1 基本概念

**进程 (Process)**
- 操作系统资源分配的最小单位
- 拥有独立的内存空间和系统资源
- 进程间通信（IPC）成本高
- 由操作系统内核调度

**线程 (Thread)**
- CPU调度的最小单位
- 共享进程的内存空间
- 线程间通信成本低，但需要同步机制
- 由操作系统内核调度（抢占式）

**协程 (Coroutine)**
- 用户态的轻量级执行单元
- 共享同一线程的内存空间
- 由程序员手动调度（协作式）
- 上下文切换成本极低

### 1.2 性能对比

| 特性 | 进程 | 线程 | 协程 |
|------|------|------|------|
| 内存开销 | 大（MB级） | 中（KB级） | 极小（字节级） |
| 切换开销 | 很大 | 中等 | 极小 |
| 调度方式 | 内核抢占 | 内核抢占 | 用户协作 |
| 数据共享 | 复杂（IPC） | 简单（需同步） | 最简单 |
| 并发数量 | 数十级 | 数百级 | 数万级 |

### 1.3 适用场景

- **进程**: CPU密集型任务，需要充分利用多核
- **线程**: I/O密集型任务，但任务数不多
- **协程**: 高并发I/O密集型任务，如网络爬虫、Web服务器

---

## 2. asyncio 事件循环工作机制

### 2.1 事件循环核心概念

事件循环是asyncio的核心，负责：
1. 管理所有协程的执行
2. 处理I/O事件的回调
3. 调度任务的执行顺序

### 2.2 工作原理

```
┌─────────────────────────────────────────┐
│           事件循环 (Event Loop)         │
│  ┌─────────┐    ┌───────────────────┐  │
│  │  就绪队列 │───▶│  执行协程/回调    │  │
│  └─────────┘    └─────────┬─────────┘  │
│                           │            │
│  ┌─────────┐              ▼            │
│  │  等待队列 │◀─────────I/O 等待        │
│  └─────────┘                           │
└─────────────────────────────────────────┘
```

1. **就绪队列**: 存放可以立即执行的协程/任务
2. **执行**: 从就绪队列取出任务执行，直到遇到await挂起
3. **等待队列**: 存放等待I/O或定时器事件的任务
4. **事件检测**: 当I/O完成或定时器到期，将任务移回就绪队列

### 2.3 事件循环的关键职责

```python
import asyncio

async def main():
    # 获取当前事件循环
    loop = asyncio.get_running_loop()
    print(f"当前事件循环: {loop}")
    
    # 调用soon线程安全版本
    loop.call_soon(lambda: print("立即执行"))
    loop.call_later(1.0, lambda: print("1秒后执行"))

asyncio.run(main())
```

---

## 3. await/async 关键字原理

### 3.1 async 关键字

**作用**: 将函数标记为协程函数

```python
# 协程函数定义
async def my_coroutine():
    return "Hello"

# 调用协程函数返回协程对象，不会立即执行
coro = my_coroutine()
print(type(coro))  # <class 'coroutine'>
```

**原理**:
- async函数在字节码层面添加了`CO_COROUTINE`标志
- 调用时不执行函数体，而是返回一个协程对象
- 需要通过事件循环或await来执行

### 3.2 await 关键字

**作用**: 挂起当前协程，等待异步操作完成

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)  # 模拟I/O操作
    return "数据"

async def main():
    # await 挂起main，等待fetch_data完成
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

**原理**:
1. 遇到`await`时，协程将控制权交还给事件循环
2. 事件循环去执行其他就绪的协程
3. 当await的操作完成后，协程被重新调度执行
4. 从await处继续执行后续代码

### 3.3 可await对象

可以被await的对象类型：
1. **协程对象 (coroutine)**
2. **Task对象**
3. **Future对象**
4. 实现了`__await__`方法的对象

---

## 4. Task 和 Future

### 4.1 Future

Future是一个"占位符"，表示一个尚未完成的异步操作的结果。

```python
import asyncio

async def future_example():
    loop = asyncio.get_running_loop()
    
    # 创建Future对象
    future = loop.create_future()
    
    # 设置Future结果
    def set_result():
        future.set_result("Future完成了")
    
    loop.call_later(1, set_result)
    
    # 等待Future完成
    result = await future
    print(result)

asyncio.run(future_example())
```

**Future状态**:
- PENDING: 等待完成
- DONE: 已完成（成功或失败）
- CANCELLED: 已取消

### 4.2 Task

Task是Future的子类，用于包装协程，使其在事件循环中并发执行。

```python
import asyncio

async def task_func(name, seconds):
    print(f"任务 {name} 开始")
    await asyncio.sleep(seconds)
    print(f"任务 {name} 完成")
    return f"结果-{name}"

async def main():
    # 创建Task（立即开始调度）
    task1 = asyncio.create_task(task_func("A", 2))
    task2 = asyncio.create_task(task_func("B", 1))
    
    # 等待任务完成
    result1 = await task1
    result2 = await task2
    
    print(f"任务1结果: {result1}")
    print(f"任务2结果: {result2}")

asyncio.run(main())
```

### 4.3 Task vs Future

| 特性 | Future | Task |
|------|--------|------|
| 代表 | 异步操作的结果 | 包装协程的执行 |
| 创建方式 | loop.create_future() | asyncio.create_task() |
| 手动设置结果 | 是 | 否（由协程返回） |
| 包装协程 | 否 | 是 |

### 4.4 常用Task API

```python
import asyncio

async def example():
    # 1. gather - 并发运行多个任务
    results = await asyncio.gather(
        asyncio.sleep(1, result="A"),
        asyncio.sleep(2, result="B"),
        asyncio.sleep(0.5, result="C")
    )
    print(f"gather结果: {results}")  # ['A', 'B', 'C']
    
    # 2. wait - 等待任务集合
    tasks = [asyncio.create_task(asyncio.sleep(i, result=i)) for i in range(3)]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
    
    # 3. as_completed - 按完成顺序迭代
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"完成: {result}")
    
    # 4. timeout - 设置超时
    try:
        await asyncio.wait_for(asyncio.sleep(5), timeout=2)
    except asyncio.TimeoutError:
        print("超时了!")

asyncio.run(example())
```

---

## 5. 异步I/O性能优势

### 5.1 为什么异步I/O更快？

**同步I/O模型**:
```
线程1: [请求1] [等待I/O...] [处理1] [请求2] [等待I/O...] [处理2]
线程2:          [请求3] [等待I/O...] [处理3]
```
- 线程在I/O时被阻塞，浪费CPU资源
- 线程切换有开销

**异步I/O模型**:
```
单线程: [请求1]→切换→[请求2]→切换→[请求3]→切换→[处理1]→切换→[处理2]...
        (I/O等待时执行其他任务)
```
- I/O等待时不阻塞，执行其他任务
- 用户态切换，开销极小
- 单线程可以处理数千并发连接

### 5.2 同步 vs 异步性能对比代码

```python
import time
import asyncio
import requests
import aiohttp

# ----------------------
# 同步版本
# ----------------------
def sync_fetch(url):
    """同步HTTP请求"""
    response = requests.get(url)
    return response.status_code

def sync_main(urls):
    """同步主函数"""
    start = time.time()
    results = [sync_fetch(url) for url in urls]
    elapsed = time.time() - start
    return elapsed, results

# ----------------------
# 异步版本
# ----------------------
async def async_fetch(session, url):
    """异步HTTP请求"""
    async with session.get(url) as response:
        return response.status

async def async_main(urls):
    """异步主函数"""
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [async_fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    return elapsed, results

# ----------------------
# 性能测试
# ----------------------
def benchmark():
    urls = [
        "https://httpbin.org/delay/1",  # 模拟1秒延迟
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1",
    ]
    
    print(f"测试 {len(urls)} 个并发请求...")
    
    # 同步测试
    print("\n=== 同步版本 ===")
    sync_time, _ = sync_main(urls)
    print(f"耗时: {sync_time:.2f} 秒")
    
    # 异步测试
    print("\n=== 异步版本 ===")
    async_time, _ = asyncio.run(async_main(urls))
    print(f"耗时: {async_time:.2f} 秒")
    
    # 对比
    print("\n=== 性能对比 ===")
    print(f"速度提升: {sync_time / async_time:.1f}x")

if __name__ == "__main__":
    benchmark()
```

**预期输出**:
```
测试 5 个并发请求...

=== 同步版本 ===
耗时: 5.23 秒

=== 异步版本 ===
耗时: 1.15 秒

=== 性能对比 ===
速度提升: 4.5x
```

### 5.3 性能优势总结

| 场景 | 同步 | 异步 | 提升倍数 |
|------|------|------|----------|
| 5个并发请求 | ~5秒 | ~1秒 | 5x |
| 10个并发请求 | ~10秒 | ~1秒 | 10x |
| 100个并发请求 | ~100秒 | ~1秒 | 100x |

---

## 6. asyncio 常见陷阱

### 6.1 陷阱1: 阻塞调用

**问题**: 在协程中使用同步阻塞函数会阻塞整个事件循环

```python
# 错误示例
async def bad_example():
    # requests是同步阻塞的！会阻塞整个事件循环
    response = requests.get("https://example.com")
    return response
```

**正确做法**:
```python
# 正确做法1: 使用异步库
async def good_example1():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://example.com") as response:
            return await response.text()

# 正确做法2: 在线程池中运行阻塞函数
async def good_example2():
    loop = asyncio.get_running_loop()
    # 将阻塞调用放到线程池中执行
    result = await loop.run_in_executor(
        None,
        requests.get,
        "https://example.com"
    )
    return result
```

**检测阻塞调用**:
```python
import asyncio

# 启用调试模式检测阻塞
asyncio.run(main(), debug=True)
```

### 6.2 陷阱2: 错误处理不足

**问题**: Task中的异常如果不await会被忽略

```python
# 错误示例
async def bad_error_handling():
    # 创建Task但不await，异常会被忽略
    asyncio.create_task(raise_error())
    await asyncio.sleep(1)
    print("任务可能已经失败了，但我们不知道")

async def raise_error():
    await asyncio.sleep(0.5)
    raise Exception("任务失败!")
```

**正确做法**:
```python
# 正确做法1: await Task
async def good_error_handling1():
    task = asyncio.create_task(raise_error())
    try:
        await task
    except Exception as e:
        print(f"捕获到异常: {e}")

# 正确做法2: 添加回调
def handle_exception(task):
    try:
        task.result()
    except Exception as e:
        print(f"回调捕获异常: {e}")

async def good_error_handling2():
    task = asyncio.create_task(raise_error())
    task.add_done_callback(handle_exception)
    await asyncio.sleep(1)

# 正确做法3: 全局异常处理
def global_exception_handler(loop, context):
    print(f"全局异常: {context['exception']}")

async def main():
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(global_exception_handler)
```

### 6.3 陷阱3: 无限创建Task

**问题**: 无限制创建Task会导致内存耗尽

```python
# 错误示例
async def bad_task_creation():
    while True:
        # 无限创建Task，内存会爆炸
        asyncio.create_task(some_work())
        await asyncio.sleep(0.001)
```

**正确做法**: 使用信号量限制并发
```python
async def good_task_creation():
    semaphore = asyncio.Semaphore(100)  # 限制最多100个并发
    
    async def limited_work():
        async with semaphore:
            return await some_work()
    
    while True:
        asyncio.create_task(limited_work())
        await asyncio.sleep(0.001)
```

### 6.4 陷阱4: 错误的超时处理

**问题**: 使用asyncio.sleep()模拟超时但没有取消任务

```python
# 错误示例
async def bad_timeout():
    task = asyncio.create_task(long_running_task())
    # 等待超时，但任务还在后台运行
    await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
```

**正确做法**:
```python
async def good_timeout():
    task = asyncio.create_task(long_running_task())
    try:
        result = await asyncio.wait_for(task, timeout=1.0)
    except asyncio.TimeoutError:
        print("超时，取消任务")
        task.cancel()
        try:
            await task  # 等待任务真正取消
        except asyncio.CancelledError:
            print("任务已取消")
```

### 6.5 陷阱5: 忘记await协程

**问题**: 只调用协程函数但不await，协程不会执行

```python
# 错误示例
async def bad_example():
    my_coroutine()  # 只是创建了协程对象，不会执行
    await asyncio.sleep(1)
    # 警告: coroutine 'my_coroutine' was never awaited
```

**正确做法**:
```python
async def good_example():
    await my_coroutine()  # 正确执行
    # 或者创建Task
    asyncio.create_task(my_coroutine())
```

### 6.6 陷阱6: 混合同步和异步代码

**问题**: 在同步函数中调用异步代码

```python
# 错误示例
def sync_func():
    # 不能在同步函数中直接await
    result = await async_func()  # 语法错误！
    return result
```

**正确做法**:
```python
# 正确做法1: 从同步代码运行异步函数
def sync_func1():
    result = asyncio.run(async_func())
    return result

# 正确做法2: 在线程中运行事件循环
def sync_func2():
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(async_func())
    loop.close()
    return result
```

---

## 7. 最佳实践总结

1. **使用异步库**: 优先使用aiohttp、aiopg等异步库
2. **避免阻塞**: 不要在协程中调用同步阻塞函数
3. **正确处理异常**: 总是await Task或添加错误回调
4. **限制并发**: 使用信号量控制并发数量
5. **使用调试模式**: 开发时开启debug检测阻塞
6. **合理取消任务**: 超时时记得取消任务
7. **资源清理**: 使用async with/async try-finally

---

## 8. 完整的性能对比示例

```python
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 模拟I/O操作
def io_bound_task(task_id):
    """模拟耗时1秒的I/O操作"""
    time.sleep(1)
    return f"任务{task_id}完成"

# ----------------------
# 1. 纯同步版本
# ----------------------
def sync_version(num_tasks):
    start = time.time()
    results = [io_bound_task(i) for i in range(num_tasks)]
    elapsed = time.time() - start
    return elapsed

# ----------------------
# 2. 多线程版本
# ----------------------
def thread_version(num_tasks, num_threads=10):
    start = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(io_bound_task, range(num_tasks)))
    elapsed = time.time() - start
    return elapsed

# ----------------------
# 3. 异步版本
# ----------------------
async def async_io_bound_task(task_id):
    """异步I/O操作模拟"""
    await asyncio.sleep(1)
    return f"任务{task_id}完成"

async def async_main(num_tasks):
    start = time.time()
    tasks = [async_io_bound_task(i) for i in range(num_tasks)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    return elapsed

def async_version(num_tasks):
    return asyncio.run(async_main(num_tasks))

# ----------------------
# 性能对比
# ----------------------
def full_benchmark():
    num_tasks = 100
    
    print(f"=== {num_tasks} 个任务的性能对比 ===\n")
    
    # 异步版本
    async_time = async_version(num_tasks)
    print(f"异步版本: {async_time:.2f} 秒")
    
    # 多线程版本
    thread_time = thread_version(num_tasks, num_threads=20)
    print(f"多线程版本: {thread_time:.2f} 秒")
    
    # 纯同步版本（只跑10个，不然太慢）
    sync_time = sync_version(10)
    print(f"纯同步版本: {sync_time:.2f} 秒 (仅10个任务)")
    
    print(f"\n异步 vs 多线程: {thread_time / async_time:.1f}x 提升")
    print(f"异步 vs 纯同步: {(sync_time * 10) / async_time:.1f}x 提升")

if __name__ == "__main__":
    full_benchmark()
```

**预期输出**:
```
=== 100 个任务的性能对比 ===

异步版本: 1.01 秒
多线程版本: 5.05 秒
纯同步版本: 10.02 秒 (仅10个任务)

异步 vs 多线程: 5.0x 提升
异步 vs 纯同步: 99.2x 提升
```

这个示例清晰地展示了异步I/O在高并发场景下的巨大优势。
