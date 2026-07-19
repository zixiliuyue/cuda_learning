# asyncio 进阶优化指南

> 配套代码：`01_connection_pool.py` ~ `07_aiohttp_vs_requests.py`
> 测试服务器：`simple_server.py`

---

## 目录

1. [正确使用连接池](#1-正确使用连接池)
2. [限制并发数 (Semaphore)](#2-限制并发数-semaphore)
3. [任务批量处理](#3-任务批量处理)
4. [asyncio 高级 API 对比](#4-asyncio-高级-api-对比)
5. [与线程池结合处理阻塞调用](#5-与线程池结合处理阻塞调用)
6. [超时和取消机制](#6-超时和取消机制)
7. [aiohttp vs requests 性能对比](#7-aiohttp-vs-requests-性能对比)

---

## 1. 正确使用连接池

### 核心问题

很多人在使用 aiohttp 时会犯一个常见错误：**每个请求创建新的 ClientSession**

```python
# ❌ 错误方式：每个请求创建新 Session
async def bad_approach(urls):
    for url in urls:
        async with aiohttp.ClientSession() as session:  # 每次新建！
            async with session.get(url) as response:
                ...
```

**问题：**
- 每次创建 Session 意味着新的 TCP 连接
- TCP 握手开销（SYN → SYN-ACK → ACK）
- 无法复用 HTTP keep-alive 连接
- 性能差 ~5-10 倍

### 正确做法

```python
# ✅ 正确方式：复用 Session
async def good_approach(urls):
    async with aiohttp.ClientSession() as session:  # 只创建一次
        for url in urls:
            async with session.get(url) as response:
                ...
```

### 优化配置

```python
# ✅ 最佳方式：配置连接池参数
connector = aiohttp.TCPConnector(
    limit=50,              # 最大连接数
    limit_per_host=10,     # 每个主机最大连接数
    ttl_dns_cache=300,     # DNS 缓存 TTL
    use_dns_cache=True,    # 启用 DNS 缓存
    force_close=False       # 保持连接
)

async with aiohttp.ClientSession(connector=connector) as session:
    ...
```

### 性能对比

| 方式 | 50 个请求耗时 | 相对性能 |
|------|--------------|----------|
| 每次新建 Session | ~10s | 1.0x |
| 复用 Session | ~2s | 5.0x |
| 连接池 + 并发 | ~0.5s | 20.0x |

### 生产环境建议

1. **全局单例 Session**：整个应用只创建一个 ClientSession
2. **合理设置 limit**：
   - 抓取单个域名：`limit_per_host = 5-10`
   - 抓取多个域名：`limit = 50-200`
3. **异常处理**：始终处理连接异常、超时

---

## 2. 限制并发数 (Semaphore)

### 为什么需要限制并发？

- 目标服务器可能有 rate limit 限制
- 过多并发会导致：
  - 连接超时
  - 服务器拒绝服务（503）
  - 本地端口耗尽
  - 内存占用过高

### 使用 asyncio.Semaphore

```python
# ✅ 用信号量限制并发
async def semaphore_limited(urls, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_limit(session, url):
        async with semaphore:  # 获取信号量，超过限制会等待
            async with session.get(url) as response:
                return await response.json()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_limit(session, url) for url in urls]
        return await asyncio.gather(*tasks)
```

### 并发数设置参考

| 场景 | 推荐并发数 |
|------|-----------|
| 公共 API（有明确限制） | 按文档要求 |
| 普通网站 | 5-20 |
| 内部服务 | 20-100 |
| 高吞吐服务 | 100+ |

### 自适应并发（高级）

根据服务器响应动态调整并发数：

```python
async def adaptive_fetch(session, url, semaphore):
    nonlocal success_count, error_count
    
    try:
        async with semaphore:
            async with session.get(url) as response:
                if response.status == 200:
                    success_count += 1
                    if success_count % 50 == 0:  # 连续成功，增加并发
                        semaphore._value = min(semaphore._value + 1, 50)
                    return True
                else:
                    error_count += 1
                    if semaphore._value > 5:  # 出错，降低并发
                        semaphore._value = max(semaphore._value - 1, 5)
                    return False
    except Exception:
        error_count += 1
        if semaphore._value > 5:
            semaphore._value = max(semaphore._value - 1, 5)
        return False
```

### 常见错误

```python
# ❌ 错误：信号量创建位置不对
for url in urls:
    semaphore = asyncio.Semaphore(10)  # 每个任务都创建新信号量！
    task = fetch_with_semaphore(semaphore, url)

# ✅ 正确：所有任务共享同一个信号量
semaphore = asyncio.Semaphore(10)
for url in urls:
    task = fetch_with_semaphore(semaphore, url)
```

---

## 3. 任务批量处理

### 问题场景

当任务数量非常大时（如 10,000+）：
- 一次性创建所有任务会占用大量内存
- 任务对象本身就有开销
- GC 压力大

### 分批处理模式

```python
async def process_in_batches(items, batch_size=20):
    all_results = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # 只创建当前批次的任务
            tasks = [session.get(url) for url in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed = [process_result(r) for r in batch_results]
            all_results.extend(processed)
            
            print(f"Progress: {min(i + batch_size, len(items))}/{len(items)}")
    
    return all_results
```

### 流式处理 vs 分批处理

| 方式 | 内存占用 | 实现复杂度 | 适用场景 |
|------|---------|-----------|---------|
| 一次性全部 | 高 | 简单 | 任务数少 (<100) |
| 分批处理 | 低 | 中等 | 任务数多 (100-10,000) |
| 信号量流式 | 中 | 中等 | 需要平滑处理 |

### 带进度和重试的批处理器

```python
class BatchProcessor:
    def __init__(self, batch_size=20, max_retries=3):
        self.batch_size = batch_size
        self.max_retries = max_retries
    
    async def process_item(self, session, item, retry_count=0):
        try:
            async with session.get(item) as response:
                return await response.json()
        except Exception as e:
            if retry_count < self.max_retries:
                await asyncio.sleep(0.1 * (retry_count + 1))
                return await self.process_item(session, item, retry_count + 1)
            raise
    
    async def process_all(self, items):
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i + self.batch_size]
                tasks = [self.process_item(session, item) for item in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                yield results
```

---

## 4. asyncio 高级 API 对比

### gather, wait, as_completed 对比

| 特性 | gather() | as_completed() | wait() |
|------|---------|---------------|--------|
| 返回时机 | 全部完成后 | 逐个完成时 | 可配置 |
| 结果顺序 | 与输入一致 | 完成顺序 | 无序 |
| 超时支持 | 需要配合 | 需要配合 | ✓ 原生支持 |
| 异常策略 | 全失败/收集异常 | 抛出第一个异常 | 需单独检测 |
| 取消未完成 | ✗ | ✗ | ✓ |
| 内存占用 | 低 | 低 | 中 |

### asyncio.gather()

**最常用的并发原语**

```python
# 等待所有任务完成，按输入顺序返回
results = await asyncio.gather(task1, task2, task3)

# 捕获异常而不中断
results = await asyncio.gather(task1, task2, return_exceptions=True)
```

**适用场景：**
- 需要所有任务的结果
- 结果顺序很重要
- 简单的批量处理

### asyncio.as_completed()

**按完成顺序逐个返回**

```python
# 先完成的先返回，不等待全部
tasks = [fetch(url) for url in urls]
for coro in asyncio.as_completed(tasks):
    result = await coro
    print(f"Got result: {result}")  # 可以立即处理，不用等全部
```

**适用场景：**
- 流式处理（爬虫实时写入数据库）
- 用户需要快速看到部分结果
- 任务之间没有依赖关系

### asyncio.wait()

**最灵活的控制方式**

```python
# 只等待第一个完成
done, pending = await asyncio.wait(
    tasks,
    return_when=asyncio.FIRST_COMPLETED
)

# 第一个异常就返回
done, pending = await asyncio.wait(
    tasks,
    return_when=asyncio.FIRST_EXCEPTION
)

# 带超时
done, pending = await asyncio.wait(tasks, timeout=5.0)

# 取消未完成的任务
for task in pending:
    task.cancel()
```

### 选择建议

```
简单批量处理 → gather()
流式处理，先完成先处理 → as_completed()
需要超时、取消、分批完成 → wait()
```

---

## 5. 与线程池结合处理阻塞调用

### 核心问题

asyncio 的最大敌人：**阻塞调用**

```python
# ❌ 灾难：在协程中调用阻塞函数
async def bad_example():
    result = requests.get(url)  # 这会阻塞整个事件循环！
    # 在 requests.get 等待期间，所有其他协程都被卡住了
```

**为什么这是灾难？**
- asyncio 是单线程事件循环
- 只要有一个地方阻塞，整个程序就卡住
- 相当于在高速公路中间停车

### 解决方案 1：使用异步库

```python
# ✅ 正确：使用异步版的库
async def good_example():
    async with aiohttp.ClientSession() as session:
        result = await session.get(url)  # 真正的异步
```

### 解决方案 2：线程池运行阻塞代码

当没有异步版本的库时：

```python
# ✅ 将阻塞代码放到线程池
async def threadpool_example():
    loop = asyncio.get_running_loop()
    
    def blocking_function():
        # 这里可以调用任何阻塞代码
        return requests.get(url).json()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        result = await loop.run_in_executor(executor, blocking_function)
```

### CPU 密集型：使用进程池

```python
# ✅ CPU 密集型用 ProcessPoolExecutor
async def cpu_bound_example():
    loop = asyncio.get_running_loop()
    
    def heavy_calculation():
        result = 0
        for i in range(1000000):
            result += i
        return result
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        result = await loop.run_in_executor(executor, heavy_calculation)
```

**为什么进程池适合 CPU 密集型？**
- 避开 GIL 限制
- 可以真正并行利用多核
- 缺点：进程创建开销大，数据传递慢

### 线程池 vs 进程池 对比

| 维度 | ThreadPoolExecutor | ProcessPoolExecutor |
|------|-------------------|---------------------|
| 内存开销 | 低 | 高 |
| 启动开销 | 低 | 高 |
| 数据共享 | 容易（同进程） | 困难（跨进程） |
| GIL 限制 | 是 | 否 |
| 适用 | IO 密集型 | CPU 密集型 |

### 常见需要线程池的场景

```python
# 1. 使用没有异步版本的第三方库
await loop.run_in_executor(None, sync_library_call)

# 2. 调用数据库同步驱动
await loop.run_in_executor(None, db_query)

# 3. 执行系统命令
await loop.run_in_executor(None, subprocess.run, ["ls", "-l"])

# 4. 复杂正则表达式
await loop.run_in_executor(None, re.match, pattern, text)
```

### 最佳实践

```python
# ✅ 封装执行器
class AsyncExecutor:
    def __init__(self, max_threads=10, max_processes=4):
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_threads)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_processes)
    
    async def run_io(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: func(*args, **kwargs)
        )
    
    async def run_cpu(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.process_pool,
            lambda: func(*args, **kwargs)
        )
```

---

## 6. 超时和取消机制

### 永远设置超时！

**网络编程第一原则：永远不要无限等待**

```python
# ❌ 危险：没有超时
async with session.get(url) as response:  # 可能永远卡住！
    ...

# ✅ 正确：使用 asyncio.wait_for 设置超时
try:
    result = await asyncio.wait_for(
        session.get(url),
        timeout=10.0  # 最多等 10 秒
    )
except asyncio.TimeoutError:
    print("Request timed out")
```

### 任务取消

```python
# 创建任务
task = asyncio.create_task(long_running_operation())

# 稍后取消
await asyncio.sleep(5.0)
task.cancel()

# 等待取消完成
try:
    await task
except asyncio.CancelledError:
    print("Task was cancelled")
```

### 正确处理 CancelledError

```python
async def cancellable_task():
    try:
        # 正常工作
        await do_something()
    except asyncio.CancelledError:
        # ⚠️ 重要：清理资源
        await cleanup_resources()
        # ⚠️ 重要：重新 raise，不要吞掉异常
        raise
```

**取消时应该清理的资源：**
- 打开的文件句柄
- 网络连接
- 数据库事务
- 锁和信号量
- 临时文件

### asyncio.shield()

保护关键任务不被取消：

```python
# 即使外层取消，critical_operation() 仍会完成
async def important_work():
    try:
        await asyncio.shield(critical_operation())
    except asyncio.CancelledError:
        # 外层被取消了，但 critical_operation 完成了
        await save_result()
        raise
```

**适用场景：**
- 数据库写入不能中断
- 文件写入需要完成
- 事务需要提交或回滚

### 超时 + 重试模式

```python
async def fetch_with_retry(url, max_retries=3, timeout=10.0):
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(
                session.get(url),
                timeout=timeout
            )
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            if attempt == max_retries - 1:
                raise
            # 指数退避
            backoff = 0.5 * (2 ** attempt)
            await asyncio.sleep(backoff)
```

---

## 7. aiohttp vs requests 性能对比

### 测试场景

测试条件：
- 100 个 HTTP 请求
- 本地测试服务器（响应 ~50ms）
- 测量总耗时和吞吐量

### 性能对比结果

| 方式 | 耗时 | 吞吐量 (req/s) | 相对性能 |
|------|------|---------------|---------|
| 纯同步 (requests) | ~25.0s | 4.0 | 1.0x |
| 同步 + Session 复用 | ~5.0s | 20.0 | 5.0x |
| 线程池 + requests (20 线程) | ~2.5s | 40.0 | 10.0x |
| aiohttp + Semaphore (20 并发) | ~0.3s | 333.3 | 83.3x |
| aiohttp 无限制并发 | ~0.06s | 1666.7 | 416.7x |

### 性能差距来源

**为什么差异这么大？**

1. **同步请求**：
   - 每个请求都在阻塞等待
   - CPU 大部分时间空闲
   - 利用率 ~5-10%

2. **线程池**：
   - 可以并行处理
   - 但线程切换有开销
   - 线程数量有限制
   - 利用率 ~50-60%

3. **异步 IO**：
   - 等待期间可以处理其他请求
   - 几乎没有切换开销
   - 利用率 ~90%+

### 什么时候用哪个？

**简单场景 (< 10 个请求)**
→ 用 requests，简单够用，不用考虑并发

**批量场景 (> 10 个请求)**
→ 用 aiohttp + Semaphore，性能最好

**已有同步代码，不想大改**
→ 用 ThreadPoolExecutor + requests，过渡方案

**需要上传大文件/流式响应**
→ 用 aiohttp，支持异步流式处理

### 常见误区

```python
# ❌ 误区1：以为用 asyncio 就自动变快
async def still_slow():
    # 还是同步！只是包了一层 async
    result = requests.get(url)

# ✅ 正确：用真正异步的库
async def fast():
    result = await session.get(url)
```

```python
# ❌ 误区2：无限并发
async def too_fast():
    # 1000 个请求同时发出！
    tasks = [session.get(url) for _ in range(1000)]
    # 服务器：你想 DOS 我？

# ✅ 正确：限制并发
semaphore = asyncio.Semaphore(20)
tasks = [fetch_with_limit(semaphore, url) for _ in range(1000)]
```

---

## 综合最佳实践清单

### 连接管理
- [ ] 全局复用一个 ClientSession
- [ ] 合理配置连接池参数
- [ ] 用完正确关闭连接

### 并发控制
- [ ] 使用 Semaphore 限制并发
- [ ] 避免无限并发
- [ ] 根据目标服务器调整并发数

### 任务管理
- [ ] 大量任务分批处理
- [ ] 选择正确的 API（gather/as_completed/wait）
- [ ] 实时显示进度

### 阻塞处理
- [ ] 优先使用异步库
- [ ] IO 密集型用 ThreadPoolExecutor
- [ ] CPU 密集型用 ProcessPoolExecutor

### 错误处理
- [ ] 所有网络请求设置超时
- [ ] 正确处理 CancelledError
- [ ] 实现超时重试机制

---

## 性能优化 Checklist

从慢到快的优化路径：

```
纯同步 requests
    ↓ 5x
复用 Session
    ↓ 2x
ThreadPoolExecutor
    ↓ 8x
aiohttp + Semaphore
    ↓ 5x
(极限) 无限制并发
```

**实际性能提升：~400x**

---

## 运行示例代码

```bash
# 1. 启动测试服务器
python simple_server.py

# 2. 打开新终端，运行各个示例
python 01_connection_pool.py
python 02_semaphore.py
python 03_batch_processing.py
python 04_asyncio_apis.py
python 05_thread_pool.py
python 06_timeout_cancel.py
python 07_aiohttp_vs_requests.py
```

---

**总结**：asyncio 的性能来自于**高效的等待** — 不要让 CPU 在等待时闲着！
