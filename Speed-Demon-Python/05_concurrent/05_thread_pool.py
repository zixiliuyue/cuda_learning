"""
05 - 与线程池结合处理阻塞调用
当异步代码中必须调用阻塞函数时,使用线程池避免阻塞事件循环

关键点:
1. 同步/阻塞代码会阻塞整个事件循环
2. 使用 run_in_executor 将阻塞代码放到线程池
3. CPU 密集型用 ProcessPoolExecutor, IO 密集型用 ThreadPoolExecutor
"""

import asyncio
import time
import concurrent.futures
import random


def blocking_io_task(task_id, duration=0.5):
    """模拟阻塞的 IO 操作"""
    time.sleep(duration)
    return f"task_{task_id}_io_done"


def blocking_cpu_task(task_id, iterations=1000000):
    """模拟 CPU 密集型计算"""
    result = 0
    for i in range(iterations):
        result += i
    return f"task_{task_id}_cpu_done"


async def bad_blocking_code():
    """
    ❌ 错误方式: 在协程中直接调用阻塞代码
    这会阻塞整个事件循环
    """
    print("\n" + "=" * 70)
    print("错误方式: 直接在协程中调用阻塞代码")
    print("=" * 70)
    
    start = time.time()
    
    async def async_blocking(task_id):
        """看起来是异步,实际是阻塞的"""
        result = blocking_io_task(task_id, 0.5)
        print(f"  {result}")
        return result
    
    # 虽然用了 gather,但实际上是串行执行的!
    tasks = [async_blocking(i) for i in range(4)]
    await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.2f}s")
    print(f"预期并发耗时: ~0.5s")
    print("问题: 阻塞代码卡住了整个事件循环!")


async def good_with_thread_pool():
    """
    ✅ 正确方式: 使用线程池运行阻塞代码
    """
    print("\n" + "=" * 70)
    print("正确方式: 使用线程池运行阻塞 IO")
    print("=" * 70)
    
    start = time.time()
    
    loop = asyncio.get_running_loop()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [
            loop.run_in_executor(executor, blocking_io_task, i, 0.5)
            for i in range(4)
        ]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            print(f"  {result}")
    
    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.2f}s")
    print(f"加速比: ~{2.0/elapsed:.1f}x")


async def cpu_bound_with_process_pool():
    """
    CPU 密集型任务用进程池
    避免 GIL 限制
    """
    print("\n" + "=" * 70)
    print("CPU 密集型: 使用进程池")
    print("=" * 70)
    
    loop = asyncio.get_running_loop()
    
    print("使用线程池 (受 GIL 限制):")
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [
            loop.run_in_executor(executor, blocking_cpu_task, i, 2000000)
            for i in range(4)
        ]
        await asyncio.gather(*tasks)
    thread_time = time.time() - start
    print(f"  耗时: {thread_time:.2f}s")
    
    await asyncio.sleep(0.5)
    
    print("\n使用进程池 (规避 GIL):")
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        tasks = [
            loop.run_in_executor(executor, blocking_cpu_task, i, 2000000)
            for i in range(4)
        ]
        await asyncio.gather(*tasks)
    process_time = time.time() - start
    print(f"  耗时: {process_time:.2f}s")
    
    print(f"\n进程池相对线程池加速: {thread_time/process_time:.2f}x")


async def mixed_async_and_blocking():
    """
    混合异步和阻塞代码的最佳实践
    """
    print("\n" + "=" * 70)
    print("混合场景: 异步 IO 与 阻塞代码 协同")
    print("=" * 70)
    
    async def async_task(task_id):
        """真正的异步任务"""
        await asyncio.sleep(random.uniform(0.1, 0.3))
        return f"async_{task_id}_done"
    
    def sync_task(task_id):
        """必须同步执行的任务"""
        time.sleep(random.uniform(0.1, 0.3))
        return f"sync_{task_id}_done"
    
    loop = asyncio.get_running_loop()
    
    start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # 异步任务
        async_tasks = [async_task(i) for i in range(4)]
        
        # 阻塞任务放入线程池
        sync_tasks = [
            loop.run_in_executor(executor, sync_task, i + 4)
            for i in range(4)
        ]
        
        all_results = await asyncio.gather(*(async_tasks + sync_tasks))
        
        for result in all_results:
            print(f"  {result}")
    
    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.2f}s")
    print(f"串行预计: ~{0.2*8:.1f}s")


class AsyncExecutor:
    """
    封装好的执行器 - 生产环境推荐
    """
    
    def __init__(self, max_threads=10, max_processes=4):
        self.max_threads = max_threads
        self.max_processes = max_processes
        self.thread_pool = None
        self.process_pool = None
    
    async def __aenter__(self):
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_threads
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_processes
        )
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
    
    async def run_io(self, func, *args, **kwargs):
        """运行 IO 密集型任务"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: func(*args, **kwargs)
        )
    
    async def run_cpu(self, func, *args, **kwargs):
        """运行 CPU 密集型任务"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.process_pool,
            lambda: func(*args, **kwargs)
        )


async def demo_executor():
    """演示封装好的执行器"""
    print("\n" + "=" * 70)
    print("生产环境: 使用封装的执行器")
    print("=" * 70)
    
    async with AsyncExecutor(max_threads=4, max_processes=2) as executor:
        start = time.time()
        
        io_results = await asyncio.gather(*[
            executor.run_io(blocking_io_task, f"io_{i}", 0.2)
            for i in range(4)
        ])
        
        cpu_results = await asyncio.gather(*[
            executor.run_cpu(blocking_cpu_task, f"cpu_{i}", 1000000)
            for i in range(2)
        ])
        
        elapsed = time.time() - start
        
        for result in io_results + cpu_results:
            print(f"  {result}")
        
        print(f"\n总耗时: {elapsed:.2f}s")


def best_practices():
    """最佳实践总结"""
    print("\n" + "=" * 70)
    print("最佳实践总结")
    print("=" * 70)
    
    practices = """
1. 永远不要在 asyncio 协程中直接调用阻塞函数
   - time.sleep(), requests.get(), 同步文件 IO 等都会阻塞事件循环
   - 使用对应的异步版本: asyncio.sleep(), aiohttp, aiofiles 等

2. 当必须使用阻塞代码时:
   - IO 密集型 → ThreadPoolExecutor
   - CPU 密集型 → ProcessPoolExecutor

3. 线程池 vs 进程池:
   - 线程池: 内存共享,开销小,受 GIL 限制
   - 进程池: 内存独立,开销大,不受 GIL 限制

4. 参数建议:
   - ThreadPoolExecutor: max_workers 可以设为 10-100
   - ProcessPoolExecutor: max_workers 建议为 CPU 核心数

5. 常见需要线程池的场景:
   - 使用没有异步版本的第三方库
   - 调用数据库同步驱动
   - 执行系统命令 subprocess
   - 复杂的正则表达式匹配
"""
    print(practices)


if __name__ == "__main__":
    asyncio.run(bad_blocking_code())
    asyncio.run(good_with_thread_pool())
    asyncio.run(cpu_bound_with_process_pool())
    asyncio.run(mixed_async_and_blocking())
    asyncio.run(demo_executor())
    best_practices()
