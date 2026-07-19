import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# 模拟阻塞 IO 操作
def blocking_io(task_id, duration=1.0):
    """模拟阻塞的 IO 操作"""
    print(f"任务 {task_id} 开始 (线程: {threading.current_thread().name})")
    time.sleep(duration)  # 阻塞调用
    print(f"任务 {task_id} 完成")
    return f"result_{task_id}"

# ============================================
# 反模式：在异步函数中调用阻塞函数
# ============================================
async def blocking_coroutine(task_id):
    """反模式：直接在协程中调用阻塞函数"""
    # ❌ 这里会阻塞整个事件循环！
    result = blocking_io(task_id, duration=0.5)
    return result

async def demo_blocking_anti_pattern():
    print("\n=== 反模式：异步代码中调用阻塞函数 ===")
    print("注意：所有任务实际上是串行执行的！")
    
    start_time = time.time()
    
    # 创建 10 个任务
    tasks = [blocking_coroutine(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"处理 10 个任务耗时: {elapsed:.2f} 秒")
    print(f"预期（串行）: ~5.0 秒")
    print(f"结果数: {len(results)}")

# ============================================
# 正确方案 1：使用 asyncio.sleep 替代 time.sleep
# ============================================
async def non_blocking_coroutine(task_id):
    """正确：使用异步等待"""
    print(f"任务 {task_id} 开始 (线程: {threading.current_thread().name})")
    await asyncio.sleep(0.5)  # ✅ 非阻塞
    print(f"任务 {task_id} 完成")
    return f"result_{task_id}"

async def demo_async_io():
    print("\n=== 正确方案 1：使用真正的异步操作 ===")
    print("所有任务真正并发执行！")
    
    start_time = time.time()
    
    tasks = [non_blocking_coroutine(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"处理 10 个任务耗时: {elapsed:.2f} 秒")
    print(f"预期（并发）: ~0.5 秒")
    print(f"结果数: {len(results)}")

# ============================================
# 正确方案 2：使用线程池运行阻塞函数
# ============================================
async def run_blocking_in_threadpool(task_id):
    """正确：在线程池中运行阻塞函数"""
    loop = asyncio.get_event_loop()
    # 使用默认线程池运行阻塞函数
    result = await loop.run_in_executor(
        None,
        blocking_io,
        task_id,
        0.5
    )
    return result

async def demo_threadpool():
    print("\n=== 正确方案 2：使用线程池运行阻塞函数 ===")
    print("阻塞函数在线程池中并发执行！")
    
    start_time = time.time()
    
    tasks = [run_blocking_in_threadpool(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"处理 10 个任务耗时: {elapsed:.2f} 秒")
    print(f"预期（线程池并发）: ~1.0-1.5 秒")
    print(f"结果数: {len(results)}")

# ============================================
# 正确方案 3：使用 asyncio.to_thread (Python 3.9+)
# ============================================
async def run_with_to_thread(task_id):
    """使用 asyncio.to_thread (Python 3.9+)"""
    result = await asyncio.to_thread(
        blocking_io,
        task_id,
        0.5
    )
    return result

async def demo_to_thread():
    print("\n=== 正确方案 3：使用 asyncio.to_thread ===")
    print("更简洁的线程池写法 (Python 3.9+)！")
    
    start_time = time.time()
    
    tasks = [run_with_to_thread(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"处理 10 个任务耗时: {elapsed:.2f} 秒")
    print(f"结果数: {len(results)}")

# ============================================
# 性能对比
# ============================================
async def compare_performance():
    print("\n" + "="*60)
    print("性能对比汇总")
    print("="*60)
    
    # 反模式
    start = time.time()
    tasks = [blocking_coroutine(i) for i in range(5)]
    await asyncio.gather(*tasks)
    blocking_time = time.time() - start
    
    # 纯异步
    start = time.time()
    tasks = [non_blocking_coroutine(i) for i in range(5)]
    await asyncio.gather(*tasks)
    async_time = time.time() - start
    
    # 线程池
    start = time.time()
    tasks = [run_blocking_in_threadpool(i) for i in range(5)]
    await asyncio.gather(*tasks)
    threadpool_time = time.time() - start
    
    print(f"{'方式':<30} {'耗时':>10} {'相对速度':>12}")
    print("-" * 54)
    print(f"{'反模式（阻塞事件循环）':<30} {blocking_time:>10.2f}s {'1.0x':>12}")
    print(f"{'纯异步':<30} {async_time:>10.2f}s {f'{blocking_time/async_time:.1f}x':>12}")
    print(f"{'线程池':<30} {threadpool_time:>10.2f}s {f'{blocking_time/threadpool_time:.1f}x':>12}")

# ============================================
# 检测阻塞调用的演示
# ============================================
async def detect_blocking_calls():
    """演示如何检测阻塞调用"""
    print("\n=== 检测阻塞调用 ===")
    print("启用 asyncio 调试模式可以检测慢回调")
    
    loop = asyncio.get_event_loop()
    original_debug = loop.get_debug()
    
    # 启用调试模式
    loop.set_debug(True)
    loop.slow_callback_duration = 0.1  # 100ms 阈值
    
    print("调试模式已开启")
    print("慢回调阈值设置为: 0.1 秒")
    
    # 恢复原始设置
    loop.set_debug(original_debug)

# ============================================
# 混合模式：异步 + 线程池
# ============================================
async def hybrid_demo():
    print("\n=== 混合模式：异步 + 线程池 ===")
    
    # 纯异步任务
    async def async_task():
        await asyncio.sleep(0.2)
        return "async result"
    
    # 阻塞任务（用线程池）
    def blocking_task():
        time.sleep(0.2)
        return "blocking result"
    
    start = time.time()
    
    results = await asyncio.gather(
        async_task(),
        asyncio.to_thread(blocking_task),
        async_task(),
        asyncio.to_thread(blocking_task)
    )
    
    elapsed = time.time() - start
    print(f"4 个混合任务耗时: {elapsed:.2f} 秒")
    print(f"结果: {results}")

async def main():
    print("="*60)
    print("异步代码中调用阻塞 IO 的反模式演示")
    print("="*60)
    
    # 反模式
    await demo_blocking_anti_pattern()
    
    # 正确方案
    await demo_async_io()
    await demo_threadpool()
    
    # 尝试使用 to_thread (如果 Python 版本支持)
    try:
        await demo_to_thread()
    except AttributeError:
        print("\n=== asyncio.to_thread 需要 Python 3.9+ ===")
    
    # 性能对比
    await compare_performance()
    
    # 检测阻塞调用
    await detect_blocking_calls()
    
    # 混合模式
    await hybrid_demo()
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("❌ 不要在 async 函数中调用阻塞函数 (time.sleep, requests.get 等)")
    print("✅ 使用异步版本的库 (aiohttp, asyncpg 等)")
    print("✅ 迫不得已时使用 asyncio.to_thread 或线程池")
    print("✅ 使用 asyncio.sleep 代替 time.sleep")
    print("✅ 启用调试模式检测慢回调")

if __name__ == "__main__":
    asyncio.run(main())
