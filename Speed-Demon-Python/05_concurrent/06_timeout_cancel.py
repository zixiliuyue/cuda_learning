"""
06 - 超时和取消机制
正确处理任务超时和取消,避免资源泄漏

关键点:
1. asyncio.wait_for() 实现超时
2. 正确捕获和处理 CancelledError
3. 任务取消时的清理工作
"""

import asyncio
import time


async def long_running_task(task_id, duration=5.0):
    """模拟长时间运行的任务"""
    try:
        print(f"  任务 {task_id}: 开始执行,预计 {duration}s")
        await asyncio.sleep(duration)
        print(f"  任务 {task_id}: 正常完成")
        return f"task_{task_id}_success"
    except asyncio.CancelledError:
        print(f"  任务 {task_id}: 被取消,正在清理...")
        await asyncio.sleep(0.1)
        print(f"  任务 {task_id}: 清理完成")
        raise


async def demo_wait_for():
    """
    asyncio.wait_for() 基础用法
    """
    print("\n" + "=" * 70)
    print("asyncio.wait_for() - 任务超时控制")
    print("=" * 70)
    
    print("\n场景1: 任务在超时前完成:")
    try:
        result = await asyncio.wait_for(long_running_task(1, 0.5), timeout=2.0)
        print(f"  结果: {result}")
    except asyncio.TimeoutError:
        print("  任务超时了!")
    
    print("\n场景2: 任务超时:")
    try:
        result = await asyncio.wait_for(long_running_task(2, 3.0), timeout=1.0)
        print(f"  结果: {result}")
    except asyncio.TimeoutError:
        print("  任务超时了!")
    
    await asyncio.sleep(0.5)


async def demo_task_cancel():
    """
    手动取消任务
    """
    print("\n" + "=" * 70)
    print("手动取消任务")
    print("=" * 70)
    
    task = asyncio.create_task(long_running_task(1, 5.0))
    
    await asyncio.sleep(1.0)
    
    print(f"\n  主线程: 1s 后取消任务...")
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print(f"  主线程: 确认任务已取消")
    
    print(f"  任务状态: done={task.done()}, cancelled={task.cancelled()}")


async def demo_cancel_multiple():
    """
    批量取消任务
    """
    print("\n" + "=" * 70)
    print("批量取消任务")
    print("=" * 70)
    
    tasks = [
        asyncio.create_task(long_running_task(i, 3.0))
        for i in range(5)
    ]
    
    await asyncio.sleep(1.0)
    
    print("\n  开始批量取消所有任务...")
    for task in tasks:
        task.cancel()
    
    print("  等待所有任务清理完成...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    cancelled_count = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
    print(f"\n  取消完成: {cancelled_count}/{len(tasks)} 个任务被取消")


async def demo_shield():
    """
    asyncio.shield() 的使用
    保护关键任务不被取消
    """
    print("\n" + "=" * 70)
    print("asyncio.shield() - 保护关键任务不被取消")
    print("=" * 70)
    
    async def critical_task():
        print("  关键任务: 开始 (不可取消)")
        await asyncio.sleep(2.0)
        print("  关键任务: 完成")
        return "critical_data"
    
    async def normal_task():
        print("  普通任务: 开始 (可取消)")
        await asyncio.sleep(2.0)
        print("  普通任务: 完成")
        return "normal_data"
    
    print("\n场景1: 没有 shield 保护:")
    task1 = asyncio.create_task(normal_task())
    await asyncio.sleep(0.5)
    task1.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        print("  普通任务: 被取消了")
    
    await asyncio.sleep(0.5)
    
    print("\n场景2: 有 shield 保护:")
    task2 = asyncio.create_task(shielded_task := critical_task())
    shielded = asyncio.shield(shielded_task)
    await asyncio.sleep(0.5)
    task2.cancel()
    try:
        await shielded
        print("  关键任务: 即使外层取消,仍然完成了!")
    except asyncio.CancelledError:
        print("  关键任务: 被取消了")


async def demo_timeout_gather():
    """
    为 gather() 中的任务设置超时
    """
    print("\n" + "=" * 70)
    print("gather() 中的超时控制")
    print("=" * 70)
    
    async def task_with_random_delay(task_id):
        delay = task_id * 0.5
        await asyncio.sleep(delay)
        print(f"  任务 {task_id}: 完成 (耗时 {delay}s)")
        return task_id
    
    print("\n场景1: 整个 gather 设置超时 (2s):")
    tasks = [task_with_random_delay(i) for i in range(1, 6)]
    
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
        print(f"  全部完成: {results}")
    except asyncio.TimeoutError:
        print("  超时! 部分任务可能已经完成,部分被取消")
    
    await asyncio.sleep(1.0)
    
    print("\n场景2: 每个任务单独设置超时:")
    tasks = [
        asyncio.wait_for(task_with_random_delay(i), timeout=i * 0.6)
        for i in range(1, 6)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results, 1):
        if isinstance(result, asyncio.TimeoutError):
            print(f"  任务 {i}: 超时")
        else:
            print(f"  任务 {i}: 成功,结果={result}")


async def demo_timeout_with_retry():
    """
    超时 + 重试的实用模式
    """
    print("\n" + "=" * 70)
    print("实用模式: 超时 + 重试")
    print("=" * 70)
    
    async def unreliable_api(call_id):
        """模拟不稳定的 API"""
        import random
        delay = random.uniform(0.1, 2.0)
        print(f"  API 调用 {call_id}: 开始,预计 {delay:.2f}s")
        await asyncio.sleep(delay)
        if delay > 1.0:
            print(f"  API 调用 {call_id}: 太慢了,超时!")
            raise TimeoutError(f"API {call_id} timeout")
        print(f"  API 调用 {call_id}: 成功")
        return f"result_{call_id}"
    
    async def call_with_retry(call_id, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await asyncio.wait_for(unreliable_api(f"{call_id}_try{attempt+1}"), timeout=1.0)
            except (asyncio.TimeoutError, TimeoutError) as e:
                if attempt == max_retries - 1:
                    print(f"  调用 {call_id}: 重试 {max_retries} 次全部失败!")
                    raise
                backoff = 0.1 * (attempt + 1)
                print(f"  调用 {call_id}: 第 {attempt+1} 次失败, {backoff}s 后重试...")
                await asyncio.sleep(backoff)
    
    start = time.time()
    results = await asyncio.gather(*[
        call_with_retry(i) for i in range(3)
    ], return_exceptions=True)
    elapsed = time.time() - start
    
    print(f"\n总耗时: {elapsed:.2f}s")
    print(f"结果: {results}")


def best_practices():
    """最佳实践总结"""
    print("\n" + "=" * 70)
    print("超时和取消 最佳实践")
    print("=" * 70)
    
    practices = """
1. 始终为网络请求设置超时
   - 永远不要 await 一个没有超时的网络请求
   - 使用 asyncio.wait_for() 设置超时

2. 正确处理 CancelledError
   - 在 try-except 中捕获并清理资源
   - 清理完成后重新 raise,不要"吞掉"异常
   - 不要在 CancelledError 处理中做耗时操作

3. shield() 的使用场景
   - 需要保证完成的关键操作
   - 注意: shield 只保护 await 的部分,任务本身仍会被标记为取消
   - shield 不能保护阻塞代码

4. 取消时的清理工作
   - 关闭打开的文件、网络连接
   - 释放锁、信号量等同步原语
   - 记录取消日志,便于调试

5. 常见模式
   - 单个任务超时: wait_for(task, timeout)
   - 批量超时: wait_for(gather(...), timeout)
   - 每个任务不同超时: gather(wait_for(t1, t1_timeout), ...)
   - 超时重试: 循环 + wait_for + 指数退避
"""
    print(practices)


if __name__ == "__main__":
    asyncio.run(demo_wait_for())
    asyncio.run(demo_task_cancel())
    asyncio.run(demo_cancel_multiple())
    asyncio.run(demo_shield())
    asyncio.run(demo_timeout_gather())
    asyncio.run(demo_timeout_with_retry())
    best_practices()
