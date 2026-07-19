"""
04 - asyncio 高级 API 对比
详解 gather, wait, as_completed 的区别和适用场景

关键点:
1. gather: 等待全部完成,按输入顺序返回
2. wait: 可自定义等待条件(FIRST_COMPLETED等)
3. as_completed: 按完成顺序逐个返回
"""

import asyncio
import time
import random


async def task_with_delay(task_id, delay=None):
    """模拟带延迟的任务"""
    if delay is None:
        delay = random.uniform(0.1, 1.0)
    await asyncio.sleep(delay)
    result = f"task_{task_id}_done"
    print(f"  ✓ {result} (delay={delay:.2f}s)")
    return result


async def demo_gather():
    """
    asyncio.gather() 演示
    - 等待所有任务完成
    - 按输入顺序返回结果
    - 一个失败默认全部失败,使用 return_exceptions=True 捕获
    """
    print("\n" + "=" * 70)
    print("asyncio.gather() - 等待全部完成, 按输入顺序返回")
    print("=" * 70)
    
    start = time.time()
    
    tasks = [task_with_delay(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    print(f"\n总耗时: {elapsed:.2f}s")
    print(f"结果顺序: {results}")
    print("\n特点:")
    print("  ✓ 等待所有任务完成")
    print("  ✓ 结果顺序与输入顺序一致")
    print("  ✗ 单个失败会导致全部失败(除非 return_exceptions=True)")
    print("  适用: 需要所有结果,且关心顺序的场景")


async def demo_as_completed():
    """
    asyncio.as_completed() 演示
    - 按完成顺序逐个返回
    - 可以及时处理先完成的任务
    """
    print("\n" + "=" * 70)
    print("asyncio.as_completed() - 按完成顺序逐个返回")
    print("=" * 70)
    
    start = time.time()
    
    tasks = [task_with_delay(i) for i in range(5)]
    results = []
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        print(f"  → 已收到结果: {result}, 立即处理...")
    
    elapsed = time.time() - start
    
    print(f"\n总耗时: {elapsed:.2f}s")
    print(f"结果顺序: {results}")
    print("\n特点:")
    print("  ✓ 按完成顺序返回,可以及时处理")
    print("  ✓ 无需等待所有任务完成")
    print("  ✗ 结果顺序与输入顺序不一致")
    print("  适用: 流式处理,先完成先处理的场景")


async def demo_wait():
    """
    asyncio.wait() 演示
    - 支持多种等待策略:
      - ALL_COMPLETED: 等待所有完成 (默认)
      - FIRST_COMPLETED: 等待第一个完成
      - FIRST_EXCEPTION: 等待第一个异常
    """
    print("\n" + "=" * 70)
    print("asyncio.wait() - 灵活的等待策略")
    print("=" * 70)
    
    tasks = [task_with_delay(i, delay=0.5 + i * 0.2) for i in range(5)]
    
    start = time.time()
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    elapsed = time.time() - start
    
    print(f"\nFIRST_COMPLETED 模式:")
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  已完成: {len(done)} 个")
    print(f"  待处理: {len(pending)} 个")
    
    for task in pending:
        task.cancel()
    
    await asyncio.sleep(0.1)
    
    print("\n" + "-" * 70)
    
    tasks2 = [task_with_delay(i, delay=0.3 + i * 0.1) for i in range(5)]
    start = time.time()
    done, pending = await asyncio.wait(tasks2, timeout=0.5)
    elapsed = time.time() - start
    
    print(f"\n带 timeout 的 wait:")
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  已完成: {len(done)} 个")
    print(f"  超时未完成: {len(pending)} 个")
    
    for task in pending:
        task.cancel()
    
    print("\n特点:")
    print("  ✓ 支持多种等待策略")
    print("  ✓ 支持超时控制")
    print("  ✓ 可以取消未完成的任务")
    print("  适用: 需要精细化控制的复杂场景")


async def demo_return_exceptions():
    """
    演示异常处理
    """
    print("\n" + "=" * 70)
    print("异常处理对比")
    print("=" * 70)
    
    async def may_fail(task_id):
        await asyncio.sleep(0.1)
        if task_id == 2:
            raise ValueError(f"任务 {task_id} 失败了!")
        return f"task_{task_id}_done"
    
    print("\n1. gather 默认行为 (失败即停止):")
    try:
        tasks = [may_fail(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        print(f"  结果: {results}")
    except ValueError as e:
        print(f"  捕获异常: {e}")
    
    print("\n2. gather return_exceptions=True:")
    tasks = [may_fail(i) for i in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"  结果: {results}")
    exceptions = [r for r in results if isinstance(r, Exception)]
    print(f"  异常数量: {len(exceptions)}")
    
    print("\n3. wait + 异常检测:")
    tasks = [asyncio.create_task(may_fail(i)) for i in range(5)]
    done, pending = await asyncio.wait(tasks)
    for task in done:
        if task.exception():
            print(f"  检测到异常: {task.exception()}")
        else:
            print(f"  成功: {task.result()}")


def api_comparison_table():
    """API 对比表"""
    print("\n" + "=" * 70)
    print("API 对比总结")
    print("=" * 70)
    
    comparison = """
| 特性            | gather()          | as_completed()    | wait()            |
|-----------------|-------------------|-------------------|-------------------|
| 返回时机        | 全部完成后        | 逐个完成时        | 可配置            |
| 结果顺序        | 与输入一致        | 完成顺序          | 无序              |
| 超时支持        | ✗ 需要配合        | ✗ 需要配合        | ✓                 |
| 异常策略        | 全失败/收集异常   | 抛出第一个异常    | 需单独检测        |
| 取消未完成      | ✗                 | ✗                 | ✓                 |
| 内存占用        | 低(一次性返回)    | 低(流式返回)      | 中(需管理pending) |
| 适用场景        | 批量处理          | 流式处理          | 复杂控制          |

选择建议:
1. 简单批量任务 → gather()
2. 流式处理,先完成先处理 → as_completed()
3. 需要超时、取消、分批完成 → wait()
"""
    print(comparison)


async def real_world_example():
    """
    实际场景: 爬虫的结果聚合
    使用 as_completed 实现"先完成先写入"
    """
    print("\n" + "=" * 70)
    print("实际场景: 爬虫数据流式处理")
    print("=" * 70)
    
    async def fetch_url(url_id):
        delay = random.uniform(0.1, 0.8)
        await asyncio.sleep(delay)
        return {"url_id": url_id, "data": f"content_{url_id}", "fetch_time": delay}
    
    urls = list(range(10))
    tasks = [fetch_url(url_id) for url_id in urls]
    
    print("开始爬取,先完成先写入数据库...")
    start = time.time()
    
    completed_count = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed_count += 1
        print(f"  [{completed_count}/10] 写入: {result['url_id']}, 耗时: {result['fetch_time']:.2f}s")
    
    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.2f}s")
    print("优势: 用户可以更快看到数据,内存占用更平缓")


if __name__ == "__main__":
    asyncio.run(demo_gather())
    asyncio.run(demo_as_completed())
    asyncio.run(demo_wait())
    asyncio.run(demo_return_exceptions())
    api_comparison_table()
    asyncio.run(real_world_example())
