"""
02 - 限制并发数 (Semaphore)
使用信号量控制并发请求数量,防止服务器过载

关键点:
1. 无限制并发可能导致目标服务器拒绝服务
2. Semaphore 可以精确控制并发数
3. 合理的并发数 = 吞吐量和稳定性的平衡
"""

import asyncio
import time
import aiohttp


async def unlimited_concurrency(urls):
    """
    无限制并发
    问题: 瞬间发起所有请求,可能触发服务器限流
    """
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for response in responses:
            if not isinstance(response, Exception):
                results.append(await response.json())
                await response.release()
            else:
                results.append(None)
        return results


async def semaphore_limited(urls, max_concurrent=10):
    """
    使用 Semaphore 限制并发
    优势: 可控的并发,避免服务器过载
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_limit(session, url):
        async with semaphore:
            async with session.get(url) as response:
                return await response.json()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_limit(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)


async def adaptive_semaphore(urls, base_concurrent=10):
    """
    自适应并发控制
    根据错误率动态调整并发数
    """
    semaphore = asyncio.Semaphore(base_concurrent)
    error_count = 0
    success_count = 0
    
    async def fetch_adaptive(session, url):
        nonlocal error_count, success_count
        async with semaphore:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        success_count += 1
                        if success_count % 20 == 0 and semaphore._value < 30:
                            semaphore._value += 1
                        return await response.json()
                    else:
                        error_count += 1
                        if semaphore._value > 5:
                            semaphore._value -= 1
                        return None
            except Exception:
                error_count += 1
                if semaphore._value > 5:
                    semaphore._value -= 1
                return None
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_adaptive(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"  统计: 成功={success_count}, 失败={error_count}, 最终并发限制={semaphore._value}")
        return results


async def benchmark_semaphore():
    """性能对比测试"""
    base_url = "http://localhost:8080"
    test_urls = [f"{base_url}/medium" for _ in range(100)]
    
    print("=" * 70)
    print("并发控制性能对比测试 (100个请求)")
    print("=" * 70)
    
    print("\n❌ 无限制并发:")
    start = time.time()
    results = await unlimited_concurrency(test_urls)
    unlimited_time = time.time() - start
    success = sum(1 for r in results if r and r.get("status") == "ok")
    print(f"  耗时: {unlimited_time:.2f}s, 成功率: {success}/{len(results)}")
    
    await asyncio.sleep(1)
    
    for limit in [5, 10, 20, 50]:
        print(f"\n✅ Semaphore 限制并发={limit}:")
        start = time.time()
        results = await semaphore_limited(test_urls, limit)
        elapsed = time.time() - start
        success = sum(1 for r in results if r and r.get("status") == "ok")
        print(f"  耗时: {elapsed:.2f}s, 成功率: {success}/{len(results)}")
        print(f"  单请求平均: {elapsed/len(results)*1000:.1f}ms")
    
    print("\n" + "=" * 70)
    print("📊 性能分析:")
    print("  - 并发数 = 目标服务器能处理的QPS")
    print("  - 过高并发会导致连接超时或拒绝服务")
    print("  - 生产环境建议从低并发开始,逐步上调")
    print("  - 自适应并发可根据服务器响应动态调整")
    print("=" * 70)


async def demo_error_endpoint():
    """演示自适应并发在错误率高的场景"""
    base_url = "http://localhost:8080"
    test_urls = [f"{base_url}/error" for _ in range(100)]
    
    print("\n" + "=" * 70)
    print("自适应并发控制演示 (30% 错误率的端点)")
    print("=" * 70)
    
    start = time.time()
    results = await adaptive_semaphore(test_urls, base_concurrent=20)
    elapsed = time.time() - start
    
    success = sum(1 for r in results if r and r.get("status") == "ok")
    print(f"总耗时: {elapsed:.2f}s, 成功率: {success}/{len(results)}")


async def bounded_semaphore_pattern():
    """
    封装好的带限制的并发模式
    生产环境可复用
    """
    class BoundedTaskExecutor:
        def __init__(self, max_concurrent=10):
            self.semaphore = asyncio.Semaphore(max_concurrent)
            self.max_concurrent = max_concurrent
        
        async def execute(self, coro):
            async with self.semaphore:
                return await coro
        
        async def map(self, coro_func, items):
            tasks = [self.execute(coro_func(item)) for item in items]
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    return BoundedTaskExecutor


if __name__ == "__main__":
    print("请先启动 simple_server.py:")
    print("  python simple_server.py")
    print()
    print("然后运行本脚本进行测试")
    
    try:
        asyncio.run(benchmark_semaphore())
        asyncio.run(demo_error_endpoint())
    except Exception as e:
        print(f"错误: {e}")
        print("请确保 simple_server.py 正在运行")
