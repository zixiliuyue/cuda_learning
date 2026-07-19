"""
03 - 任务批量处理
处理大量任务时,分批处理避免内存和性能问题

关键点:
1. 大量任务同时创建会占用过多内存
2. 分批处理可以及时释放资源
3. 结合 Semaphore 实现更精细的控制
"""

import asyncio
import time
import aiohttp
from typing import List, Any


async def process_all_at_once(items):
    """
    一次性处理所有任务
    问题: 任务量大时内存占用高,创建任务耗时
    """
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed = []
        for result in results:
            if not isinstance(result, Exception):
                processed.append(await result.json())
                await result.release()
            else:
                processed.append(None)
        return processed


async def process_in_batches(items, batch_size=20):
    """
    分批处理
    优势: 内存占用稳定,可以实时处理中间结果
    """
    all_results = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            tasks = [session.get(url) for url in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed = []
            for result in batch_results:
                if not isinstance(result, Exception):
                    processed.append(await result.json())
                    await result.release()
                else:
                    processed.append(None)
            
            all_results.extend(processed)
            print(f"  已处理 {min(i + batch_size, len(items))}/{len(items)} 项")
    
    return all_results


async def process_with_bounded_semaphore(items, max_concurrent=20):
    """
    使用 Semaphore 实现流式处理
    优势: 比分批更平滑,不会有波峰波谷
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_limit(session, url):
        async with semaphore:
            async with session.get(url) as response:
                return await response.json()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_limit(session, url) for url in items]
        return await asyncio.gather(*tasks, return_exceptions=True)


async def process_with_progress(items, batch_size=20):
    """
    带进度显示的分批处理
    生产环境中可以集成到日志系统
    """
    all_results = []
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(items), batch_size):
            batch_start = time.time()
            batch = items[i:i + batch_size]
            
            tasks = [session.get(url) for url in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed = []
            for result in batch_results:
                if not isinstance(result, Exception):
                    processed.append(await result.json())
                    await result.release()
                else:
                    processed.append(None)
            
            all_results.extend(processed)
            
            elapsed = time.time() - batch_start
            total_elapsed = time.time() - start_time
            progress = (i + len(batch)) / len(items) * 100
            eta = total_elapsed / (i + len(batch)) * (len(items) - i - len(batch)) if i > 0 else 0
            
            print(f"  进度: {progress:.1f}% | 本批: {elapsed:.2f}s | 预计剩余: {eta:.1f}s")
    
    return all_results


class BatchProcessor:
    """
    批量处理器 - 生产环境推荐
    支持: 自定义批大小, 回调, 进度跟踪
    """
    
    def __init__(self, batch_size=20, max_retries=3, callback=None):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.callback = callback
        self.results = []
        self.errors = []
    
    async def process_item(self, session, item, retry_count=0):
        """处理单个项,支持重试"""
        try:
            async with session.get(item) as response:
                result = await response.json()
                if self.callback:
                    self.callback(result)
                return result
        except Exception as e:
            if retry_count < self.max_retries:
                await asyncio.sleep(0.1 * (retry_count + 1))
                return await self.process_item(session, item, retry_count + 1)
            else:
                self.errors.append((item, str(e)))
                return None
    
    async def process_batch(self, session, batch):
        """处理一批"""
        tasks = [self.process_item(session, item) for item in batch]
        return await asyncio.gather(*tasks)
    
    async def process_all(self, items):
        """处理所有项"""
        total = len(items)
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, total, self.batch_size):
                batch = items[i:i + self.batch_size]
                batch_results = await self.process_batch(session, batch)
                self.results.extend(batch_results)
                
                completed = min(i + self.batch_size, total)
                print(f"  进度: {completed}/{total} ({completed/total*100:.1f}%)")
        
        return self.results


async def benchmark_batch_processing():
    """性能对比测试"""
    base_url = "http://localhost:8080"
    test_urls = [f"{base_url}/fast" for _ in range(200)]
    
    print("=" * 70)
    print("批量处理性能对比测试 (200个请求)")
    print("=" * 70)
    
    print("\n❌ 一次性处理:")
    start = time.time()
    await process_all_at_once(test_urls)
    all_time = time.time() - start
    print(f"  耗时: {all_time:.2f}s")
    
    await asyncio.sleep(1)
    
    for batch_size in [10, 20, 50]:
        print(f"\n✅ 分批处理 (batch_size={batch_size}):")
        start = time.time()
        await process_in_batches(test_urls, batch_size)
        batch_time = time.time() - start
        print(f"  耗时: {batch_time:.2f}s")
    
    print("\n✅ 信号量流式处理 (max_concurrent=20):")
    start = time.time()
    await process_with_bounded_semaphore(test_urls, 20)
    sem_time = time.time() - start
    print(f"  耗时: {sem_time:.2f}s")
    
    print("\n" + "=" * 70)
    print("📊 性能分析:")
    print("  - 一次性处理内存峰值高")
    print("  - 分批处理更稳定,适合任务量巨大的场景")
    print("  - 信号量方式吞吐量更稳定平滑")
    print("  - 批大小需要根据实际场景调整")
    print("=" * 70)


async def demo_batch_processor():
    """演示批量处理器"""
    base_url = "http://localhost:8080"
    test_urls = [f"{base_url}/variable" for _ in range(100)]
    
    print("\n" + "=" * 70)
    print("批量处理器演示 (带进度和重试)")
    print("=" * 70)
    
    processor = BatchProcessor(batch_size=15, max_retries=2)
    
    start = time.time()
    results = await processor.process_all(test_urls)
    elapsed = time.time() - start
    
    success = sum(1 for r in results if r and r.get("status") == "ok")
    print(f"\n完成! 总耗时: {elapsed:.2f}s")
    print(f"成功: {success}/{len(results)}, 错误: {len(processor.errors)}")


if __name__ == "__main__":
    print("请先启动 simple_server.py:")
    print("  python simple_server.py")
    print()
    print("然后运行本脚本进行测试")
    
    try:
        asyncio.run(benchmark_batch_processing())
        asyncio.run(demo_batch_processor())
    except Exception as e:
        print(f"错误: {e}")
        print("请确保 simple_server.py 正在运行")
