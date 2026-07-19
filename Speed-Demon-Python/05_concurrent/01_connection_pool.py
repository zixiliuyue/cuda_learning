"""
01 - 正确使用连接池
对比正确和错误的连接池使用方式

关键点:
1. 不要为每个请求创建新的 ClientSession
2. 合理配置 limit 和 limit_per_host
3. 复用连接池提升性能
"""

import asyncio
import time
import aiohttp


async def bad_approach(urls):
    """
    错误方式: 每个请求创建新的 Session
    问题: 创建连接开销大,无法复用连接
    """
    results = []
    for url in urls:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                results.append(await response.json())
    return results


async def good_approach(urls):
    """
    正确方式: 复用同一个 Session
    优势: 连接复用,减少握手开销
    """
    async with aiohttp.ClientSession() as session:
        results = []
        for url in urls:
            async with session.get(url) as response:
                results.append(await response.json())
        return results


async def optimized_approach(urls, limit=50):
    """
    优化方式: 配置连接池参数 + 并发请求
    """
    connector = aiohttp.TCPConnector(
        limit=limit,
        limit_per_host=10,
        ttl_dns_cache=300,
        use_dns_cache=True
    )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for response in responses:
            if isinstance(response, Exception):
                results.append(None)
            else:
                results.append(await response.json())
                await response.release()
        
        return results


async def benchmark_connection_pool():
    """性能对比测试"""
    base_url = "http://localhost:8080"
    test_urls = [f"{base_url}/fast" for _ in range(50)]
    
    print("=" * 70)
    print("连接池性能对比测试 (50个请求)")
    print("=" * 70)
    
    start = time.time()
    await bad_approach(test_urls)
    bad_time = time.time() - start
    print(f"❌ 错误方式 (每个请求新建Session): {bad_time:.2f}s")
    
    await asyncio.sleep(1)
    
    start = time.time()
    await good_approach(test_urls)
    good_time = time.time() - start
    print(f"✅ 正确方式 (复用Session): {good_time:.2f}s, 加速比: {bad_time/good_time:.1f}x")
    
    await asyncio.sleep(1)
    
    start = time.time()
    await optimized_approach(test_urls, limit=20)
    opt_time = time.time() - start
    print(f"🚀 优化方式 (连接池+并发): {opt_time:.2f}s, 加速比: {bad_time/opt_time:.1f}x")
    
    print("=" * 70)
    print("\n📊 性能分析:")
    print("  - 复用 Session 可减少 TCP 握手开销")
    print("  - 合理配置 limit 可避免目标服务器过载")
    print("  - limit_per_host 防止对单一主机发起过多请求")
    print("=" * 70)


class ConnectionPoolManager:
    """
    连接池管理器 - 生产环境推荐
    支持全局复用连接池
    """
    
    def __init__(self, max_connections=100, max_per_host=20):
        self.max_connections = max_connections
        self.max_per_host = max_per_host
        self._session = None
        self._connector = None
    
    async def __aenter__(self):
        self._connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_per_host,
            ttl_dns_cache=300
        )
        self._session = aiohttp.ClientSession(connector=self._connector)
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()
    
    async def fetch(self, url):
        if not self._session:
            raise RuntimeError("Session not initialized")
        async with self._session.get(url) as response:
            return await response.json()
    
    async def fetch_many(self, urls):
        tasks = [self.fetch(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)


async def demo_pool_manager():
    """演示连接池管理器的使用"""
    base_url = "http://localhost:8080"
    urls = [f"{base_url}/fast" for _ in range(30)]
    
    print("\n" + "=" * 70)
    print("连接池管理器演示")
    print("=" * 70)
    
    async with ConnectionPoolManager(max_connections=20, max_per_host=10) as pool:
        start = time.time()
        results = await pool.fetch_many(urls)
        elapsed = time.time() - start
        
        success = sum(1 for r in results if r and r.get("status") == "ok")
        print(f"完成 {len(urls)} 个请求, 成功 {success} 个, 耗时: {elapsed:.2f}s")


if __name__ == "__main__":
    print("请先启动 simple_server.py:")
    print("  python simple_server.py")
    print()
    print("然后运行本脚本进行测试")
    
    try:
        asyncio.run(benchmark_connection_pool())
        asyncio.run(demo_pool_manager())
    except Exception as e:
        print(f"错误: {e}")
        print("请确保 simple_server.py 正在运行")
