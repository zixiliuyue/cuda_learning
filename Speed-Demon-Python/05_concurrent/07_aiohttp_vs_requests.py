"""
07 - aiohttp vs requests 性能对比
完整的 Web 请求性能测试

关键点:
1. requests 是同步阻塞的,在异步代码中会阻塞事件循环
2. aiohttp 真正异步,可以充分利用并发
3. 并发数越大,性能差异越明显
"""

import asyncio
import time
import requests
import aiohttp


def benchmark_sync_requests(url, count=50):
    """
    同步方式: requests.get()
    完全串行执行
    """
    print(f"\n同步 requests: {count} 个请求...")
    start = time.time()
    
    success = 0
    for i in range(count):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                success += 1
        except Exception as e:
            print(f"  请求 {i} 失败: {e}")
    
    elapsed = time.time() - start
    print(f"  完成: {success}/{count}, 耗时: {elapsed:.2f}s")
    print(f"  平均: {elapsed/count*1000:.1f}ms/请求, 吞吐量: {count/elapsed:.1f} req/s")
    return elapsed


def benchmark_sync_session(url, count=50):
    """
    同步但复用 Session
    比每次新建请求快,但仍然是串行
    """
    print(f"\n同步 Session 复用: {count} 个请求...")
    start = time.time()
    
    success = 0
    with requests.Session() as session:
        for i in range(count):
            try:
                response = session.get(url, timeout=10)
                if response.status_code == 200:
                    success += 1
            except Exception as e:
                print(f"  请求 {i} 失败: {e}")
    
    elapsed = time.time() - start
    print(f"  完成: {success}/{count}, 耗时: {elapsed:.2f}s")
    print(f"  平均: {elapsed/count*1000:.1f}ms/请求, 吞吐量: {count/elapsed:.1f} req/s")
    return elapsed


async def benchmark_aiohttp(url, count=50, concurrency=20):
    """
    异步 aiohttp
    真正的并发请求
    """
    print(f"\n异步 aiohttp: {count} 个请求,并发={concurrency}...")
    start = time.time()
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def fetch(session):
        async with semaphore:
            async with session.get(url) as response:
                return response.status == 200
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session) for _ in range(count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    success = sum(1 for r in results if r is True)
    elapsed = time.time() - start
    
    print(f"  完成: {success}/{count}, 耗时: {elapsed:.2f}s")
    print(f"  平均: {elapsed/count*1000:.1f}ms/请求, 吞吐量: {count/elapsed:.1f} req/s")
    return elapsed


async def benchmark_aiohttp_no_limit(url, count=50):
    """
    无限制并发
    可能压垮服务器或触发限流
    """
    print(f"\n异步 aiohttp 无并发限制: {count} 个请求...")
    start = time.time()
    
    async def fetch(session):
        async with session.get(url) as response:
            return response.status == 200
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session) for _ in range(count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    success = sum(1 for r in results if r is True)
    elapsed = time.time() - start
    
    print(f"  完成: {success}/{count}, 耗时: {elapsed:.2f}s")
    print(f"  平均: {elapsed/count*1000:.1f}ms/请求, 吞吐量: {count/elapsed:.1f} req/s")
    return elapsed


async def benchmark_concurrency_scaling(url, base_count=100):
    """
    不同并发数下的性能对比
    """
    print("\n" + "=" * 70)
    print("并发数 vs 性能 对比")
    print("=" * 70)
    
    concurrency_levels = [5, 10, 20, 50, 100]
    results = {}
    
    for concurrency in concurrency_levels:
        elapsed = await benchmark_aiohttp(url, base_count, concurrency)
        results[concurrency] = elapsed
        await asyncio.sleep(1)
    
    print("\n" + "-" * 70)
    print("总结:")
    for concurrency in concurrency_levels:
        elapsed = results[concurrency]
        speedup = results[concurrency_levels[0]] / elapsed
        print(f"  并发 {concurrency:3d}: {elapsed:6.2f}s (加速比 {speedup:.2f}x)")
    
    optimal = min(results.items(), key=lambda x: x[1] / x[0])
    print(f"\n最佳并发效率: {optimal[0]} (单位并发耗时: {optimal[1]/optimal[0]:.3f}s)")


def benchmark_threadpool_requests(url, count=50, max_workers=20):
    """
    线程池 + requests
    这是人们常用的"变通"方案,但不如真正的异步高效
    """
    import concurrent.futures
    
    print(f"\n线程池 + requests: {count} 个请求,线程数={max_workers}...")
    start = time.time()
    
    def fetch():
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch) for _ in range(count)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    success = sum(1 for r in results if r is True)
    elapsed = time.time() - start
    
    print(f"  完成: {success}/{count}, 耗时: {elapsed:.2f}s")
    print(f"  平均: {elapsed/count*1000:.1f}ms/请求, 吞吐量: {count/elapsed:.1f} req/s")
    return elapsed


async def full_comparison(base_url, request_count=100):
    """
    完整的性能对比
    """
    print("=" * 70)
    print(f"完整性能对比: {request_count} 个请求")
    print("=" * 70)
    
    results = {}
    
    # 1. 纯同步
    results['sync_requests'] = benchmark_sync_requests(base_url, request_count // 5)
    
    # 2. 同步+Session
    results['sync_session'] = benchmark_sync_session(base_url, request_count // 5)
    
    await asyncio.sleep(1)
    
    # 3. 线程池+requests
    results['threadpool_requests'] = benchmark_threadpool_requests(
        base_url, request_count, max_workers=20
    )
    
    await asyncio.sleep(1)
    
    # 4. 异步+aiohttp (有限并发)
    results['aiohttp_limited'] = await benchmark_aiohttp(
        base_url, request_count, concurrency=20
    )
    
    await asyncio.sleep(1)
    
    # 5. 异步+aiohttp (无限并发)
    results['aiohttp_unlimited'] = await benchmark_aiohttp_no_limit(
        base_url, request_count
    )
    
    print("\n" + "=" * 70)
    print("性能总结 (按耗时排序):")
    print("=" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    baseline = results['sync_requests']
    
    for name, elapsed in sorted_results:
        speedup = baseline / elapsed
        print(f"  {name:25s}: {elapsed:6.2f}s (加速比 {speedup:5.2f}x)")
    
    print("\n" + "=" * 70)
    print("关键结论:")
    print("=" * 70)
    print(f"  1. 异步比纯同步快 {results['sync_requests']/results['aiohttp_limited']:.1f}x")
    print(f"  2. Session复用可提升同步性能 {results['sync_requests']/results['sync_session']:.1f}x")
    print(f"  3. 线程池比纯同步快 {results['sync_requests']/results['threadpool_requests']:.1f}x")
    print(f"  4. 纯异步比线程池快 {results['threadpool_requests']/results['aiohttp_limited']:.1f}x")
    print("=" * 70)


def when_to_use_which():
    """
    使用场景建议
    """
    print("\n" + "=" * 70)
    print("使用场景建议")
    print("=" * 70)
    
    advice = """
选择正确的 HTTP 客户端:

1. 简单脚本 (< 10 个请求)
   → 用 requests,简单够用

2. 批量爬取/API 调用 (> 10 个请求)
   → 用 aiohttp + Semaphore,性能最好

3. 已有同步代码,不想大改
   → 用 ThreadPoolExecutor + requests,过渡方案

4. 需要上传大文件/流式响应
   → 用 aiohttp,支持异步流式处理

5. 需要连接池、Cookie、重试等高级特性
   → 两者都有,但 aiohttp 性能更好

性能差异来源:
- requests: 每个请求阻塞等待,CPU空闲
- aiohttp: 等待期间可以发起其他请求,CPU利用率高
- 线程池: 有线程创建和切换开销,不如原生异步高效
"""
    print(advice)


if __name__ == "__main__":
    print("=" * 70)
    print("请先启动 simple_server.py")
    print("=" * 70)
    
    base_url = "http://localhost:8080/fast"
    
    try:
        # 先测试服务器是否可用
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✓ 测试服务器可用")
            
            # 运行完整对比
            asyncio.run(full_comparison(base_url, request_count=100))
            
            # 运行并发伸缩测试
            asyncio.run(benchmark_concurrency_scaling(base_url, base_count=100))
            
            # 给出建议
            when_to_use_which()
        else:
            print("✗ 服务器响应异常")
    except Exception as e:
        print(f"✗ 无法连接到服务器: {e}")
        print("\n请先在另一个终端运行:")
        print("  python simple_server.py")
