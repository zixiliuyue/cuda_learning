#!/usr/bin/env python3
"""
电商商品详情接口压测脚本

使用方法:
    python loadtest.py --url http://localhost:5000/product/1 --requests 100 --concurrency 10
"""

import argparse
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, median, stdev
import threading

class LoadTestResult:
    """压测结果统计"""
    def __init__(self):
        self.lock = threading.Lock()
        self.success_count = 0
        self.error_count = 0
        self.response_times = []
        self.execution_times = []
    
    def add_success(self, response_time, execution_time=None):
        with self.lock:
            self.success_count += 1
            self.response_times.append(response_time)
            if execution_time:
                self.execution_times.append(execution_time)
    
    def add_error(self):
        with self.lock:
            self.error_count += 1
    
    def get_stats(self):
        with self.lock:
            if not self.response_times:
                return {}
            
            sorted_times = sorted(self.response_times)
            return {
                "total_requests": self.success_count + self.error_count,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "success_rate": self.success_count / (self.success_count + self.error_count) * 100,
                "min_response_time": min(self.response_times),
                "max_response_time": max(self.response_times),
                "avg_response_time": mean(self.response_times),
                "median_response_time": median(self.response_times),
                "p95_response_time": sorted_times[int(len(sorted_times) * 0.95)],
                "p99_response_time": sorted_times[int(len(sorted_times) * 0.99)],
                "std_dev": stdev(self.response_times) if len(self.response_times) > 1 else 0,
                "avg_server_execution_time": mean(self.execution_times) if self.execution_times else 0
            }

def make_request(url, timeout=10):
    """发起单个请求"""
    start_time = time.time()
    
    try:
        response = requests.get(url, timeout=timeout)
        response_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        if response.status_code == 200:
            data = response.json()
            execution_time = data.get("execution_time", 0) * 1000
            return True, response_time, execution_time
        else:
            return False, response_time, None
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return False, response_time, None

def run_load_test(url, total_requests, concurrency, timeout):
    """运行压测"""
    result = LoadTestResult()
    
    print(f"\n{'=' * 70}")
    print(f"开始压测")
    print(f"{'=' * 70}")
    print(f"目标URL: {url}")
    print(f"总请求数: {total_requests}")
    print(f"并发数: {concurrency}")
    print(f"超时时间: {timeout}秒")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(make_request, url, timeout)
            for _ in range(total_requests)
        ]
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            success, response_time, execution_time = future.result()
            
            if success:
                result.add_success(response_time, execution_time)
            else:
                result.add_error()
            
            # 显示进度
            if completed % 10 == 0 or completed == total_requests:
                progress = completed / total_requests * 100
                print(f"\r进度: {completed}/{total_requests} ({progress:.1f}%)", end="", flush=True)
    
    total_time = (time.time() - start_time) * 1000
    stats = result.get_stats()
    
    print(f"\n\n{'=' * 70}")
    print(f"压测结果")
    print(f"{'=' * 70}")
    
    print(f"\n总耗时: {total_time:.2f} ms")
    print(f"QPS: {stats['total_requests'] / (total_time / 1000):.2f} req/s")
    
    print(f"\n请求统计:")
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  成功: {stats['success_count']}")
    print(f"  失败: {stats['error_count']}")
    print(f"  成功率: {stats['success_rate']:.2f}%")
    
    print(f"\n响应时间统计 (ms):")
    print(f"  最小: {stats['min_response_time']:.2f}")
    print(f"  最大: {stats['max_response_time']:.2f}")
    print(f"  平均: {stats['avg_response_time']:.2f}")
    print(f"  中位数: {stats['median_response_time']:.2f}")
    print(f"  P95: {stats['p95_response_time']:.2f}")
    print(f"  P99: {stats['p99_response_time']:.2f}")
    print(f"  标准差: {stats['std_dev']:.2f}")
    
    if stats['avg_server_execution_time'] > 0:
        print(f"\n服务端执行时间:")
        print(f"  平均: {stats['avg_server_execution_time']:.2f} ms")
    
    print(f"{'=' * 70}\n")
    
    return stats

def compare_versions():
    """对比两个版本的性能"""
    print("\n" + "=" * 70)
    print("自动对比 v1_bad vs v2_good")
    print("=" * 70)
    print("\n请确保两个服务都已启动:")
    print("  v1_bad: python v1_bad/app.py (端口 5000)")
    print("  v2_good: python v2_good/app.py (端口 5001)")
    print("")
    
    input("按 Enter 继续...")
    
    # 先清空v2_good的缓存
    try:
        requests.get("http://localhost:5001/cache/clear", timeout=5)
        print("\n已清空 v2_good 缓存\n")
    except:
        print("\n警告: 无法连接到 v2_good 服务清空缓存\n")
    
    # 压测配置
    test_requests = 50
    test_concurrency = 10
    
    print(f"压测配置: {test_requests} 请求, {test_concurrency} 并发\n")
    
    # 压测 v1_bad
    print("正在压测 v1_bad (http://localhost:5000/product/1)...")
    stats_v1 = run_load_test(
        "http://localhost:5000/product/1",
        total_requests=test_requests,
        concurrency=test_concurrency,
        timeout=30
    )
    
    # 压测 v2_good (第一次 - 无缓存)
    print("正在压测 v2_good (无缓存, http://localhost:5001/product/1)...")
    stats_v2_nocache = run_load_test(
        "http://localhost:5001/product/1",
        total_requests=test_requests,
        concurrency=test_concurrency,
        timeout=30
    )
    
    # 压测 v2_good (第二次 - 有缓存)
    print("正在压测 v2_good (有缓存, http://localhost:5001/product/1)...")
    stats_v2_cache = run_load_test(
        "http://localhost:5001/product/1",
        total_requests=test_requests,
        concurrency=test_concurrency,
        timeout=30
    )
    
    # 对比结果
    print("=" * 70)
    print("性能对比总结")
    print("=" * 70)
    
    print(f"\n{'指标':<25} {'v1_bad':<15} {'v2_good(无缓存)':<15} {'v2_good(有缓存)':<15}")
    print("-" * 70)
    
    qps_v1 = stats_v1['total_requests'] / (stats_v1['avg_response_time'] / 1000)
    qps_v2_nocache = stats_v2_nocache['total_requests'] / (stats_v2_nocache['avg_response_time'] / 1000)
    qps_v2_cache = stats_v2_cache['total_requests'] / (stats_v2_cache['avg_response_time'] / 1000)
    
    print(f"{'QPS (估算)':<25} {qps_v1:<15.2f} {qps_v2_nocache:<15.2f} {qps_v2_cache:<15.2f}")
    print(f"{'平均响应时间(ms)':<25} {stats_v1['avg_response_time']:<15.2f} {stats_v2_nocache['avg_response_time']:<15.2f} {stats_v2_cache['avg_response_time']:<15.2f}")
    print(f"{'P95响应时间(ms)':<25} {stats_v1['p95_response_time']:<15.2f} {stats_v2_nocache['p95_response_time']:<15.2f} {stats_v2_cache['p95_response_time']:<15.2f}")
    print(f"{'成功率(%)':<25} {stats_v1['success_rate']:<15.2f} {stats_v2_nocache['success_rate']:<15.2f} {stats_v2_cache['success_rate']:<15.2f}")
    
    # 计算提升百分比
    improvement_nocache = (stats_v1['avg_response_time'] - stats_v2_nocache['avg_response_time']) / stats_v1['avg_response_time'] * 100
    improvement_cache = (stats_v1['avg_response_time'] - stats_v2_cache['avg_response_time']) / stats_v1['avg_response_time'] * 100
    
    print(f"\n性能提升:")
    print(f"  v2_good(无缓存) 比 v1_bad 快: {improvement_nocache:.1f}%")
    print(f"  v2_good(有缓存) 比 v1_bad 快: {improvement_cache:.1f}%")
    
    print("=" * 70)
    print("\n优化效果说明:")
    print("  1. 无缓存情况下: 并行查询 + 连接池 + 高效数据构建带来提升")
    print("  2. 有缓存情况下: 缓存命中避免了所有数据库和HTTP调用")
    print("  3. 内存泄漏: v1_bad内存持续增长，v2_good有界缓存安全")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="电商商品详情接口压测脚本")
    parser.add_argument("--url", type=str, help="压测目标URL")
    parser.add_argument("--requests", type=int, default=100, help="总请求数 (默认: 100)")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数 (默认: 10)")
    parser.add_argument("--timeout", type=int, default=10, help="超时时间(秒) (默认: 10)")
    parser.add_argument("--compare", action="store_true", help="自动对比两个版本性能")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_versions()
    elif args.url:
        run_load_test(args.url, args.requests, args.concurrency, args.timeout)
    else:
        print("请指定 --url 或使用 --compare 进行自动对比")
        print("\n使用示例:")
        print("  压测单个接口: python loadtest.py --url http://localhost:5000/product/1 --requests 100 --concurrency 10")
        print("  自动对比版本: python loadtest.py --compare")
        print()

if __name__ == "__main__":
    main()
