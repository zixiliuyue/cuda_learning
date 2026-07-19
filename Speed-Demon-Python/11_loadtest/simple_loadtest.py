"""
简单的 HTTP 负载测试脚本
使用 requests + 线程池进行并发压测

功能:
    - 支持自定义并发数和请求数
    - 计算 TP50/TP90/TP95/TP99/TP999 分位响应时间
    - 计算 QPS 和成功率
    - 支持对比两个服务的性能
    - 支持保存结果到 JSON 文件

使用方法:
    python simple_loadtest.py --url http://localhost:5000 --concurrency 10 --requests 100
    python simple_loadtest.py --compare --url1 http://localhost:5000 --url2 http://localhost:5001
"""

import argparse
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import numpy as np


class LoadTestResult:
    """负载测试结果容器"""

    def __init__(self):
        self.response_times = []  # 响应时间列表（毫秒）
        self.status_codes = []    # 状态码列表
        self.errors = []          # 错误列表
        self.start_time = None
        self.end_time = None
        self.concurrency = 0
        self.total_requests = 0
        self.url = ""

    def add_response(self, response_time: float, status_code: int, error: str = None):
        """添加一个请求结果"""
        self.response_times.append(response_time)
        self.status_codes.append(status_code)
        if error:
            self.errors.append(error)

    @property
    def success_count(self):
        """成功请求数"""
        return sum(1 for code in self.status_codes if 200 <= code < 300)

    @property
    def failure_count(self):
        """失败请求数"""
        return len(self.status_codes) - self.success_count

    @property
    def success_rate(self):
        """成功率"""
        if not self.status_codes:
            return 0.0
        return self.success_count / len(self.status_codes) * 100

    @property
    def total_time(self):
        """总耗时（秒）"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def qps(self):
        """每秒请求数"""
        if self.total_time > 0:
            return len(self.status_codes) / self.total_time
        return 0.0

    @property
    def avg_response_time(self):
        """平均响应时间（毫秒）"""
        if self.response_times:
            return np.mean(self.response_times)
        return 0.0

    @property
    def min_response_time(self):
        """最小响应时间（毫秒）"""
        if self.response_times:
            return np.min(self.response_times)
        return 0.0

    @property
    def max_response_time(self):
        """最大响应时间（毫秒）"""
        if self.response_times:
            return np.max(self.response_times)
        return 0.0

    def get_percentile(self, percentile: float) -> float:
        """获取指定分位的响应时间（毫秒）"""
        if self.response_times:
            return np.percentile(self.response_times, percentile)
        return 0.0

    @property
    def tp50(self):
        return self.get_percentile(50)

    @property
    def tp90(self):
        return self.get_percentile(90)

    @property
    def tp95(self):
        return self.get_percentile(95)

    @property
    def tp99(self):
        return self.get_percentile(99)

    @property
    def tp999(self):
        return self.get_percentile(99.9)

    def to_dict(self):
        """转换为字典格式"""
        return {
            "url": self.url,
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "total_time": self.total_time,
            "qps": self.qps,
            "response_times": {
                "avg": self.avg_response_time,
                "min": self.min_response_time,
                "max": self.max_response_time,
                "tp50": self.tp50,
                "tp90": self.tp90,
                "tp95": self.tp95,
                "tp99": self.tp99,
                "tp999": self.tp999,
                "raw": self.response_times  # 原始数据用于绘图
            },
            "status_codes": dict(Counter(self.status_codes)),
            "errors": self.errors[:10]  # 只保留前10个错误
        }

    def to_json(self, indent: int = 2):
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def make_request(url: str, endpoint: str, timeout: int) -> tuple:
    """执行单个HTTP请求

    Returns:
        (响应时间毫秒, 状态码, 错误信息或None)
    """
    start_time = time.time()
    try:
        full_url = f"{url.rstrip('/')}/{endpoint.lstrip('/')}"
        response = requests.get(full_url, timeout=timeout)
        elapsed = (time.time() - start_time) * 1000  # 转换为毫秒
        return elapsed, response.status_code, None
    except requests.Timeout:
        elapsed = (time.time() - start_time) * 1000
        return elapsed, 408, "请求超时"
    except requests.ConnectionError:
        elapsed = (time.time() - start_time) * 1000
        return elapsed, 503, "连接错误"
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        return elapsed, 500, str(e)


def run_load_test(
    url: str,
    endpoint: str = "/product/1",
    concurrency: int = 10,
    total_requests: int = 100,
    timeout: int = 30
) -> LoadTestResult:
    """执行负载测试

    Args:
        url: 目标URL
        endpoint: 测试端点
        concurrency: 并发数
        total_requests: 总请求数
        timeout: 超时时间（秒）

    Returns:
        LoadTestResult 对象
    """
    result = LoadTestResult()
    result.url = url
    result.concurrency = concurrency
    result.total_requests = total_requests
    result.start_time = time.time()

    print(f"开始压测: {url}{endpoint}")
    print(f"并发数: {concurrency}, 总请求数: {total_requests}")
    print("-" * 60)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(make_request, url, endpoint, timeout)
            for _ in range(total_requests)
        ]

        completed = 0
        for future in as_completed(futures):
            elapsed, status_code, error = future.result()
            result.add_response(elapsed, status_code, error)
            completed += 1
            if completed % 10 == 0:
                print(f"已完成: {completed}/{total_requests} ({completed/total_requests*100:.1f}%)")

    result.end_time = time.time()
    return result


def print_result(result: LoadTestResult, title: str = "负载测试结果"):
    """打印测试结果"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print("\n基本信息:")
    print(f"  目标URL:             {result.url}")
    print(f"  并发用户数:          {result.concurrency}")
    print(f"  总请求数:            {result.total_requests}")
    print(f"  成功请求数:          {result.success_count}")
    print(f"  失败请求数:          {result.failure_count}")
    print(f"  成功率:              {result.success_rate:.2f}%")

    print("\n时间统计:")
    print(f"  总耗时:              {result.total_time:.2f}s")
    print(f"  QPS:                 {result.qps:.2f}")
    print(f"  平均响应时间:        {result.avg_response_time:.2f}ms")
    print(f"  最小响应时间:        {result.min_response_time:.2f}ms")
    print(f"  最大响应时间:        {result.max_response_time:.2f}ms")

    print("\n分位数响应时间:")
    print(f"  TP50:                {result.tp50:.2f}ms")
    print(f"  TP90:                {result.tp90:.2f}ms")
    print(f"  TP95:                {result.tp95:.2f}ms")
    print(f"  TP99:                {result.tp99:.2f}ms")
    print(f"  TP999:               {result.tp999:.2f}ms")

    print("\n状态码统计:")
    for code, count in sorted(Counter(result.status_codes).items()):
        print(f"  {code}:                 {count}")

    if result.errors:
        print("\n错误统计 (前10个):")
        for i, error in enumerate(result.errors[:10], 1):
            print(f"  {i}. {error}")

    print("=" * 60 + "\n")


def compare_results(result1: LoadTestResult, result2: LoadTestResult, name1: str = "v1", name2: str = "v2"):
    """对比两个测试结果"""
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)

    metrics = [
        ("QPS", result1.qps, result2.qps, True),   # 越高越好
        ("平均响应时间(ms)", result1.avg_response_time, result2.avg_response_time, False),
        ("TP50(ms)", result1.tp50, result2.tp50, False),
        ("TP90(ms)", result1.tp90, result2.tp90, False),
        ("TP95(ms)", result1.tp95, result2.tp95, False),
        ("TP99(ms)", result1.tp99, result2.tp99, False),
        ("成功率(%)", result1.success_rate, result2.success_rate, True),
    ]

    print(f"\n{'指标':<20} {name1:<15} {name2:<15} {'差异':<15}")
    print("-" * 70)

    for name, val1, val2, higher_is_better in metrics:
        diff = val2 - val1
        if higher_is_better:
            diff_pct = (diff / val1 * 100) if val1 > 0 else 0
            indicator = "↑" if diff > 0 else "↓" if diff < 0 else "="
        else:
            diff_pct = (-diff / val1 * 100) if val1 > 0 else 0
            indicator = "↓" if diff < 0 else "↑" if diff > 0 else "="

        print(f"{name:<20} {val1:<15.2f} {val2:<15.2f} {indicator} {abs(diff_pct):.1f}%")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="简单的 HTTP 负载测试工具")
    parser.add_argument("--url", default="http://localhost:5000", help="目标URL")
    parser.add_argument("--endpoint", default="/product/1", help="测试端点")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    parser.add_argument("--requests", type=int, default=100, help="总请求数")
    parser.add_argument("--timeout", type=int, default=30, help="超时时间（秒）")
    parser.add_argument("--output", help="输出JSON结果文件")
    parser.add_argument("--compare", action="store_true", help="对比两个服务的性能")
    parser.add_argument("--url1", default="http://localhost:5000", help="第一个服务URL（对比模式）")
    parser.add_argument("--url2", default="http://localhost:5001", help="第二个服务URL（对比模式）")

    args = parser.parse_args()

    if args.compare:
        # 对比模式
        print("=" * 60)
        print("对比模式: 测试两个服务的性能差异")
        print("=" * 60)

        print("\n[1/2] 测试第一个服务...")
        result1 = run_load_test(
            url=args.url1,
            endpoint=args.endpoint,
            concurrency=args.concurrency,
            total_requests=args.requests,
            timeout=args.timeout
        )
        print_result(result1, f"服务1测试结果: {args.url1}")

        print("\n[2/2] 测试第二个服务...")
        result2 = run_load_test(
            url=args.url2,
            endpoint=args.endpoint,
            concurrency=args.concurrency,
            total_requests=args.requests,
            timeout=args.timeout
        )
        print_result(result2, f"服务2测试结果: {args.url2}")

        compare_results(result1, result2, name1="v1_bad", name2="v2_good")

        if args.output:
            output_data = {
                "service1": result1.to_dict(),
                "service2": result2.to_dict()
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"对比结果已保存到: {args.output}")
    else:
        # 单服务模式
        result = run_load_test(
            url=args.url,
            endpoint=args.endpoint,
            concurrency=args.concurrency,
            total_requests=args.requests,
            timeout=args.timeout
        )
        print_result(result)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result.to_json())
            print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
