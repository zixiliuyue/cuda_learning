"""
Locust 负载测试脚本
针对 05_full_case 的 Flask 服务进行压测

使用方法:
    1. Web界面: locust -f locustfile.py --host=http://localhost:5000
    2. 无头模式: locust -f locustfile.py --host=http://localhost:5000 --headless -u 10 -r 2 --run-time 30s
"""

from locust import HttpUser, TaskSet, task, between, events
import random
import time


class UserBehavior(TaskSet):
    """用户行为模型"""

    def on_start(self):
        """每个用户开始时执行"""
        self.product_ids = [1, 2, 3, 4, 5]

    @task(7)
    def get_product_detail(self):
        """商品详情查询 - 主要业务场景，权重最高"""
        product_id = random.choice(self.product_ids)
        with self.client.get(
            f"/product/{product_id}",
            catch_response=True,
            name="/product/{id}"
        ) as response:
            if response.status_code != 200:
                response.failure(f"状态码错误: {response.status_code}")
            else:
                try:
                    data = response.json()
                    if "product" not in data:
                        response.failure("响应格式错误: 缺少 product 字段")
                except Exception as e:
                    response.failure(f"JSON解析失败: {str(e)}")

    @task(2)
    def health_check(self):
        """健康检查 - 轻量级请求"""
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code != 200:
                response.failure(f"状态码错误: {response.status_code}")

    @task(1)
    def clear_cache(self):
        """清空缓存 - 仅v2_good版本支持"""
        # 只有一定概率执行清空，避免缓存完全失效
        if random.random() < 0.1:
            with self.client.get(
                "/cache/clear",
                catch_response=True,
                name="/cache/clear"
            ) as response:
                if response.status_code not in [200, 404]:
                    response.failure(f"状态码错误: {response.status_code}")


class WebsiteUser(HttpUser):
    """网站用户模型"""

    tasks = [UserBehavior]

    # 用户思考时间范围（秒）
    wait_time = between(0.1, 0.5)

    # 网络超时设置
    network_timeout = 60.0
    connection_timeout = 60.0


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Locust 初始化事件"""
    print("=" * 60)
    print("Locust 负载测试启动")
    print("=" * 60)
    print(f"目标主机: {environment.host}")
    print(f"测试场景:")
    print("  - 商品详情查询 (70%权重)")
    print("  - 健康检查 (20%权重)")
    print("  - 缓存操作 (10%权重)")
    print("=" * 60)


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """测试开始事件"""
    print("\n" + "=" * 60)
    print(f"测试开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if hasattr(environment, 'parsed_options'):
        opts = environment.parsed_options
        if hasattr(opts, 'num_users'):
            print(f"预期用户数: {opts.num_users}")
        if hasattr(opts, 'spawn_rate'):
            print(f"启动速率: {opts.spawn_rate} 用户/秒")
        if hasattr(opts, 'run_time') and opts.run_time:
            print(f"运行时长: {opts.run_time}")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """测试结束事件"""
    print("\n" + "=" * 60)
    print(f"测试结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """请求完成事件 - 可用于自定义统计"""
    # 这里可以添加自定义的统计逻辑
    pass


if __name__ == "__main__":
    print("请使用 locust 命令运行此脚本:")
    print("  locust -f locustfile.py --host=http://localhost:5000")
    print("  访问 Web 界面: http://localhost:8089")
