import time
import random
from typing import Callable
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

app = FastAPI(title="Python可观测性示例服务")

REQUEST_COUNT = Counter(
    'http_requests_total',
    'HTTP请求总数',
    ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP请求延迟（秒）',
    ['method', 'endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10]
)

REQUESTS_IN_PROGRESS = Gauge(
    'http_requests_in_progress',
    '正在处理的HTTP请求数',
    ['method', 'endpoint']
)

ERROR_COUNT = Counter(
    'http_errors_total',
    'HTTP错误总数',
    ['method', 'endpoint', 'error_type']
)

USER_REGISTRATIONS = Counter(
    'user_registrations_total',
    '用户注册总数',
    ['region']
)

ORDER_PROCESSING_DURATION = Histogram(
    'order_processing_duration_seconds',
    '订单处理耗时（秒）',
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

ACTIVE_USERS = Gauge(
    'active_users',
    '当前活跃用户数'
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    method = request.method
    endpoint = request.url.path

    if endpoint == "/metrics":
        return await call_next(request)

    REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()

    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        status_code = response.status_code

        duration = time.perf_counter() - start_time
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=str(status_code)).inc()

        if status_code >= 400:
            ERROR_COUNT.labels(
                method=method,
                endpoint=endpoint,
                error_type=f"http_{status_code}"
            ).inc()

        return response

    except Exception as e:
        duration = time.perf_counter() - start_time
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status="500").inc()
        ERROR_COUNT.labels(
            method=method,
            endpoint=endpoint,
            error_type=type(e).__name__
        ).inc()
        raise

    finally:
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/")
async def root():
    return {"message": "Python可观测性示例服务", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    sleep_time = random.uniform(0.01, 0.1)
    time.sleep(sleep_time)

    return {
        "user_id": user_id,
        "name": f"用户{user_id}",
        "email": f"user{user_id}@example.com"
    }

@app.post("/api/users")
async def create_user(region: str = "cn"):
    sleep_time = random.uniform(0.05, 0.2)
    time.sleep(sleep_time)

    USER_REGISTRATIONS.labels(region=region).inc()
    ACTIVE_USERS.inc()

    return {"status": "success", "message": "用户创建成功"}

@app.get("/api/orders")
async def list_orders():
    sleep_time = random.uniform(0.02, 0.15)
    time.sleep(sleep_time)

    orders = [
        {"order_id": i, "status": "completed" if i % 2 == 0 else "pending"}
        for i in range(1, 6)
    ]
    return {"orders": orders}

@app.post("/api/orders")
async def create_order():
    with ORDER_PROCESSING_DURATION.time():
        sleep_time = random.uniform(0.1, 1.0)
        time.sleep(sleep_time)

        if random.random() < 0.1:
            raise ValueError("订单处理失败")

        return {"order_id": random.randint(1000, 9999), "status": "created"}

@app.get("/api/slow")
async def slow_endpoint():
    sleep_time = random.uniform(0.5, 2.0)
    time.sleep(sleep_time)
    return {"message": "这是一个慢响应接口"}

@app.get("/api/error")
async def error_endpoint():
    if random.random() < 0.5:
        raise ValueError("模拟的内部错误")
    return {"message": "你很幸运，没有出错"}

if __name__ == "__main__":
    import uvicorn
    print("启动带Prometheus指标的FastAPI服务...")
    print("访问 http://localhost:8000/docs 查看API文档")
    print("访问 http://localhost:8000/metrics 查看指标")
    uvicorn.run(app, host="0.0.0.0", port=8000)
