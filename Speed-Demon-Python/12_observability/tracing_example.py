import time
import random
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from fastapi import FastAPI, Request
import httpx

def init_tracer(service_name: str = "python-tracing-service"):
    resource = Resource(attributes={
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": "development"
    })

    provider = TracerProvider(resource=resource)

    console_exporter = ConsoleSpanExporter()
    processor = BatchSpanProcessor(console_exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    return trace.get_tracer(__name__)

tracer = init_tracer()

app = FastAPI(title="分布式追踪示例服务")

def simulate_user_authentication(user_id: str) -> bool:
    with tracer.start_as_current_span("user_authentication") as span:
        span.set_attribute("user.id", user_id)
        span.add_event("开始验证用户凭证")

        time.sleep(random.uniform(0.01, 0.05))

        if random.random() < 0.1:
            span.set_status(Status(StatusCode.ERROR, "认证失败"))
            span.add_event("认证失败", attributes={"reason": "无效凭证"})
            return False

        span.add_event("认证成功", attributes={"auth_level": "full"})
        return True

def get_user_data_from_db(user_id: str) -> dict:
    with tracer.start_as_current_span("database_query") as span:
        span.set_attribute("db.system", "postgresql")
        span.set_attribute("db.user", "app_user")
        span.set_attribute("db.name", "users_db")
        span.set_attribute("db.statement", f"SELECT * FROM users WHERE id = '{user_id}'")

        time.sleep(random.uniform(0.05, 0.15))

        result = {
            "user_id": user_id,
            "name": f"用户{user_id}",
            "email": f"user{user_id}@example.com",
            "orders": random.randint(1, 50)
        }

        span.set_attribute("db.result.rows", 1)
        return result

def call_external_api(user_id: str) -> dict:
    with tracer.start_as_current_span("external_api_call") as span:
        span.set_attribute("http.method", "GET")
        span.set_attribute("http.url", f"https://api.example.com/users/{user_id}/stats")

        try:
            time.sleep(random.uniform(0.1, 0.3))

            if random.random() < 0.05:
                raise Exception("外部API连接超时")

            result = {
                "last_login": "2024-01-15T10:30:00Z",
                "total_spent": random.uniform(100, 5000),
                "loyalty_points": random.randint(0, 10000)
            }

            span.set_attribute("http.status_code", 200)
            return result

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("http.status_code", 504)
            span.record_exception(e)
            raise

def process_order_data(user_data: dict) -> dict:
    with tracer.start_as_current_span("order_processing") as span:
        span.set_attribute("user.orders_count", user_data["orders"])

        time.sleep(random.uniform(0.02, 0.08))

        processed = {
            **user_data,
            "order_summary": {
                "total": user_data["orders"],
                "completed": int(user_data["orders"] * 0.8),
                "pending": int(user_data["orders"] * 0.2)
            },
            "processed_at": time.time()
        }

        span.add_event("订单处理完成", attributes={"processed_orders": user_data["orders"]})
        return processed

@app.get("/api/users/{user_id}/profile")
async def get_user_profile(user_id: str, request: Request):
    with tracer.start_as_current_span("get_user_profile") as span:
        span.set_attribute("http.method", "GET")
        span.set_attribute("http.route", "/api/users/{user_id}/profile")
        span.set_attribute("user.id", user_id)

        try:
            if not simulate_user_authentication(user_id):
                return {"error": "认证失败", "status": 401}

            user_data = get_user_data_from_db(user_id)

            try:
                external_stats = call_external_api(user_id)
                user_data.update(external_stats)
            except Exception as e:
                span.add_event("外部API调用失败，使用缓存数据", attributes={"error": str(e)})
                user_data["external_stats"] = "unavailable"

            final_data = process_order_data(user_data)

            span.set_status(Status(StatusCode.OK))
            return {
                "status": "success",
                "data": final_data,
                "trace_id": span.get_span_context().trace_id
            }

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

@app.post("/api/orders")
async def create_order(request: Request):
    with tracer.start_as_current_span("create_order") as span:
        span.set_attribute("http.method", "POST")

        try:
            time.sleep(random.uniform(0.1, 0.3))

            order_id = f"ORD-{random.randint(10000, 99999)}"
            span.set_attribute("order.id", order_id)

            with tracer.start_as_current_span("validate_inventory"):
                time.sleep(random.uniform(0.01, 0.05))

            with tracer.start_as_current_span("payment_processing") as payment_span:
                payment_span.set_attribute("payment.method", "credit_card")
                time.sleep(random.uniform(0.05, 0.15))

                if random.random() < 0.05:
                    raise Exception("支付处理失败")

            with tracer.start_as_current_span("send_notification"):
                time.sleep(random.uniform(0.01, 0.03))

            span.set_status(Status(StatusCode.OK))
            return {
                "status": "success",
                "order_id": order_id,
                "trace_id": span.get_span_context().trace_id
            }

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            return {"error": str(e), "status": 500}

@app.get("/api/batch-process")
async def batch_process():
    with tracer.start_as_current_span("batch_processing") as span:
        span.set_attribute("batch.size", 10)

        results = []
        for i in range(10):
            with tracer.start_as_current_span(f"process_item_{i}") as item_span:
                item_span.set_attribute("item.index", i)
                time.sleep(random.uniform(0.01, 0.03))
                results.append({"index": i, "status": "processed"})

        return {
            "status": "completed",
            "processed_items": len(results),
            "trace_id": span.get_span_context().trace_id
        }

async def call_downstream_service(service_name: str, data: dict):
    with tracer.start_as_current_span(f"call_{service_name}") as span:
        span.set_attribute("service.name", service_name)
        span.set_attribute("call.type", "http")

        current_span = trace.get_current_span()
        trace_id = current_span.get_span_context().trace_id
        span.set_attribute("trace.propagation.id", format(trace_id, '032x'))

        time.sleep(random.uniform(0.05, 0.2))

        return {"service": service_name, "status": "ok"}

@app.get("/api/chained-call")
async def chained_call():
    with tracer.start_as_current_span("chained_service_call") as span:
        result_a = await call_downstream_service("service_a", {"step": 1})
        result_b = await call_downstream_service("service_b", {"step": 2})
        result_c = await call_downstream_service("service_c", {"step": 3})

        return {
            "status": "success",
            "results": [result_a, result_b, result_c],
            "trace_id": span.get_span_context().trace_id
        }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "tracing-example"}

if __name__ == "__main__":
    import uvicorn
    print("启动分布式追踪示例服务...")
    print("访问 http://localhost:8001/docs 查看API文档")
    print("Span数据将输出到控制台...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
