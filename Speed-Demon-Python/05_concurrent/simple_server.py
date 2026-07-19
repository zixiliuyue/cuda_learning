"""
简单的HTTP测试服务器
用于 asyncio 性能测试场景
支持不同延迟的请求
"""

import asyncio
import random
from aiohttp import web


async def handle_fast(request):
    """快速响应 10-50ms"""
    await asyncio.sleep(random.uniform(0.01, 0.05))
    return web.json_response({"status": "ok", "delay": "fast"})


async def handle_medium(request):
    """中等响应 100-300ms"""
    await asyncio.sleep(random.uniform(0.1, 0.3))
    return web.json_response({"status": "ok", "delay": "medium"})


async def handle_slow(request):
    """慢速响应 500-1000ms"""
    await asyncio.sleep(random.uniform(0.5, 1.0))
    return web.json_response({"status": "ok", "delay": "slow"})


async def handle_variable(request):
    """可变延迟 10-500ms"""
    delay = random.uniform(0.01, 0.5)
    await asyncio.sleep(delay)
    return web.json_response({"status": "ok", "delay": delay})


async def handle_data(request):
    """返回数据"""
    data_size = int(request.query.get("size", 1024))
    await asyncio.sleep(0.01)
    return web.json_response({
        "status": "ok",
        "data": "x" * data_size
    })


async def handle_error(request):
    """随机错误"""
    if random.random() < 0.3:
        return web.json_response({"error": "server error"}, status=500)
    await asyncio.sleep(0.05)
    return web.json_response({"status": "ok"})


def create_app():
    app = web.Application()
    app.router.add_get('/fast', handle_fast)
    app.router.add_get('/medium', handle_medium)
    app.router.add_get('/slow', handle_slow)
    app.router.add_get('/variable', handle_variable)
    app.router.add_get('/data', handle_data)
    app.router.add_get('/error', handle_error)
    return app


if __name__ == '__main__':
    print("=" * 60)
    print("测试服务器启动在 http://localhost:8080")
    print("可用端点:")
    print("  GET /fast      - 快速响应 10-50ms")
    print("  GET /medium    - 中等响应 100-300ms")
    print("  GET /slow      - 慢速响应 500-1000ms")
    print("  GET /variable  - 可变延迟 10-500ms")
    print("  GET /data      - 返回数据 (可指定 size 参数)")
    print("  GET /error     - 30% 概率返回错误")
    print("=" * 60)
    web.run_app(create_app(), host='127.0.0.1', port=8080)
