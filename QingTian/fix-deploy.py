#!/usr/bin/env python3
"""
QingTian 部署修复脚本

修复：
1. nsqd 在 host 模式下 DNS 解析失败问题
2. 前端登录 API 404 问题（nginx 代理路径）
3. 添加自动初始化管理员账号

使用：
    python fix-deploy.py --network host --port 8580
    docker compose up -d
"""

import argparse
import os
import json
import time
import urllib.request
from pathlib import Path

def reset_docker_compose(project_path: str, network_mode: str = 'bridge', frontend_port: int = 80):
    """重新生成 docker-compose.yml"""
    
    if network_mode == 'host':
        content = f"""version: '3.8'

services:
  mongodb:
    image: mongo:6
    container_name: qingtian-mongodb
    restart: unless-stopped
    volumes:
      - mongodb_data:/data/db
    network_mode: host

  redis:
    image: redis:7-alpine
    container_name: qingtian-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    network_mode: host

  nsqlookupd:
    image: nsqio/nsq:v1.2.1
    container_name: qingtian-nsqlookupd
    restart: unless-stopped
    command: /nsqlookupd --broadcast-address=127.0.0.1
    network_mode: host

  nsqd:
    image: nsqio/nsq:v1.2.1
    container_name: qingtian-nsqd
    restart: unless-stopped
    command: /nsqd --lookupd-tcp-address=127.0.0.1:4160 --broadcast-address=127.0.0.1
    depends_on:
      - nsqlookupd
    network_mode: host

  minio:
    image: minio/minio:latest
    container_name: qingtian-minio
    restart: unless-stopped
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    network_mode: host

  apiserver:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: qingtian-apiserver
    restart: unless-stopped
    command: ./apiserver -config configs/apiserver.yaml
    environment:
      QT_MONGO_URI: mongodb://127.0.0.1:27017
      QT_REDIS_ADDR: 127.0.0.1:6379
      QT_NSQD_ADDR: 127.0.0.1:4150
      QT_NSQ_LOOKUPD_ADDR: 127.0.0.1:4161
      QT_MINIO_ENDPOINT: 127.0.0.1:9000
      QT_JWT_SECRET: qingtian-secret-key
    depends_on:
      - mongodb
      - redis
      - nsqd
      - minio
    network_mode: host

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: qingtian-scheduler
    restart: unless-stopped
    command: ./scheduler -config configs/scheduler.yaml
    environment:
      QT_MONGO_URI: mongodb://127.0.0.1:27017
      QT_REDIS_ADDR: 127.0.0.1:6379
      QT_NSQD_ADDR: 127.0.0.1:4150
      QT_NSQ_LOOKUPD_ADDR: 127.0.0.1:4161
      QT_MINIO_ENDPOINT: 127.0.0.1:9000
    depends_on:
      - mongodb
      - redis
      - nsqd
      - minio
    network_mode: host

  agent:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: qingtian-agent
    restart: unless-stopped
    command: ./agent -config configs/agent.yaml
    environment:
      QT_MONGO_URI: mongodb://127.0.0.1:27017
      QT_REDIS_ADDR: 127.0.0.1:6379
      QT_NSQD_ADDR: 127.0.0.1:4150
      QT_NSQ_LOOKUPD_ADDR: 127.0.0.1:4161
      QT_MINIO_ENDPOINT: 127.0.0.1:9000
      QT_CONTROL_PLANE_URL: http://127.0.0.1:8080
    depends_on:
      - mongodb
      - redis
      - nsqd
      - minio
    network_mode: host

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: qingtian-frontend
    restart: unless-stopped
    volumes:
      - ./web/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - apiserver
    network_mode: host

volumes:
  mongodb_data:
  redis_data:
  minio_data:
"""
        nginx_content = get_nginx_conf(frontend_port, '127.0.0.1')
    else:
        content = """version: '3.8'

services:
  mongodb:
    image: mongo:6
    container_name: qingtian-mongodb
    restart: unless-stopped
    volumes:
      - mongodb_data:/data/db
    networks:
      - qingtian-network

  redis:
    image: redis:7-alpine
    container_name: qingtian-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    networks:
      - qingtian-network

  nsqlookupd:
    image: nsqio/nsq:v1.2.1
    container_name: qingtian-nsqlookupd
    restart: unless-stopped
    command: /nsqlookupd
    networks:
      - qingtian-network

  nsqd:
    image: nsqio/nsq:v1.2.1
    container_name: qingtian-nsqd
    restart: unless-stopped
    command: /nsqd --lookupd-tcp-address=nsqlookupd:4160
    depends_on:
      - nsqlookupd
    networks:
      - qingtian-network

  minio:
    image: minio/minio:latest
    container_name: qingtian-minio
    restart: unless-stopped
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    networks:
      - qingtian-network

  apiserver:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: qingtian-apiserver
    restart: unless-stopped
    command: ./apiserver -config configs/apiserver.yaml
    environment:
      QT_MONGO_URI: mongodb://mongodb:27017
      QT_REDIS_ADDR: redis:6379
      QT_NSQD_ADDR: nsqd:4150
      QT_NSQ_LOOKUPD_ADDR: nsqlookupd:4161
      QT_MINIO_ENDPOINT: minio:9000
      QT_JWT_SECRET: qingtian-secret-key
    depends_on:
      - mongodb
      - redis
      - nsqd
      - minio
    networks:
      - qingtian-network

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: qingtian-scheduler
    restart: unless-stopped
    command: ./scheduler -config configs/scheduler.yaml
    environment:
      QT_MONGO_URI: mongodb://mongodb:27017
      QT_REDIS_ADDR: redis:6379
      QT_NSQD_ADDR: nsqd:4150
      QT_NSQ_LOOKUPD_ADDR: nsqlookupd:4161
      QT_MINIO_ENDPOINT: minio:9000
    depends_on:
      - mongodb
      - redis
      - nsqd
      - minio
    networks:
      - qingtian-network

  agent:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: qingtian-agent
    restart: unless-stopped
    command: ./agent -config configs/agent.yaml
    environment:
      QT_MONGO_URI: mongodb://mongodb:27017
      QT_REDIS_ADDR: redis:6379
      QT_NSQD_ADDR: nsqd:4150
      QT_NSQ_LOOKUPD_ADDR: nsqlookupd:4161
      QT_MINIO_ENDPOINT: minio:9000
      QT_CONTROL_PLANE_URL: http://apiserver:8080
    depends_on:
      - mongodb
      - redis
      - nsqd
      - minio
    networks:
      - qingtian-network

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: qingtian-frontend
    restart: unless-stopped
    volumes:
      - ./web/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    ports:
      - "${FRONTEND_PORT:-80}:80"
    depends_on:
      - apiserver
    networks:
      - qingtian-network

volumes:
  mongodb_data:
  redis_data:
  minio_data:

networks:
  qingtian-network:
    driver: bridge
"""
        nginx_content = get_nginx_conf(80, 'apiserver')

    compose_path = os.path.join(project_path, 'docker-compose.yml')
    with open(compose_path, 'w') as f:
        f.write(content)
    print(f"✓ 已生成 docker-compose.yml (network={network_mode})")

    nginx_path = os.path.join(project_path, 'web/nginx.conf')
    with open(nginx_path, 'w') as f:
        f.write(nginx_content)
    print(f"✓ 已生成 web/nginx.conf")


def get_nginx_conf(listen_port: int, api_host: str) -> str:
    """生成 nginx 配置"""
    return f"""server {{
    listen {listen_port};
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    location / {{
        try_files $uri $uri/ /index.html;
    }}

    location /api/ {{
        proxy_pass http://{api_host}:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
"""


def _register_once(base_url: str, username: str, password: str):
    """向指定 base_url 发起一次注册请求。

    返回 (状态标志, 是否连接成功)：
      - ('created', True)  创建成功
      - ('exists',  True)  账号已存在（409）
      - ('http_err', True) 连接成功但返回其他 HTTP 错误
      - (None, False)      连接失败（该地址不可达，可尝试下一个候选）
    """
    api_url = f"{base_url}/api/v1/auth/register"
    data = json.dumps({'username': username, 'password': password, 'role': 'admin'}).encode()
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status in (200, 201):
                print(f"✓ 管理员账号创建成功: {username}  (via {base_url})")
                return 'created', True
            print(f"⚠ 账号创建返回状态: {response.status}  (via {base_url})")
            return 'http_err', True
    except urllib.error.HTTPError as e:
        # 连接成功但服务端返回错误码，说明地址可达，无需再试其他候选。
        if e.code == 409:
            print(f"✓ 管理员账号已存在: {username}  (via {base_url})")
            return 'exists', True
        print(f"⚠ 账号创建失败: {e.code} - {e.read().decode()}  (via {base_url})")
        return 'http_err', True
    except Exception as e:
        # 连接层失败（拒绝连接/超时），该地址不可达，交给上层尝试下一个候选。
        print(f"· 地址不可达，跳过: {base_url}  ({e})")
        return None, False


def init_admin_account(network: str = 'host', frontend_port: int = 80,
                       username: str = 'admin', password: str = 'qingtian123'):
    """初始化管理员账号，按网络模式自适应选择可达的 apiserver 地址。

    - host 模式：apiserver 直接监听宿主机 8080，优先访问 http://localhost:8080。
    - bridge 模式：apiserver 端口未映射到宿主机，通过前端 nginx 代理访问
      http://localhost:<frontend_port>（nginx 会把 /api/ 转发到 apiserver:8080）。

    为兼容用户实际部署方式，两种候选都会尝试，任一可达即成功。
    """
    if network == 'host':
        candidates = [
            'http://localhost:8080',                 # host 模式 apiserver 直连
            f'http://localhost:{frontend_port}',     # 兜底：经前端代理
        ]
    else:
        candidates = [
            f'http://localhost:{frontend_port}',     # bridge 模式经前端代理
            'http://localhost:8080',                 # 兜底：apiserver 若恰好映射了端口
        ]

    for base_url in candidates:
        status, reachable = _register_once(base_url, username, password)
        if reachable:
            # 地址可达（无论创建成功、已存在还是返回其他错误），不再尝试其他候选。
            return status in ('created', 'exists')
    print("⚠ 所有候选地址均不可达，请确认服务已启动。")
    return False


def main():
    parser = argparse.ArgumentParser(description='QingTian 部署修复脚本')
    parser.add_argument('--network', default='bridge', choices=['host', 'bridge'], help='网络模式')
    parser.add_argument('--port', type=int, default=80, help='前端访问端口（host 模式为 nginx 监听端口；--init-only 时用作经前端代理注册的端口）')
    parser.add_argument('--admin-user', default='admin', help='管理员用户名')
    parser.add_argument('--admin-pass', default='qingtian123', help='管理员密码')
    parser.add_argument('--init-only', action='store_true', help='仅初始化账号，不生成配置')
    
    args = parser.parse_args()
    
    project_path = str(Path(__file__).parent)
    
    print("=" * 60)
    print("QingTian 部署修复脚本")
    print("=" * 60)
    print()
    
    # --init-only：仅初始化管理员账号，直接调用注册接口，不生成配置也不打印后续步骤。
    if args.init_only:
        print(f"初始化管理员账号: {args.admin_user}")
        print("-" * 60)
        print()
        # 按网络模式自适应选择可达地址（host 直连 8080 / bridge 经前端代理）。
        ok = init_admin_account(network=args.network, frontend_port=args.port,
                                username=args.admin_user, password=args.admin_pass)
        print()
        if ok:
            print("=" * 60)
            print(f"完成，现在可用以下账号登录：")
            print(f"   用户名: {args.admin_user}")
            print(f"   密码: {args.admin_pass}")
            print("=" * 60)
        else:
            print("=" * 60)
            print("初始化失败，请检查 apiserver 是否已启动并监听 8080 端口。")
            print("排查：docker logs qingtian-apiserver --tail 30")
            print("=" * 60)
        return

    print(f"项目路径: {project_path}")
    print(f"网络模式: {args.network}")
    if args.network == 'host':
        print(f"前端端口: {args.port}")
    print("-" * 60)
    print()

    reset_docker_compose(project_path, args.network, args.port)

    print()
    print("=" * 60)
    print("下一步操作:")
    print("=" * 60)
    print()
    print("1. 清理旧容器和数据（首次部署）:")
    print("   docker compose down -v")
    print()
    print("2. 构建并启动服务:")
    print("   docker compose up -d --build")
    print()
    print("   或者前台运行查看日志:")
    print("   docker compose up --build")
    print()
    print("3. 等待所有服务启动完成（约 10-30 秒）")
    print()
    print("4. 初始化管理员账号:")
    print(f"   python fix-deploy.py --init-only --admin-user {args.admin_user} --admin-pass {args.admin_pass}")
    print()
    print(f"5. 访问: http://localhost:{args.port}")
    print(f"   用户名: {args.admin_user}")
    print(f"   密码: {args.admin_pass}")
    print()


if __name__ == '__main__':
    main()
