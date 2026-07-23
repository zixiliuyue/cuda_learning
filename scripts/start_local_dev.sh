#!/bin/bash
# 用途：启动 QingTian 本地开发模式 Docker 服务
# 执行目录：QingTian 项目根目录
# 端口：8580
# 成功判断：服务启动后，前端可访问 http://localhost:8580
# 风险：会启动多个 Docker 容器，占用本地端口

set -e

cd "$(dirname "$0")/../QingTian"

echo "[INFO] 启动 QingTian 本地开发服务，端口 8580..."
python deploy_docker.py local --dev --port 8580
