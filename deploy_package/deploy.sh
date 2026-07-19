#!/bin/bash
# QingTian 自动部署脚本

REMOTE_PATH="/opt/qingtian"
PASSWORD="123"

echo "开始部署 QingTian 项目..."

# 创建目录
mkdir -p $REMOTE_PATH

# 解压
echo "解压文件..."
tar -xzf qingtian.tar.gz -C $REMOTE_PATH

# 进入目录
cd $REMOTE_PATH

# 启动服务
echo "启动 Docker Compose..."
echo $PASSWORD | sudo -S docker compose up -d --build

echo "等待服务启动..."
sleep 10

# 检查状态
echo "服务状态:"
echo $PASSWORD | sudo -S docker compose ps

echo ""
echo "部署完成!"
echo "前端地址: http://192.168.10.82"
