#!/bin/bash
# QingTian 项目完整部署脚本 - 服务器端执行
# 此脚本应在 192.168.10.82 服务器上执行

set -e

echo "========================================"
echo "  QingTian 项目 Docker 部署"
echo "========================================"
echo ""

# 配置
REMOTE_PATH="/opt/qingtian"
PASSWORD="123"

# 步骤 1: 检查并安装 Docker
echo "[步骤 1/5] 检查 Docker 安装状态..."

if command -v docker &> /dev/null; then
    echo "✓ Docker 已安装: $(docker --version)"
    DOCKER_INSTALLED=1
else
    echo "✗ Docker 未安装，开始安装..."
    
    # 更新包索引
    echo $PASSWORD | sudo -S apt-get update -y
    
    # 安装依赖
    echo $PASSWORD | sudo -S apt-get install -y ca-certificates curl gnupg lsb-release
    
    # 添加 Docker GPG 密钥
    echo $PASSWORD | sudo -S mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # 设置 Docker 仓库
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # 更新包索引
    echo $PASSWORD | sudo -S apt-get update -y
    
    # 安装 Docker
    echo $PASSWORD | sudo -S apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # 启动并启用 Docker
    echo $PASSWORD | sudo -S systemctl enable docker
    echo $PASSWORD | sudo -S systemctl start docker
    
    echo "✓ Docker 安装完成"
    DOCKER_INSTALLED=1
fi

echo ""

# 步骤 2: 创建部署目录
echo "[步骤 2/5] 创建部署目录..."
echo $PASSWORD | sudo -S mkdir -p $REMOTE_PATH
echo $PASSWORD | sudo -S chown -R $USER:$USER $REMOTE_PATH
echo "✓ 部署目录已创建: $REMOTE_PATH"
echo ""

# 步骤 3: 提示上传文件
echo "[步骤 3/5] 上传项目文件..."
echo "请在 Windows 机器的命令行中执行以下命令上传文件："
echo ""
echo "  cd F:\\code\\cuda_learning\\deploy_package"
echo "  scp qingtian.tar.gz sen@192.168.10.82:$REMOTE_PATH/"
echo ""
echo "上传时输入密码: 123"
echo ""
read -p "文件上传完成后按 Enter 键继续..."

# 检查文件是否存在
if [ ! -f "$REMOTE_PATH/qingtian.tar.gz" ]; then
    echo "✗ 未找到压缩包文件，请重新上传"
    exit 1
fi

echo "✓ 文件已上传"
echo ""

# 步骤 4: 解压文件
echo "[步骤 4/5] 解压项目文件..."
cd $REMOTE_PATH
tar -xzf qingtian.tar.gz

# 检查 docker-compose.yml 是否存在
if [ ! -f "docker-compose.yml" ]; then
    echo "✗ docker-compose.yml 不存在，请检查压缩包"
    exit 1
fi

echo "✓ 文件解压完成"
echo ""

# 步骤 5: 启动 Docker Compose
echo "[步骤 5/5] 启动 Docker Compose..."
echo "开始构建和启动所有服务，这可能需要 5-15 分钟，请耐心等待..."
echo ""

echo $PASSWORD | sudo -S docker compose up -d --build

echo ""
echo "✓ Docker Compose 执行完成"
echo ""

# 等待服务启动
echo "等待服务启动 (10秒)..."
sleep 10
echo ""

# 检查服务状态
echo "服务状态检查:"
echo $PASSWORD | sudo -S docker compose ps
echo ""

echo "========================================"
echo "  部署完成！"
echo "========================================"
echo ""
echo "前端访问地址: http://192.168.10.82"
echo ""
echo "默认登录账号:"
echo "  用户名: admin"
echo "  密码: admin123"
echo ""
echo "常用管理命令:"
echo "  cd $REMOTE_PATH"
echo "  查看状态: echo '123' | sudo -S docker compose ps"
echo "  查看日志: echo '123' | sudo -S docker compose logs -f"
echo "  重启服务: echo '123' | sudo -S docker compose restart"
echo "  停止服务: echo '123' | sudo -S docker compose down"
echo ""

# 尝试访问前端
echo "检查前端服务是否可访问..."
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:80 | grep -q "200\|401\|404"; then
    echo "✓ 前端服务响应正常"
else
    echo "  前端服务可能还在启动中，请稍后再试"
fi
echo ""