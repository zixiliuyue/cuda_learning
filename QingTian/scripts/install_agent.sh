#!/bin/bash
# QingTian Agent 一键安装脚本
# 用法: curl -sfL https://raw.githubusercontent.com/your-repo/QingTian/main/scripts/install_agent.sh | bash -s -- <CONTROL_PLANE_URL>
#
# 参数:
#   CONTROL_PLANE_URL: QingTian 控制平面 API 地址，例如 http://192.168.1.100:8080/api/v1

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取控制平面地址
if [ $# -ne 1 ]; then
    echo -e "${RED}错误: 缺少控制平面地址参数${NC}"
    echo "用法: $0 <CONTROL_PLANE_URL>"
    echo "示例: $0 http://192.168.1.100:8080/api/v1"
    exit 1
fi
CONTROL_PLANE_URL="$1"

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  QingTian Agent 一键安装${NC}"
echo -e "${GREEN}  控制平面地址: $CONTROL_PLANE_URL${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

# 检查系统依赖
echo "[1/6] 检查系统依赖..."
command -v git >/dev/null 2>&1 || { echo -e "${RED}git 未安装，请先安装 git${NC}"; exit 1; }
command -v go >/dev/null 2>&1 || { echo -e "${RED}Go 未安装，请先安装 Go 1.20+${NC}"; exit 1; }
command -v gcc >/dev/null 2>&1 || { echo -e "${YELLOW}警告: gcc 未找到，编译可能失败${NC}"; }

# 创建工作目录
echo "[2/6] 创建工作目录..."
INSTALL_DIR="$HOME/qingtian"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# 克隆代码
echo "[3/6] 克隆代码..."
REPO_URL="${QT_REPO:-https://github.com/qingtian-com/QingTian.git}"
if [ ! -d "$INSTALL_DIR/QingTian" ]; then
  git clone "$REPO_URL"
else
  echo "代码已存在，拉取最新变更..."
  cd QingTian
  git pull
  cd ..
fi

cd QingTian

# 编译agent
echo "[4/6] 编译 agent..."
cd cmd/agent
go build -o "$INSTALL_DIR/bin/agent"

# 创建配置文件
echo "[5/6] 生成配置文件..."
mkdir -p "$INSTALL_DIR/configs"
cat > "$INSTALL_DIR/configs/agent.yaml" << EOF
# QingTian Agent 配置
node_name: "$(hostname)"
control_plane_url: $CONTROL_PLANE_URL
heartbeat_interval: 5s
work_dir: $INSTALL_DIR/work
EOF

# 创建systemd服务
echo "[6/6] 安装 systemd 服务..."
if [ -d "/etc/systemd/system" ]; then
  sudo tee /etc/systemd/system/qingtian-agent.service > /dev/null << EOF2
[Unit]
Description=QingTian Agent
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/bin/agent -config $INSTALL_DIR/configs/agent.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF2

  sudo systemctl daemon-reload
  sudo systemctl enable qingtian-agent
  sudo systemctl start qingtian-agent

  echo ""
  echo -e "${GREEN}=========================================${NC}"
  echo -e "${GREEN}  安装完成!${NC}"
  echo -e "${GREEN}  状态查询: sudo systemctl status qingtian-agent${NC}"
  echo -e "${GREEN}  查看日志: sudo journalctl -u qingtian-agent -f${NC}"
  echo -e "${GREEN}  说明: Agent 不监听端口，主动连接控制平面${NC}"
  echo -e "${GREEN}=========================================${NC}"
else
  echo ""
  echo -e "${YELLOW}未检测到 systemd，请手动启动:${NC}"
  echo "  $INSTALL_DIR/bin/agent -config $INSTALL_DIR/configs/agent.yaml &"
  echo -e "${GREEN}安装完成${NC}"
fi
