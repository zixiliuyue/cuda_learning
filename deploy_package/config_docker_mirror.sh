#!/bin/bash

PASSWORD="123"

echo "========================================"
echo "  Configuring Docker Mirror"
echo "========================================"
echo ""

echo "[1/3] Creating Docker daemon config..."
echo "$PASSWORD" | sudo -S mkdir -p /etc/docker
echo "$PASSWORD" | sudo -S tee /etc/docker/daemon.json > /dev/null << 'EOF'
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
EOF
echo "[OK] Config created"
echo ""

echo "[2/3] Restarting Docker service..."
echo "$PASSWORD" | sudo -S systemctl daemon-reload
echo "$PASSWORD" | sudo -S systemctl restart docker
echo "[OK] Docker restarted"
echo ""

echo "[3/3] Verifying Docker status..."
echo "$PASSWORD" | sudo -S systemctl status docker --no-pager -l | head -20
echo ""

echo "========================================"
echo "  DOCKER MIRROR CONFIGURED!"
echo "========================================"
echo ""
