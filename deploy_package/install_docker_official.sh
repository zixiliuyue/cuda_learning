#!/bin/bash

PASSWORD="123"

echo "========================================"
echo "  Installing Docker using official script"
echo "========================================"
echo ""

echo "[1/3] Downloading Docker official install script..."
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
echo "[OK] Script downloaded"
echo ""

echo "[2/3] Running Docker install script..."
echo "$PASSWORD" | sudo -S sh /tmp/get-docker.sh --mirror Aliyun
echo "[OK] Docker installed"
echo ""

echo "[3/3] Starting and enabling Docker..."
echo "$PASSWORD" | sudo -S systemctl enable docker
echo "$PASSWORD" | sudo -S systemctl start docker
echo "[OK] Docker started"
echo ""

echo "Verifying Docker installation..."
echo "$PASSWORD" | sudo -S docker --version
echo ""

echo "Cleaning up..."
rm -f /tmp/get-docker.sh
echo ""

echo "========================================"
echo "  DOCKER INSTALLATION COMPLETED!"
echo "========================================"
echo ""
