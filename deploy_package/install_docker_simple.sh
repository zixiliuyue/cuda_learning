#!/bin/bash

PASSWORD="123"

echo "========================================"
echo "  Installing Docker using apt-key"
echo "========================================"
echo ""

echo "[1/5] Updating package list..."
echo "$PASSWORD" | sudo -S apt-get update
echo "[OK] Package list updated"
echo ""

echo "[2/5] Installing prerequisites..."
echo "$PASSWORD" | sudo -S apt-get install -y ca-certificates curl gnupg lsb-release
echo "[OK] Prerequisites installed"
echo ""

echo "[3/5] Adding Docker GPG key using apt-key..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | echo "$PASSWORD" | sudo -S apt-key add -
echo "[OK] GPG key added"
echo ""

echo "[4/5] Setting up Docker repository..."
echo "$PASSWORD" | sudo -S sh -c 'echo "deb [arch=$(dpkg --print-architecture)] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null'
echo "[OK] Repository setup"
echo ""

echo "[5/5] Installing Docker..."
echo "$PASSWORD" | sudo -S apt-get update
echo "$PASSWORD" | sudo -S apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
echo "$PASSWORD" | sudo -S systemctl enable docker
echo "$PASSWORD" | sudo -S systemctl start docker
echo "[OK] Docker installed and started"
echo ""

echo "Verifying Docker installation..."
echo "$PASSWORD" | sudo -S docker --version
echo ""

echo "========================================"
echo "  DOCKER INSTALLATION COMPLETED!"
echo "========================================"
echo ""
