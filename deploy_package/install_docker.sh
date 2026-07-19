#!/bin/bash

PASSWORD="123"

echo "========================================"
echo "  Installing Docker on Remote Server"
echo "========================================"
echo ""

echo "[1/7] Updating package list..."
echo "$PASSWORD" | sudo -S apt-get update
echo "[OK] Package list updated"
echo ""

echo "[2/7] Installing prerequisites..."
echo "$PASSWORD" | sudo -S apt-get install -y ca-certificates curl gnupg lsb-release
echo "[OK] Prerequisites installed"
echo ""

echo "[3/7] Creating keyrings directory..."
echo "$PASSWORD" | sudo -S mkdir -p /etc/apt/keyrings
echo "[OK] Directory created"
echo ""

echo "[4/7] Adding Docker GPG key..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | echo "$PASSWORD" | sudo -S gpg --no-tty --dearmor -o /etc/apt/keyrings/docker.gpg
echo "[OK] GPG key added"
echo ""

echo "[5/7] Setting up Docker repository..."
echo "$PASSWORD" | sudo -S sh -c 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null'
echo "[OK] Repository setup"
echo ""

echo "[6/7] Installing Docker..."
echo "$PASSWORD" | sudo -S apt-get update
echo "$PASSWORD" | sudo -S apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
echo "[OK] Docker installed"
echo ""

echo "[7/7] Starting and enabling Docker..."
echo "$PASSWORD" | sudo -S systemctl enable docker
echo "$PASSWORD" | sudo -S systemctl start docker
echo "[OK] Docker started"
echo ""

echo "Verifying Docker installation..."
echo "$PASSWORD" | sudo -S docker --version
echo ""

echo "========================================"
echo "  DOCKER INSTALLATION COMPLETED!"
echo "========================================"
echo ""
