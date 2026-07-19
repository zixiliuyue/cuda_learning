#!/bin/bash

PASSWORD="123"

echo "========================================"
echo "  Cleaning up and installing Docker"
echo "========================================"
echo ""

echo "[1/6] Removing problematic Docker repository..."
echo "$PASSWORD" | sudo -S rm -f /etc/apt/sources.list.d/docker.list
echo "$PASSWORD" | sudo -S rm -f /etc/apt/keyrings/docker.gpg 2>/dev/null || true
echo "$PASSWORD" | sudo -S apt-key del 7EA0A9C3F273FCD8 2>/dev/null || true
echo "[OK] Repository cleaned"
echo ""

echo "[2/6] Updating package list..."
echo "$PASSWORD" | sudo -S apt-get update
echo "[OK] Package list updated"
echo ""

echo "[3/6] Installing docker.io from Ubuntu repo (alternative)..."
echo "$PASSWORD" | sudo -S apt-get install -y docker.io docker-compose
echo "[OK] Docker installed"
echo ""

echo "[4/6] Starting and enabling Docker..."
echo "$PASSWORD" | sudo -S systemctl enable docker
echo "$PASSWORD" | sudo -S systemctl start docker
echo "[OK] Docker started"
echo ""

echo "[5/6] Adding user to docker group..."
echo "$PASSWORD" | sudo -S usermod -aG docker $USER
echo "[OK] User added to docker group"
echo ""

echo "[6/6] Verifying Docker installation..."
echo "$PASSWORD" | sudo -S docker --version
echo ""

echo "========================================"
echo "  DOCKER INSTALLATION COMPLETED!"
echo "========================================"
echo ""
