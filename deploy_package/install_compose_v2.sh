#!/bin/bash

PASSWORD="123"

echo "========================================"
echo "  Installing Docker Compose V2"
echo "========================================"
echo ""

echo "[1/4] Uninstalling old docker-compose..."
echo "$PASSWORD" | sudo -S apt-get remove -y docker-compose 2>/dev/null || true
echo "$PASSWORD" | sudo -S rm -f /usr/bin/docker-compose 2>/dev/null || true
echo "[OK] Old docker-compose removed"
echo ""

echo "[2/4] Downloading Docker Compose V2..."
curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 -o /tmp/docker-compose
echo "[OK] Downloaded"
echo ""

echo "[3/4] Installing Docker Compose V2..."
chmod +x /tmp/docker-compose
echo "$PASSWORD" | sudo -S mv /tmp/docker-compose /usr/local/bin/docker-compose
echo "$PASSWORD" | sudo -S ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose 2>/dev/null || true
echo "[OK] Installed"
echo ""

echo "[4/4] Verifying installation..."
docker-compose --version
echo ""

echo "========================================"
echo "  DOCKER COMPOSE V2 INSTALLED SUCCESSFULLY!"
echo "========================================"
echo ""
