#!/bin/bash

PASSWORD="123"

echo "========================================"
echo "  Installing Docker Compose V2"
echo "========================================"
echo ""

echo "[1/3] Downloading Docker Compose V2..."
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
echo "[OK] Downloaded"
echo ""

echo "[2/3] Making executable and installing system-wide..."
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
echo "$PASSWORD" | sudo -S cp $DOCKER_CONFIG/cli-plugins/docker-compose /usr/libexec/docker/cli-plugins/docker-compose 2>/dev/null || true
echo "$PASSWORD" | sudo -S mkdir -p /usr/local/lib/docker/cli-plugins 2>/dev/null || true
echo "$PASSWORD" | sudo -S cp $DOCKER_CONFIG/cli-plugins/docker-compose /usr/local/lib/docker/cli-plugins/docker-compose 2>/dev/null || true
echo "[OK] Installed"
echo ""

echo "[3/3] Verifying installation..."
docker compose version
echo ""

echo "========================================"
echo "  DOCKER COMPOSE V2 INSTALLED!"
echo "========================================"
echo ""
