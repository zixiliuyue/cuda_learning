#!/bin/bash

PASSWORD="123"

echo "========================================"
echo "  Fixing Python distutils issue"
echo "========================================"
echo ""

echo "[1/2] Installing python3-distutils..."
echo "$PASSWORD" | sudo -S apt-get install -y python3-distutils
echo "[OK] python3-distutils installed"
echo ""

echo "[2/2] Verifying docker-compose..."
docker-compose --version
echo ""

echo "========================================"
echo "  PYTHON DISTUTILS FIXED!"
echo "========================================"
echo ""
