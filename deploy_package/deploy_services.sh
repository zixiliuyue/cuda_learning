#!/bin/bash

PASSWORD="123"
REMOTE_PATH="/opt/qingtian"

echo "========================================"
echo "  Starting QingTian Services Deployment"
echo "========================================"
echo ""

echo "[1/5] Navigating to deploy directory..."
cd $REMOTE_PATH
echo "[OK] At $REMOTE_PATH"
echo ""

echo "[2/5] Cleaning up existing containers..."
echo "$PASSWORD" | sudo -S docker-compose down 2>/dev/null || true
echo "$PASSWORD" | sudo -S docker rm -f $(docker ps -aq --filter name=qingtian 2>/dev/null) 2>/dev/null || true
echo "[OK] Cleanup completed"
echo ""

echo "[3/5] Building and starting services (this may take 5-15 minutes)..."
echo "=================================================="
echo "$PASSWORD" | sudo -S docker-compose up -d --build
echo "=================================================="
echo "[OK] Services started"
echo ""

echo "[4/5] Waiting for services to initialize (30 seconds)..."
sleep 30
echo "[OK] Wait completed"
echo ""

echo "[5/5] Checking running containers..."
echo "$PASSWORD" | sudo -S docker ps --filter name=qingtian
echo ""

echo "========================================"
echo "  DEPLOYMENT COMPLETED!"
echo "========================================"
echo ""
echo "Frontend URL: http://192.168.10.82:80"
echo ""
echo "Useful commands:"
echo "  Enter deploy dir: cd $REMOTE_PATH"
echo "  List all containers: sudo docker ps -a"
echo "  View all logs: sudo docker-compose logs -f"
echo "  Restart services: sudo docker-compose restart"
echo "  Stop services: sudo docker-compose down"
echo ""
