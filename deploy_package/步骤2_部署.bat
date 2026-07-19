@echo off
echo ========================================
echo   Step 2: 在服务器上部署
echo ========================================
echo.

echo 正在连接服务器并执行部署命令...
echo.

ssh -o StrictHostKeyChecking=no sen@192.168.10.82 "
echo '========================================'
echo '  开始部署'
echo '========================================'
echo ''

echo '[1/6] 创建目录...'
echo '123' | sudo -S mkdir -p /opt/qingtian
echo '123' | sudo -S chown sen:sen /opt/qingtian
cd /opt/qingtian

echo ''
echo '[2/6] 解压文件...'
mv /tmp/qingtian.tar.gz .
tar -xzf qingtian.tar.gz
rm -f qingtian.tar.gz

echo ''
echo '[3/6] 停止现有容器...'
echo '123' | sudo -S docker compose down 2>/dev/null || true
echo '123' | sudo -S docker rm -f $(docker ps -aq --filter name=qingtian 2>/dev/null) 2>/dev/null || true

echo ''
echo '[4/6] 清理旧镜像（可选）...'
echo '123' | sudo -S docker system prune -f 2>/dev/null || true

echo ''
echo '[5/6] 开始构建并启动服务...'
echo '这可能需要 5-15 分钟，请耐心等待...'
echo '========================================'
echo '123' | sudo -S docker compose up -d --build

echo ''
echo '[6/6] 等待服务启动...'
sleep 10

echo ''
echo '========================================'
echo '  部署完成！'
echo '========================================'
echo ''
echo '运行中的容器：'
echo '123' | sudo -S docker ps --filter name=qingtian
echo ''
echo '========================================'
echo '前端访问地址: http://192.168.10.82'
echo '========================================'
echo ''
echo '常用命令:'
echo '  查看日志: sudo docker compose logs -f'
echo '  查看状态: sudo docker compose ps'
echo '  重启服务: sudo docker compose restart'
echo '  停止服务: sudo docker compose down'
"

echo.
echo 按任意键退出...
pause >nul
