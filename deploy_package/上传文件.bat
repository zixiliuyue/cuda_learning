@echo off
chcp 65001 >nul
echo ========================================
echo   QingTian 项目文件上传工具
echo ========================================
echo.

echo [1/3] 连接服务器并创建目录...
ssh -o StrictHostKeyChecking=no sen@192.168.10.82 "mkdir -p /opt/qingtian"

echo.
echo [2/3] 上传压缩包...
scp qingtian.tar.gz sen@192.168.10.82:/opt/qingtian/

echo.
echo [3/3] 上传部署脚本...
scp deploy.sh sen@192.168.10.82:/opt/qingtian/

echo.
echo ========================================
echo   文件上传完成！
echo ========================================
echo.
echo 接下来请 SSH 登录服务器执行部署:
echo   ssh sen@192.168.10.82
echo   cd /opt/qingtian
echo   chmod +x deploy.sh
echo   bash deploy.sh
echo.
echo 服务器密码: 123
echo.
pause
