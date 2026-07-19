@echo off
echo ========================================
echo   Step 1: 上传文件到服务器
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] 检查压缩包...
if not exist "qingtian.tar.gz" (
    echo ERROR: qingtian.tar.gz not found!
    pause
    exit /b 1
)
echo OK: 找到 qingtian.tar.gz

echo.
echo [2/3] 测试 SSH 连接...
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 sen@192.168.10.82 "echo SSH连接成功"
if errorlevel 1 (
    echo.
    echo ERROR: SSH 连接失败！
    echo 请确认：
    echo   1. 服务器 192.168.10.82 可以访问
    echo   2. 用户名 sen 和密码 123 正确
    echo   3. SSH 服务正在运行
    pause
    exit /b 1
)

echo.
echo [3/3] 上传文件（需要输入密码 123）...
echo 请稍候，这可能需要几分钟...
scp -o StrictHostKeyChecking=no qingtian.tar.gz sen@192.168.10.82:/tmp/
if errorlevel 1 (
    echo ERROR: 上传失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo   上传完成！
echo ========================================
echo.
echo 现在运行 步骤2_部署.bat
echo.
pause
