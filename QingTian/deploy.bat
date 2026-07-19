@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   QingTian Project Auto Deploy Script
echo ========================================

set HOST=192.168.10.82
set PORT=80
set USER=sen
set PASSWORD=123
set REMOTE_PATH=/opt/qingtian

echo Target: %USER%@%HOST%
echo Frontend Port: %PORT%
echo.

echo [1/6] Checking Docker installation...
ssh -o StrictHostKeyChecking=no %USER%@%HOST% "docker --version" 2>nul
if %errorlevel% equ 0 (
    echo [OK] Docker is installed
) else (
    echo [INFO] Installing Docker...
    ssh %USER%@%HOST% "echo %PASSWORD% | sudo -S apt-get update && echo %PASSWORD% | sudo -S apt-get install -y ca-certificates curl gnupg lsb-release && echo %PASSWORD% | sudo -S mkdir -p /etc/apt/keyrings && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | echo %PASSWORD% | sudo -S gpg --dearmor -o /etc/apt/keyrings/docker.gpg && echo 'deb [arch='$(dpkg --print-architecture)' signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu '$(lsb_release -cs)' stable' | echo %PASSWORD% | sudo -S tee /etc/apt/sources.list.d/docker.list > /dev/null && echo %PASSWORD% | sudo -S apt-get update && echo %PASSWORD% | sudo -S apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin"
    echo [OK] Docker installed
)

echo.
echo [2/6] Preparing project files...
cd /d "%~dp0"
powershell -Command "Compress-Archive -Path '*' -DestinationPath '%TEMP%\qingtian.zip' -Force"

echo [3/6] Creating remote directory...
ssh %USER%@%HOST% "echo %PASSWORD% | sudo -S mkdir -p %REMOTE_PATH% && echo %PASSWORD% | sudo -S chown %USER%:%USER% %REMOTE_PATH%"

echo [4/6] Uploading project files...
scp -o StrictHostKeyChecking=no "%TEMP%\qingtian.zip" %USER%@%HOST%:%REMOTE_PATH%/
ssh %USER%@%HOST% "cd %REMOTE_PATH% && unzip -o qingtian.zip && rm -f qingtian.zip"

echo [5/6] Configuring for host network mode...
ssh %USER%@%HOST% "cd %REMOTE_PATH% && sed -i 's/    networks:/    network_mode: host\\n    # networks:/g' docker-compose.yml && sed -i '/^networks:/,+2d' docker-compose.yml && sed -i '/ports:/d' docker-compose.yml && sed -i 's/mongodb:\/\/mongodb:/mongodb:\/\/127.0.0.1:/g' docker-compose.yml && sed -i 's/redis:6379/127.0.0.1:6379/g' docker-compose.yml && sed -i 's/nsqd:4150/127.0.0.1:4150/g' docker-compose.yml && sed -i 's/nsqlookupd:4161/127.0.0.1:4161/g' docker-compose.yml && sed -i 's/minio:9000/127.0.0.1:9000/g' docker-compose.yml && sed -i 's/apiserver:8080/127.0.0.1:8080/g' docker-compose.yml && sed -i 's/API_SERVER=apiserver/API_SERVER=127.0.0.1\\n      - NGINX_PORT=%PORT%/g' docker-compose.yml"

echo [6/6] Building and starting services...
ssh %USER%@%HOST% "cd %REMOTE_PATH% && echo %PASSWORD% | sudo -S docker compose up -d --build"

echo.
echo ========================================
echo   Deploy Complete!
echo ========================================
echo Access URL: http://%HOST%:%PORT%
echo.
echo Check services:
echo   ssh %USER%@%HOST%
echo   cd %REMOTE_PATH%
echo   sudo docker compose ps
echo   sudo docker compose logs -f
echo.

del "%TEMP%\qingtian.zip" 2>nul
pause
