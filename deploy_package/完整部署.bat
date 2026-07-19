@echo off
setlocal enabledelayedexpansion

set SERVER=192.168.10.82
set USER=sen
set PASSWORD=123
set REMOTE_PATH=/opt/qingtian
set PACKAGE_DIR=F:\code\cuda_learning\deploy_package

echo ========================================
echo   QingTian 项目自动部署工具
echo ========================================
echo.

REM Step 1: 检查服务器连接
echo [1/5] 检查服务器连接...
ping %SERVER% -n 1 -w 3000 >nul
if errorlevel 1 (
    echo   [错误] 无法连接到服务器 %SERVER%
    pause
    exit /b 1
)
echo   ✓ 服务器连接正常

REM Step 2: 检查 Docker
echo.
echo [2/5] 检查 Docker 安装状态...
REM 使用临时 PowerShell 脚本
set "PS_SCRIPT=%TEMP%\check_docker.ps1"
echo $pinfo = New-Object System.Diagnostics.ProcessStartInfo > "%PS_SCRIPT%"
echo $pinfo.FileName = "ssh" >> "%PS_SCRIPT%"
echo $pinfo.Arguments = "-o StrictHostKeyChecking=no -o ConnectTimeout=10 %USER%@%SERVER% `"which docker`"" >> "%PS_SCRIPT%"
echo $pinfo.RedirectStandardInput = $true >> "%PS_SCRIPT%"
echo $pinfo.RedirectStandardOutput = $true >> "%PS_SCRIPT%"
echo $pinfo.RedirectStandardError = $true >> "%PS_SCRIPT%"
echo $pinfo.UseShellExecute = $false >> "%PS_SCRIPT%"
echo $p = New-Object System.Diagnostics.Process >> "%PS_SCRIPT%"
echo $p.StartInfo = $pinfo >> "%PS_SCRIPT%"
echo $p.Start() ^| Out-Null >> "%PS_SCRIPT%"
echo Start-Sleep -m 1500 >> "%PS_SCRIPT%"
echo $p.StandardInput.WriteLine("%PASSWORD%") >> "%PS_SCRIPT%"
echo $p.StandardInput.Flush() >> "%PS_SCRIPT%"
echo Start-Sleep -m 2000 >> "%PS_SCRIPT%"
echo if ($p.WaitForExit(5000)) { >> "%PS_SCRIPT%"
echo   $output = $p.StandardOutput.ReadToEnd() >> "%PS_SCRIPT%"
echo   Write-Host "DOCKER_OUTPUT:$output" >> "%PS_SCRIPT%"
echo   exit $p.ExitCode >> "%PS_SCRIPT%"
echo } else { >> "%PS_SCRIPT%"
echo   $p.Kill() >> "%PS_SCRIPT%"
echo   exit 1 >> "%PS_SCRIPT%"
echo } >> "%PS_SCRIPT%"

for /f "tokens=*" %%i in ('powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%PS_SCRIPT%" 2^>^&1') do (
    set "LINE=%%i"
    if "!LINE:~0,13!"=="DOCKER_OUTPUT:" (
        set "DOCKER_OUT=!LINE:~13!"
    )
)

del "%PS_SCRIPT%" 2>nul

if defined DOCKER_OUT (
    if "!DOCKER_OUT!"=="" (
        echo   [!] Docker 未安装
        set INSTALL_DOCKER=1
    ) else (
        echo   ✓ Docker 已安装: !DOCKER_OUT!
        set INSTALL_DOCKER=0
    )
) else (
    echo   [!] 无法检测 Docker 状态，继续部署
    set INSTALL_DOCKER=0
)

REM Step 3: 上传文件
echo.
echo [3/5] 上传项目文件...
echo   正在创建远程目录...
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 %USER%@%SERVER% "mkdir -p %REMOTE_PATH%"

echo   正在上传压缩包...
scp -o StrictHostKeyChecking=no "%PACKAGE_DIR%\qingtian.tar.gz" %USER%@%SERVER%:%REMOTE_PATH%/
if errorlevel 1 (
    echo   [错误] 文件上传失败
    echo   请确保:
    echo   1. 已输入正确密码: %PASSWORD%
    echo   2. 服务器可访问
    echo   3. 有 /opt/qingtian 目录的写权限
    pause
    exit /b 1
)
echo   ✓ 文件上传成功

REM Step 4: 解压并部署
echo.
echo [4/5] 执行远程部署...

set "DEPLOY_SCRIPT=%TEMP%\deploy_remote.sh"
echo #!/bin/bash > "%DEPLOY_SCRIPT%"
echo set -e >> "%DEPLOY_SCRIPT%"
echo cd %REMOTE_PATH% >> "%DEPLOY_SCRIPT%"
echo tar -xzf qingtian.tar.gz >> "%DEPLOY_SCRIPT%"
echo echo "解压完成" >> "%DEPLOY_SCRIPT%"
echo echo '%PASSWORD%' ^| sudo -S docker compose up -d --build >> "%DEPLOY_SCRIPT%"
echo echo "Docker Compose 执行完成" >> "%DEPLOY_SCRIPT%"
echo sleep 5 >> "%DEPLOY_SCRIPT%"
echo echo '%PASSWORD%' ^| sudo -S docker compose ps >> "%DEPLOY_SCRIPT%"

scp -o StrictHostKeyChecking=no "%DEPLOY_SCRIPT%" %USER%@%SERVER%:%REMOTE_PATH%/deploy.sh
del "%DEPLOY_SCRIPT%" 2>nul

echo   在远程服务器执行部署命令...
echo   这可能需要 5-15 分钟，请耐心等待...
echo.

REM 执行远程部署脚本
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=600 %USER%@%SERVER% "chmod +x %REMOTE_PATH%/deploy.sh && bash %REMOTE_PATH%/deploy.sh"

REM Step 5: 完成
echo.
echo ========================================
echo   [5/5] 部署完成！
echo ========================================
echo.
echo 前端访问地址: http://%SERVER%
echo.
echo 默认登录账号:
echo   用户名: admin
echo   密码: admin123
echo.
echo 常用命令:
echo   ssh %USER%@%SERVER%
echo   cd %REMOTE_PATH%
echo   echo '%PASSWORD%' ^| sudo -S docker compose ps
echo   echo '%PASSWORD%' ^| sudo -S docker compose logs -f
echo.
pause