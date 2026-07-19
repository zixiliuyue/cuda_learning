@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM 一键运行所有压测对比脚本 (Windows 版本)
REM 使用方法: run_all_loadtests.bat
REM ============================================================

set "SCRIPT_DIR=%~dp0"
set "BASE_URL=http://localhost:8080"
set "OUTPUT_DIR=%SCRIPT_DIR%results"
set "TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

cd /d "%SCRIPT_DIR%"

cls
echo.
echo ╔═════════════════════════════════════════╗
echo ║      Speed Demon 压测对比工具           ║
echo ╚═════════════════════════════════════════╝
echo.

REM 检查服务是否启动
echo [INFO] 检查服务是否启动...
set "SERVICE_UP=0"
for /l %%i in (1,1,30) do (
    curl -s "%BASE_URL%/stats" >nul 2>&1
    if not errorlevel 1 (
        set "SERVICE_UP=1"
        goto service_found
    )
    timeout /t 1 /nobreak >nul
)
:service_found

if "%SERVICE_UP%"=="0" (
    echo [ERROR] 服务未启动! 请先启动 v1_bad 或 v2_good 服务
    echo.
    echo 启动命令:
    echo   cd ..\05_full_case\v1_bad ^&^& go run main.go
    echo   或
    echo   cd ..\05_full_case\v2_good ^&^& go run main.go
    echo.
    exit /b 1
)

echo [SUCCESS] 服务已启动!

echo.
echo ==========================================
echo 收集当前服务信息
echo ==========================================
curl -s "%BASE_URL%/stats"
echo.

echo.
echo ==========================================
echo 开始压测
echo ==========================================
echo [INFO] 压测目标: %BASE_URL%
echo [INFO] 结果目录: %OUTPUT_DIR%
echo [INFO] 时间戳: %TIMESTAMP%
echo.

set /p "VERSION=请确认服务版本 (v1_bad/v2_good): "

if /i not "%VERSION%"=="v1_bad" if /i not "%VERSION%"=="v2_good" (
    echo [ERROR] 无效的版本!
    exit /b 1
)

echo [INFO] 将对 %VERSION% 进行压测...

echo.
echo ==========================================
echo 步骤 1: 预热
echo ==========================================
echo [INFO] 运行预热: 500 请求, 10 并发...
hey -n 500 -c 10 -disable-keepalive=false "%BASE_URL%/api/product?id=123" > "%OUTPUT_DIR%\%VERSION%_warmup_%TIMESTAMP%.txt" 2>&1
echo [SUCCESS] 预热完成

timeout /t 5 /nobreak >nul

echo.
echo ==========================================
echo 步骤 2: 基准测试
echo ==========================================
echo [INFO] 运行基准测试: 2000 请求, 50 并发...
hey -n 2000 -c 50 -disable-keepalive=false "%BASE_URL%/api/product?id=123" > "%OUTPUT_DIR%\%VERSION%_baseline_%TIMESTAMP%.txt" 2>&1
echo [SUCCESS] 基准测试完成

timeout /t 10 /nobreak >nul

echo.
echo ==========================================
echo 步骤 3: 高并发测试
echo ==========================================
echo [INFO] 运行高并发测试: 5000 请求, 100 并发...
hey -n 5000 -c 100 -disable-keepalive=false "%BASE_URL%/api/product?id=123" > "%OUTPUT_DIR%\%VERSION%_high_%TIMESTAMP%.txt" 2>&1
echo [SUCCESS] 高并发测试完成

echo.
echo ==========================================
echo 压测完成!
echo ==========================================

set "HIGH_FILE="
for /f "delims=" %%f in ('dir /b /o-d "%OUTPUT_DIR%\%VERSION%_high_*.txt" 2^>nul') do (
    set "HIGH_FILE=%OUTPUT_DIR%\%%f"
    goto found_high
)
:found_high

echo [SUCCESS] 高并发结果文件: %HIGH_FILE%

echo.
echo 关键指标:
if exist "%HIGH_FILE%" (
    findstr /c:"Requests/sec" /c:"P99" /c:"P95" /c:"P90" /c:"P50" "%HIGH_FILE%"
)

echo.
echo [INFO] 提示: 你可以切换到另一个版本的服务后再次运行此脚本
echo [INFO]       然后手动对比两个版本的结果
echo.
echo 下一步操作:
echo   1. Ctrl+C 停止当前服务
echo   2. 启动另一个版本的服务
echo   3. 再次运行此脚本
echo   4. 对比两个版本的性能差异
echo.

endlocal
