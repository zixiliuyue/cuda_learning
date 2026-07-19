@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM hey 压测脚本 (Windows 版本)
REM 使用方法: loadtest_hey.bat <目标>
REM 目标: v1_bad | v2_good
REM ============================================================

set "TARGET=%1"
set "BASE_URL=http://localhost:8080"
set "OUTPUT_DIR=.\results"
set "TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

if "%TARGET%"=="" (
    echo 用法: %0 ^<目标^>
    echo 目标: v1_bad ^| v2_good
    exit /b 1
)

hey --help >nul 2>&1
if errorlevel 1 (
    echo 错误: hey 未安装
    echo 安装方法: go install github.com/rakyll/hey@latest
    exit /b 1
)

echo ==========================================
echo 开始压测: %TARGET%
echo 目标地址: %BASE_URL%
echo 时间: %TIMESTAMP%
echo ==========================================

echo.
echo ==========================================
echo 步骤 1: 预热 (低并发)
echo ==========================================
hey -n 500 -c 10 %BASE_URL%/api/product?id=123 > "%OUTPUT_DIR%\%TARGET%_warmup_%TIMESTAMP%.txt"
echo 预热完成!

timeout /t 5 /nobreak >nul

echo.
echo ==========================================
echo 步骤 2: 基准测试 (中等并发)
echo ==========================================
hey -n 2000 -c 50 %BASE_URL%/api/product?id=123 > "%OUTPUT_DIR%\%TARGET%_baseline_%TIMESTAMP%.txt"
echo 基准测试完成!

timeout /t 10 /nobreak >nul

echo.
echo ==========================================
echo 步骤 3: 高并发测试
echo ==========================================
hey -n 5000 -c 100 %BASE_URL%/api/product?id=123 > "%OUTPUT_DIR%\%TARGET%_high_concurrency_%TIMESTAMP%.txt"
echo 高并发测试完成!

echo.
echo ==========================================
echo 压测完成! 结果保存在: %OUTPUT_DIR%
echo ==========================================

echo.
echo 最新结果摘要:
for /f "delims=" %%f in ('dir /b /o-d "%OUTPUT_DIR%\%TARGET%_high_concurrency_*.txt" 2^>nul') do (
    set "LATEST_FILE=%OUTPUT_DIR%\%%f"
    goto :found
)
:found

if exist "%LATEST_FILE%" (
    findstr /c:"Requests/sec" /c:"Summary" /c:"Slowest" /c:"Fastest" /c:"Average" /c:"P99" /c:"P95" /c:"P90" /c:"P50" "%LATEST_FILE%"
)

echo.
echo 提示: 可以对比 v1_bad 和 v2_good 的结果差异!

endlocal
