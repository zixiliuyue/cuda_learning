@echo off
setlocal enabledelayedexpansion

for /F "delims=#" %%E in ('"prompt #$E# & for %%E in (1) do rem"') do set "ESC=%%E"

title Speed Demon Python - 一键运行脚本
cls

set "RED=%ESC%[91m"
set "GREEN=%ESC%[92m"
set "YELLOW=%ESC%[93m"
set "BLUE=%ESC%[94m"
set "MAGENTA=%ESC%[95m"
set "CYAN=%ESC%[96m"
set "WHITE=%ESC%[97m"
set "RESET=%ESC%[0m"

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo %MAGENTA%========================================================================%RESET%
echo %MAGENTA%=%RESET%   %CYAN%Speed Demon Python - 一键运行脚本%RESET%              %MAGENTA%=%RESET%
echo %MAGENTA%=%RESET%   %WHITE%Python性能优化实战教程%RESET%                                  %MAGENTA%=%RESET%
echo %MAGENTA%========================================================================%RESET%
echo.

echo %BLUE%[1/6]%RESET% %YELLOW%检查Python环境...%RESET%
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%错误: 未找到Python，请先安装Python 3.8+%RESET%
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PY_VERSION=%%i"
echo %GREEN%OK - Python版本: %PY_VERSION%%RESET%
echo.

echo %BLUE%[2/6]%RESET% %YELLOW%检查并安装依赖...%RESET%
if not exist "venv" (
    echo %CYAN%- 创建虚拟环境...%RESET%
    python -m venv venv
    if %errorlevel% neq 0 (
        echo %RED%错误: 无法创建虚拟环境%RESET%
        pause
        exit /b 1
    )
    echo %GREEN%OK - 虚拟环境创建成功%RESET%
)

echo %CYAN%- 激活虚拟环境...%RESET%
call venv\Scripts\activate.bat

echo %CYAN%- 安装/更新依赖包...%RESET%
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo %RED%警告: 部分依赖安装失败，继续执行...%RESET%
) else (
    echo %GREEN%OK - 依赖安装完成%RESET%
)
echo.

:start
echo %BLUE%[3/6]%RESET% %YELLOW%请选择要运行的内容:%RESET%
echo.
echo %CYAN%  1.%RESET% 运行所有 Benchmark (性能测试)
echo %CYAN%  2.%RESET% 完整案例对比 (v1_bad vs v2_good)
echo %CYAN%  3.%RESET% 代码质量检查
echo %CYAN%  4.%RESET% 运行全部 (Benchmark + 案例对比 + 代码检查)
echo %CYAN%  5.%RESET% 仅安装依赖并退出
echo %CYAN%  0.%RESET% 退出
echo.

set /p "CHOICE=%YELLOW%请输入选项 [0-5]: %RESET%"

if "%CHOICE%"=="1" goto run_benchmarks
if "%CHOICE%"=="2" goto run_full_case
if "%CHOICE%"=="3" goto run_checks
if "%CHOICE%"=="4" goto run_all
if "%CHOICE%"=="5" goto end
if "%CHOICE%"=="0" goto end

echo %RED%无效选项，退出...%RESET%
pause
exit /b 1

:run_benchmarks
echo.
echo %BLUE%[4/6]%RESET% %YELLOW%运行所有 Benchmark...%RESET%
call run_benchmarks.bat
goto check_continue

:run_full_case
echo.
echo %BLUE%[4/6]%RESET% %YELLOW%运行完整案例对比...%RESET%
call run_full_case.bat
goto check_continue

:run_checks
echo.
echo %BLUE%[4/6]%RESET% %YELLOW%运行代码质量检查...%RESET%
call run_checks.bat
goto check_continue

:run_all
echo.
echo %BLUE%[4/6]%RESET% %YELLOW%运行全部测试...%RESET%
echo.
echo %CYAN%=== 第一部分: Benchmarks ===%RESET%
call run_benchmarks.bat
echo.
echo %CYAN%=== 第二部分: 完整案例对比 ===%RESET%
call run_full_case.bat
echo.
echo %CYAN%=== 第三部分: 代码质量检查 ===%RESET%
call run_checks.bat
goto end

:check_continue
echo.
set /p "CONTINUE=%YELLOW%是否继续运行其他项目? (y/n): %RESET%"
if /i "%CONTINUE%"=="y" (
    cls
    goto start
)
goto end

:end
echo.
echo %MAGENTA%========================================================================%RESET%
echo %MAGENTA%=%RESET%   %GREEN%运行完成！感谢使用 Speed Demon Python%RESET%            %MAGENTA%=%RESET%
echo %MAGENTA%========================================================================%RESET%
echo.
pause
