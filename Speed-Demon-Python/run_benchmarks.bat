@echo off
setlocal enabledelayedexpansion

for /F "delims=#" %%E in ('"prompt #$E# & for %%E in (1) do rem"') do set "ESC=%%E"

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
echo %MAGENTA%=%RESET%   %CYAN%运行所有 Benchmark - 性能测试套件%RESET%                %MAGENTA%=%RESET%
echo %MAGENTA%========================================================================%RESET%
echo.

if exist "venv" (
    echo %CYAN%- 激活虚拟环境...%RESET%
    call venv\Scripts\activate.bat
)

echo %BLUE%[1/2]%RESET% %YELLOW%检查依赖...%RESET%
pip install pytest-benchmark numpy numba --quiet 2>nul
echo %GREEN%OK - 依赖检查完成%RESET%
echo.

echo %BLUE%[2/2]%RESET% %YELLOW%开始运行 Benchmark...%RESET%
echo.

cd 08_benchmarks
python run_all_benchmarks.py
set "EXIT_CODE=%errorlevel%"
cd ..

echo.
if %EXIT_CODE% equ 0 (
    echo %GREEN%========================================================================%RESET%
    echo %GREEN%=%RESET%   Benchmark 运行完成！
    echo %GREEN%========================================================================%RESET%
) else (
    echo %RED%========================================================================%RESET%
    echo %RED%=%RESET%   Benchmark 运行完成，但有部分错误
    echo %RED%========================================================================%RESET%
)
echo.
