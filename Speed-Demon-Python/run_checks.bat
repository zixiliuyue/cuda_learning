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
echo %MAGENTA%=%RESET%   %CYAN%代码质量检查%RESET%                                            %MAGENTA%=%RESET%
echo %MAGENTA%========================================================================%RESET%
echo.

if exist "venv" (
    echo %CYAN%- 激活虚拟环境...%RESET%
    call venv\Scripts\activate.bat
)

echo %BLUE%[1/4]%RESET% %YELLOW%安装检查工具...%RESET%
pip install flake8 pylint mypy --quiet 2>nul
echo %GREEN%OK - 检查工具已安装%RESET%
echo.

echo %BLUE%[2/4]%RESET% %YELLOW%运行 flake8 代码风格检查...%RESET%
echo %WHITE%-------------------------------------------------------------%RESET%
flake8 --max-line-length=120 --exclude=venv,__pycache__ --ignore=E501,W503 10_full_case 08_benchmarks
if %errorlevel% equ 0 (
    echo %GREEN%flake8 - 未发现严重的代码风格问题%RESET%
) else (
    echo %YELLOW%flake8 - 发现代码风格问题，建议修复%RESET%
)
echo %WHITE%-------------------------------------------------------------%RESET%
echo.

echo %BLUE%[3/4]%RESET% %YELLOW%运行 pylint 代码质量检查...%RESET%
echo %WHITE%-------------------------------------------------------------%RESET%
pylint --disable=C0114,C0116,C0301 10_full_case 08_benchmarks
if %errorlevel% equ 0 (
    echo %GREEN%pylint - 未发现严重的代码质量问题%RESET%
) else (
    echo %YELLOW%pylint - 发现代码质量问题，建议修复%RESET%
)
echo %WHITE%-------------------------------------------------------------%RESET%
echo.

echo %BLUE%[4/4]%RESET% %YELLOW%运行 mypy 类型检查...%RESET%
echo %WHITE%-------------------------------------------------------------%RESET%
mypy --ignore-missing-imports 10_full_case 08_benchmarks
if %errorlevel% equ 0 (
    echo %GREEN%mypy - 未发现类型错误%RESET%
) else (
    echo %YELLOW%mypy - 发现类型错误，建议修复%RESET%
)
echo %WHITE%-------------------------------------------------------------%RESET%
echo.

echo %GREEN%========================================================================%RESET%
echo %GREEN%=%RESET%   代码质量检查完成！
echo %GREEN%========================================================================%RESET%
echo.
