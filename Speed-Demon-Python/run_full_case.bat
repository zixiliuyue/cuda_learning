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
echo %MAGENTA%=%RESET%   %CYAN%完整案例对比 - v1_bad vs v2_good%RESET%                   %MAGENTA%=%RESET%
echo %MAGENTA%========================================================================%RESET%
echo.

if exist "venv" (
    echo %CYAN%- 激活虚拟环境...%RESET%
    call venv\Scripts\activate.bat
)

echo %BLUE%[1/5]%RESET% %YELLOW%检查依赖...%RESET%
pip install flask requests --quiet 2>nul
echo %GREEN%OK - 依赖检查完成%RESET%
echo.

echo %BLUE%[2/5]%RESET% %YELLOW%启动 v1_bad 服务 (端口 5000)...%RESET%
cd 10_full_case\v1_bad
start /B python app.py > nul 2>&1
cd ..\..
timeout /t 3 /nobreak > nul
echo %GREEN%OK - v1_bad 服务已启动%RESET%
echo.

echo %BLUE%[3/5]%RESET% %YELLOW%启动 v2_good 服务 (端口 5001)...%RESET%
cd 10_full_case\v2_good
start /B python app.py > nul 2>&1
cd ..\..
timeout /t 3 /nobreak > nul
echo %GREEN%OK - v2_good 服务已启动%RESET%
echo.

echo %BLUE%[4/5]%RESET% %YELLOW%等待服务就绪...%RESET%
timeout /t 2 /nobreak > nul
echo %GREEN%OK - 服务就绪%RESET%
echo.

echo %BLUE%[5/5]%RESET% %YELLOW%开始运行压测对比...%RESET%
echo.
cd 10_full_case
python loadtest.py --compare
cd ..

echo.
echo %CYAN%正在停止服务...%RESET%
taskkill /F /IM python.exe > nul 2>&1
timeout /t 1 /nobreak > nul

echo.
echo %GREEN%========================================================================%RESET%
echo %GREEN%=%RESET%   完整案例对比完成！
echo %GREEN%========================================================================%RESET%
echo.
