@echo off
echo ========================================
echo   Speed-Demon-Cuda 构建脚本 (Windows)
echo ========================================
echo.

REM 创建构建目录
if not exist build mkdir build
cd build

REM 配置CMake
echo [1/2] 配置CMake项目...
cmake .. -G "Visual Studio 17 2022" -A x64
if %errorlevel% neq 0 (
    echo CMake配置失败！
    cd ..
    exit /b 1
)

REM 编译项目
echo.
echo [2/2] 编译项目...
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo 编译失败！
    cd ..
    exit /b 1
)

cd ..
echo.
echo ========================================
echo   构建完成！✅
echo ========================================
echo.
pause
