@echo off
chcp 65001 >nul
echo [Day 37] 正在编译 CUDA ctypes 动态库 (cuda_ctypes.dll) ...

REM 检查 nvcc 是否可用
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 nvcc。请安装 CUDA Toolkit 并配置环境变量。
    exit /b 1
)

REM 创建并进入构建目录
if not exist build mkdir build
cd build

REM 使用 Visual Studio 生成器配置项目
cmake .. -G "Visual Studio 17 2022" -A x64
if %errorlevel% neq 0 (
    cd ..
    exit /b 1
)

REM 编译 Release 版本
cmake --build . --config Release
if %errorlevel% neq 0 (
    cd ..
    exit /b 1
)

cd ..
echo.
echo [Day 37] 编译成功！
echo   动态库位置: build\Release\cuda_ctypes.dll (或 build\cuda_ctypes.dll)
echo   运行演示  : python python\demo.py
