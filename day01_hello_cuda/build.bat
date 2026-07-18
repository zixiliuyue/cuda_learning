@echo off
chcp 65001 >nul
REM =============================================================================
REM Day 1: Hello CUDA — Windows 一键编译脚本
REM =============================================================================
REM 使用方法: 直接在 PowerShell 或 CMD 中运行 .\build.bat
REM 前提条件: 已安装 CUDA Toolkit 和 Visual Studio 2022
REM =============================================================================

echo [Day 1] 正在编译 Hello CUDA ...

REM 检查 nvcc 是否可用
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 nvcc。请确保 CUDA Toolkit 已安装并在 PATH 中。
    echo [提示] 默认安装路径: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
    exit /b 1
)

echo [Day 1] 检测到 CUDA 编译器:
nvcc --version

REM 创建构建目录
if not exist build mkdir build
cd build

REM 使用 CMake 生成 Visual Studio 项目
echo [Day 1] 正在生成 CMake 项目 ...
cmake .. -G "Visual Studio 17 2022" -A x64

if %errorlevel% neq 0 (
    echo [错误] CMake 配置失败。
    exit /b 1
)

REM 编译项目
echo [Day 1] 正在编译 ...
cmake --build . --config Debug

if %errorlevel% neq 0 (
    echo [错误] 编译失败。
    exit /b 1
)

echo [Day 1] 编译成功！
echo [Day 1] 可执行文件位置: build\Debug\hello_cuda.exe
echo [Day 1] 运行命令: .\build\Debug\hello_cuda.exe

cd ..
