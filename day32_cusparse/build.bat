@echo off
chcp 65001 >nul
REM =============================================================================
REM Day 32: cuSPARSE 稀疏矩阵运算 — Windows 一键编译脚本
REM =============================================================================
REM 使用方法: 直接在 PowerShell 或 CMD 中运行 .\build.bat
REM 前提条件: 已安装 CUDA Toolkit（含 cusparse 库）和 Visual Studio 2022
REM 目标架构: sm_120 (Blackwell, 如 RTX 50 系列)
REM =============================================================================

echo [Day 32] 正在编译 cuSPARSE SpMV ...

REM 检查 nvcc 是否可用
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 nvcc。请确保 CUDA Toolkit 已安装并在 PATH 中。
    exit /b 1
)

if not exist build mkdir build
cd build

REM 使用 CMake 生成 Visual Studio 项目，指定 sm_120 架构
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
if %errorlevel% neq 0 (
    echo [错误] CMake 配置失败。
    cd ..
    exit /b 1
)

cmake --build . --config Release
if %errorlevel% neq 0 (
    echo [错误] 编译失败。
    cd ..
    exit /b 1
)

echo [Day 32] 编译成功！运行: .\build\Release\cusparse_spmv.exe
cd ..
