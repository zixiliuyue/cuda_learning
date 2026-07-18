@echo off
chcp 65001 >nul
REM =============================================================================
REM Day 33: CUDA IPC 进程间通信 — Windows 一键编译脚本
REM =============================================================================
REM 使用方法: 直接在 PowerShell 或 CMD 中运行 .\build.bat
REM 前提条件: 已安装 CUDA Toolkit 和 Visual Studio 2022
REM 目标架构: sm_120 (Blackwell, 如 RTX 50 系列)
REM 注意: CUDA IPC 在 Windows 上有平台限制（部分功能需 WDDM/TCC 模式支持），
REM       如遇 IPC 相关错误，建议在 Linux 环境运行以获得完整功能。
REM =============================================================================

echo [Day 33] 正在编译 CUDA IPC ...

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

echo [Day 33] 编译成功！运行: .\build\Release\cuda_ipc.exe
cd ..
