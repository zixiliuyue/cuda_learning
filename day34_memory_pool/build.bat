@echo off
chcp 65001 >nul
REM =============================================================================
REM Day 34: 自定义 GPU 内存池 — Windows 一键编译脚本
REM =============================================================================
REM 使用 nvcc 直接编译，指定 sm_120（Blackwell）架构 与 /utf-8 选项
REM 前提条件: 已安装 CUDA Toolkit 13+ 和 Visual Studio 2022
REM =============================================================================

echo [Day 34] 正在编译 Memory Pool ...

REM 检查 nvcc 是否可用
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 nvcc。请确保 CUDA Toolkit 已安装并在 PATH 中。
    exit /b 1
)

REM 创建构建目录
if not exist build mkdir build

REM 使用 nvcc 直接编译（sm_120 架构，/utf-8 处理中文源码）
nvcc -arch=sm_120 -O3 -std=c++14 ^
     -Xcompiler "/utf-8" ^
     -I..\common\include ^
     -I common\include ^
     src\main.cu -o build\memory_pool.exe

if %errorlevel% neq 0 (
    echo [错误] 编译失败。
    exit /b 1
)

echo [Day 34] 编译成功！运行: .\build\memory_pool.exe
