@echo off
chcp 65001 >nul
REM =============================================================================
REM Day 35: Nsight 深度性能分析 — Windows 一键编译脚本
REM =============================================================================
REM 使用 nvcc 直接编译，指定 sm_120（Blackwell）架构 与 /utf-8 选项
REM -lineinfo 便于 Nsight Compute 将指标映射回源码行
REM 前提条件: 已安装 CUDA Toolkit 13+ 和 Visual Studio 2022
REM =============================================================================

echo [Day 35] 正在编译 Nsight Profiling ...

REM 检查 nvcc 是否可用
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 nvcc。请确保 CUDA Toolkit 已安装并在 PATH 中。
    exit /b 1
)

REM 创建构建目录
if not exist build mkdir build

REM 使用 nvcc 直接编译（sm_120 架构，/utf-8 处理中文源码，-lineinfo 供 ncu 使用）
nvcc -arch=sm_120 -O3 -std=c++14 -lineinfo ^
     -Xcompiler "/utf-8" ^
     -I..\common\include ^
     -I common\include ^
     src\main.cu -o build\nsight_profiling.exe

if %errorlevel% neq 0 (
    echo [错误] 编译失败。
    exit /b 1
)

echo [Day 35] 编译成功！运行: .\build\nsight_profiling.exe
echo [Day 35] 性能分析: ncu --set full .\build\nsight_profiling.exe
