@echo off
chcp 65001 >nul
REM ============================================================
REM  Day 36: pybind11 CUDA 封装调用 - Windows 编译脚本
REM  会通过 FetchContent 自动下载 pybind11，首次编译需联网
REM  产物: build\cuda_pybind*.pyd（Python 可直接 import）
REM ============================================================
echo [Day 36] 正在编译 pybind11 CUDA 扩展模块 ...

REM 检查 nvcc 是否存在
where nvcc >nul 2>nul
if %errorlevel% neq 0 (echo [错误] 未找到 nvcc，请先安装 CUDA Toolkit。& exit /b 1)

REM 检查 cmake 是否存在
where cmake >nul 2>nul
if %errorlevel% neq 0 (echo [错误] 未找到 cmake，请先安装 CMake 3.18+。& exit /b 1)

if not exist build mkdir build
cd build

REM 配置：FetchContent 会自动拉取 pybind11
cmake .. -G "Visual Studio 17 2022" -A x64
if %errorlevel% neq 0 (echo [错误] CMake 配置失败。& cd .. & exit /b 1)

REM 编译 Release 版本
cmake --build . --config Release
if %errorlevel% neq 0 (echo [错误] 编译失败。& cd .. & exit /b 1)

cd ..
echo.
echo [Day 36] 编译成功！扩展模块已输出到 build\ 目录。
echo [Day 36] 运行演示: python python\demo.py
