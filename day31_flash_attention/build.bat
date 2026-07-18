@echo off
chcp 65001 >nul
echo [Day 31] 正在编译 Flash Attention ...
where nvcc >nul 2>nul
if %errorlevel% neq 0 (echo [错误] 未找到 nvcc。请确认已安装 CUDA Toolkit 并配置好 PATH。& exit /b 1)
if not exist build mkdir build
rem 直接使用 nvcc 编译：
rem   -arch=sm_120     针对 Blackwell 架构（如 RTX 50 系列），可按需修改
rem   -I common/include 引入跨项目公共头文件（cuda_utils.h / timer.h / math_utils.h）
rem   -Xcompiler=/utf-8 让 MSVC 以 UTF-8 解析含中文注释的源码，消除 C4819 警告
nvcc src/main.cu -o build/flash_attention.exe -arch=sm_120 -I common/include -Xcompiler=/utf-8 -O3
if %errorlevel% neq 0 (echo [错误] 编译失败！& exit /b 1)
echo [Day 31] 编译成功！运行: .\build\flash_attention.exe
