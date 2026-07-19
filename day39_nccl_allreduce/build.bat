@echo off
chcp 65001 >nul
echo [Day 39] 正在编译 NCCL AllReduce ...
where nvcc >nul 2>nul
if %errorlevel% neq 0 (echo [错误] 未找到 nvcc。& exit /b 1)
if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
if %errorlevel% neq 0 exit /b 1
cmake --build . --config Release
if %errorlevel% neq 0 exit /b 1
echo [Day 39] 编译成功！运行: .\build\Release\nccl_demo.exe
echo [提示] NCCL 官方仅支持 Linux，Windows 下通常为 CPU 模拟版本。
echo        如需真实多 GPU 通信，请在 Linux 或 WSL2 中安装 NCCL。
cd ..
