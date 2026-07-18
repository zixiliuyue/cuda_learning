@echo off
chcp 65001 >nul
echo [Day 21] 正在编译 PyTorch Extension Demo ...
where nvcc >nul 2>nul
if %errorlevel% neq 0 (echo [错误] 未找到 nvcc。& exit /b 1)
if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
if %errorlevel% neq 0 exit /b 1
cmake --build . --config Release
if %errorlevel% neq 0 exit /b 1
echo [Day 21] 编译成功！运行: .\build\Release\pytorch_extension_demo.exe
cd ..
