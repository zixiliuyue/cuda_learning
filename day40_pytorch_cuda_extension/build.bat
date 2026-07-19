@echo off
chcp 65001 >nul
echo =============================================================
echo   Day 40: PyTorch CUDA Extension 两步构建
echo =============================================================

where nvcc >nul 2>nul
if %errorlevel% neq 0 (echo [错误] 未找到 nvcc，请先安装 CUDA Toolkit。& exit /b 1)

echo.
echo [步骤 1/2] 用 CMake 编译独立验证程序（验证 Kernel 逻辑，不依赖 PyTorch）...
if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
if %errorlevel% neq 0 (cd .. & exit /b 1)
cmake --build . --config Release
if %errorlevel% neq 0 (cd .. & exit /b 1)
cd ..
echo [步骤 1/2] 完成！运行验证: .\build\Release\verify_kernels.exe

echo.
echo [步骤 2/2] 编译 PyTorch 扩展（python setup.py install）...
where python >nul 2>nul
if %errorlevel% neq 0 (echo [警告] 未找到 python，跳过 PyTorch 扩展编译。& exit /b 0)
python setup.py install
if %errorlevel% neq 0 (echo [警告] PyTorch 扩展编译失败，请检查是否已安装 torch。& exit /b 0)
echo [步骤 2/2] 完成！运行演示: python python\demo.py

echo.
echo =============================================================
echo   全部完成。
echo =============================================================
