@echo off
chcp 65001 >nul
echo ============================================================
echo [Day 38] Cython + CUDA 混合编程 —— 编译脚本
echo ============================================================

where nvcc >nul 2>nul
if %errorlevel% neq 0 (echo [错误] 未找到 nvcc，请先安装 CUDA Toolkit。& exit /b 1)

echo.
echo [步骤 1/2] 使用 CMake 编译独立 CUDA 可执行文件(验证 Kernel)...
if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
if %errorlevel% neq 0 (echo [错误] CMake 配置失败。& cd .. & exit /b 1)
cmake --build . --config Release
if %errorlevel% neq 0 (echo [错误] CMake 编译失败。& cd .. & exit /b 1)
cd ..
echo [步骤 1/2] 完成！可运行: .\build\Release\cython_cuda_demo.exe

echo.
echo [步骤 2/2] 使用 setup.py 编译 Cython CUDA 扩展...
where python >nul 2>nul
if %errorlevel% neq 0 (echo [警告] 未找到 python，跳过 Cython 扩展编译。& exit /b 0)
cd python
python setup.py build_ext --inplace
if %errorlevel% neq 0 (echo [错误] Cython 扩展编译失败。& cd .. & exit /b 1)
cd ..
echo [步骤 2/2] 完成！运行演示: cd python ^&^& python demo.py

echo.
echo ============================================================
echo [Day 38] 全部编译完成！
echo ============================================================
