#!/bin/bash
# =============================================================================
# Day 38: Cython + CUDA 混合编程 —— 编译脚本 (Linux / macOS)
# =============================================================================
echo "============================================================"
echo "[Day 38] Cython + CUDA 混合编程 —— 编译脚本"
echo "============================================================"

if ! command -v nvcc &> /dev/null; then
    echo "[警告] 未找到 nvcc。macOS 无 NVIDIA GPU 用户请使用 Python/Colab 环境。"
    exit 1
fi

echo ""
echo "[步骤 1/2] 使用 CMake 编译独立 CUDA 可执行文件(验证 Kernel)..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ]; then echo "[错误] CMake 配置失败。"; cd ..; exit 1; fi
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [ $? -ne 0 ]; then echo "[错误] CMake 编译失败。"; cd ..; exit 1; fi
cd ..
echo "[步骤 1/2] 完成！可运行: ./build/cython_cuda_demo"

echo ""
echo "[步骤 2/2] 使用 setup.py 编译 Cython CUDA 扩展..."
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "[警告] 未找到 python，跳过 Cython 扩展编译。"
    exit 0
fi
PY=$(command -v python3 || command -v python)
cd python
$PY setup.py build_ext --inplace
if [ $? -ne 0 ]; then echo "[错误] Cython 扩展编译失败。"; cd ..; exit 1; fi
cd ..
echo "[步骤 2/2] 完成！运行演示: cd python && $PY demo.py"

echo ""
echo "============================================================"
echo "[Day 38] 全部编译完成！"
echo "============================================================"
