#!/bin/bash
# Day 40: PyTorch CUDA Extension 两步构建
echo "============================================================="
echo "  Day 40: PyTorch CUDA Extension 两步构建"
echo "============================================================="

if ! command -v nvcc &> /dev/null; then
    echo "[错误] 未找到 nvcc，请先安装 CUDA Toolkit。"
    exit 1
fi

echo ""
echo "[步骤 1/2] 用 CMake 编译独立验证程序（验证 Kernel 逻辑，不依赖 PyTorch）..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [ $? -ne 0 ]; then cd ..; exit 1; fi
cd ..
echo "[步骤 1/2] 完成！运行验证: ./build/verify_kernels"

echo ""
echo "[步骤 2/2] 编译 PyTorch 扩展（python setup.py install）..."
if ! command -v python &> /dev/null; then
    echo "[警告] 未找到 python，跳过 PyTorch 扩展编译。"
    exit 0
fi
python setup.py install
if [ $? -ne 0 ]; then
    echo "[警告] PyTorch 扩展编译失败，请检查是否已安装 torch。"
    exit 0
fi
echo "[步骤 2/2] 完成！运行演示: python python/demo.py"

echo ""
echo "============================================================="
echo "  全部完成。"
echo "============================================================="
