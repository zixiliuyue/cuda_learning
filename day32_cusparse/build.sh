#!/bin/bash
# =============================================================================
# Day 32: cuSPARSE 稀疏矩阵运算 — macOS/Linux 一键编译脚本
# =============================================================================
echo "[Day 32] 正在编译 cuSPARSE SpMV ..."

if ! command -v nvcc &> /dev/null; then
    echo "[警告] 未找到 nvcc。macOS 用户请使用 Python 版本 (python/cusparse.py)。"
    exit 1
fi

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ]; then
    echo "[错误] CMake 配置失败。"
    exit 1
fi

cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [ $? -ne 0 ]; then
    echo "[错误] 编译失败。"
    exit 1
fi

echo "[Day 32] 编译成功！运行: ./build/cusparse_spmv"
cd ..
