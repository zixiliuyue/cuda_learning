#!/bin/bash
# ============================================================
#  Day 36: pybind11 CUDA 封装调用 - Linux 编译脚本
#  会通过 FetchContent 自动下载 pybind11，首次编译需联网
#  产物: build/cuda_pybind*.so（Python 可直接 import）
# ============================================================
echo "[Day 36] 正在编译 pybind11 CUDA 扩展模块 ..."

# 检查 nvcc
if ! command -v nvcc &> /dev/null; then
    echo "[警告] 未找到 nvcc。macOS/无 GPU 环境请使用 python demo.py 的 NumPy fallback。"
    exit 1
fi

# 检查 cmake
if ! command -v cmake &> /dev/null; then
    echo "[错误] 未找到 cmake，请先安装 CMake 3.18+。"
    exit 1
fi

mkdir -p build && cd build

# 配置：FetchContent 会自动拉取 pybind11
cmake .. -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ]; then
    echo "[错误] CMake 配置失败。"
    cd ..
    exit 1
fi

# 编译（自动检测 CPU 核心数并行编译）
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [ $? -ne 0 ]; then
    echo "[错误] 编译失败。"
    cd ..
    exit 1
fi

cd ..
echo ""
echo "[Day 36] 编译成功！扩展模块已输出到 build/ 目录。"
echo "[Day 36] 运行演示: python3 python/demo.py"
