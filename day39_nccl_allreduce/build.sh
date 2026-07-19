#!/bin/bash
echo "[Day 39] 正在编译 NCCL AllReduce ..."
if ! command -v nvcc &> /dev/null; then
    echo "[警告] 未找到 nvcc。macOS 用户请使用 Python 版本 (python/nccl_demo.py)。"
    exit 1
fi
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "[Day 39] 编译成功！运行: ./build/nccl_demo"
echo "[提示] 若安装了 NCCL (libnccl-dev)，将运行真实多 GPU AllReduce；"
echo "       否则自动退化为 CPU 模拟。"
cd ..
