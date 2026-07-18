#!/bin/bash
echo "[Day 6] 正在编译 Tiled MatMul ..."
if ! command -v nvcc &> /dev/null; then
    echo "[警告] 未找到 nvcc。macOS 用户请使用 Python 版本。"
    exit 1
fi
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "[Day 6] 编译成功！运行: ./build/tiled_matmul"
cd ..
