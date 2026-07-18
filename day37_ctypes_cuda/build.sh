#!/bin/bash
# Day 37: 编译 CUDA ctypes 动态库 (libcuda_ctypes.so)
echo "[Day 37] 正在编译 CUDA ctypes 动态库 ..."

# 检查 nvcc 是否可用
if ! command -v nvcc &> /dev/null; then
    echo "[警告] 未找到 nvcc。macOS 无 NVIDIA GPU 用户请在有 GPU 的 Linux 上编译。"
    exit 1
fi

# 创建并进入构建目录
mkdir -p build && cd build

# 配置并编译 Release 版本
cmake .. -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ]; then
    cd ..
    exit 1
fi

cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [ $? -ne 0 ]; then
    cd ..
    exit 1
fi

cd ..
echo ""
echo "[Day 37] 编译成功！"
echo "  动态库位置: build/libcuda_ctypes.so"
echo "  运行演示  : python3 python/demo.py"
