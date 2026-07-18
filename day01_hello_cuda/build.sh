#!/bin/bash
# =============================================================================
# Day 1: Hello CUDA — macOS / Linux 一键编译脚本
# =============================================================================
# 使用方法: chmod +x build.sh && ./build.sh
# 前提条件: 已安装 CUDA Toolkit（Linux）或仅用于编译验证（macOS）
# =============================================================================

echo "[Day 1] 正在编译 Hello CUDA ..."

# 检查 nvcc 是否可用
if ! command -v nvcc &> /dev/null; then
    echo "[警告] 未找到 nvcc。"
    echo "[提示] Linux: 请安装 CUDA Toolkit"
    echo "[提示] macOS: 本地无法运行 CUDA，但可以使用 Python 版本: python/python/hello_cuda.py"
    exit 1
fi

echo "[Day 1] 检测到 CUDA 编译器:"
nvcc --version

# 创建构建目录
mkdir -p build
cd build

# 使用 CMake 生成项目
echo "[Day 1] 正在生成 CMake 项目 ..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "[错误] CMake 配置失败。"
    exit 1
fi

# 编译项目
echo "[Day 1] 正在编译 ..."
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    echo "[错误] 编译失败。"
    exit 1
fi

echo "[Day 1] 编译成功！"
echo "[Day 1] 可执行文件位置: build/hello_cuda"
echo "[Day 1] 运行命令: ./build/hello_cuda"

cd ..
