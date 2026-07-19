#!/bin/bash
echo "========================================"
echo "  Speed-Demon-Cuda 构建脚本 (Linux/Mac)"
echo "========================================"
echo ""

# 创建构建目录
mkdir -p build
cd build

# 配置CMake
echo "[1/2] 配置CMake项目..."
cmake ..
if [ $? -ne 0 ]; then
    echo "CMake配置失败！"
    cd ..
    exit 1
fi

# 编译项目
echo ""
echo "[2/2] 编译项目..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "编译失败！"
    cd ..
    exit 1
fi

cd ..
echo ""
echo "========================================"
echo "  构建完成！✅"
echo "========================================"
