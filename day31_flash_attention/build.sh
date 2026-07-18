#!/bin/bash
# Day 31: Flash Attention — Linux/macOS 编译脚本
echo "[Day 31] 正在编译 Flash Attention ..."
if ! command -v nvcc &> /dev/null; then
    echo "[警告] 未找到 nvcc。macOS 无 NVIDIA GPU 时请使用 Python 版本。"
    exit 1
fi
mkdir -p build
# 直接使用 nvcc 编译：
#   -arch=sm_75      通用较新架构，可按 GPU 修改（如 sm_86 / sm_89 / sm_120）
#   -I common/include 引入跨项目公共头文件
#   -O3              开启优化
nvcc src/main.cu -o build/flash_attention -arch=sm_75 -I common/include -O3
if [ $? -ne 0 ]; then
    echo "[错误] 编译失败！"
    exit 1
fi
echo "[Day 31] 编译成功！运行: ./build/flash_attention"
