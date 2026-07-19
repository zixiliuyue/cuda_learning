#!/usr/bin/env bash
# Day 70: 张量并行 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 70] 张量并行：Column/Row Parallel Linear 与单卡等价性验证"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。脚本会用 numpy 演示切分/通信逻辑（不影响学习）。"
fi
python python/tensor_parallel.py
