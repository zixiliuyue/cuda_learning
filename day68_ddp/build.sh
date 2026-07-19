#!/usr/bin/env bash
# Day 68: 数据并行 DDP — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 68] 数据并行 DDP：梯度同步与单卡等价性验证"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。脚本会用 numpy 手推梯度演示等价性（不影响学习）。"
fi
python python/ddp.py
