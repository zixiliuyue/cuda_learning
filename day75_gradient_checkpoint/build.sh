#!/usr/bin/env bash
# Day 75: 梯度检查点 / 激活重计算 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 75] 梯度检查点 / 激活重计算（Gradient Checkpointing）演示"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。脚本会自动用 NumPy 手写演示（不影响学习）。"
fi
python python/gradient_checkpoint.py
