#!/usr/bin/env bash
# Day 60: PTQ 训练后量化 + 校准 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 60] PTQ 训练后量化 + 三种激活校准"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。有 GPU 建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 演示三种校准原理（不影响学习）。"
fi
python python/ptq_calibration.py
