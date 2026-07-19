#!/usr/bin/env bash
# Day 59: 量化基础 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 59] 量化基础：对称/非对称 × per-tensor/per-channel"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。有 GPU 建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 演示等价原理（不影响学习）。"
fi
python python/quant_basics.py
