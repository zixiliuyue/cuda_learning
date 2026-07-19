#!/usr/bin/env bash
# Day 63: 剪枝 — 结构化 vs 非结构化 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 63] 剪枝：结构化 vs 非结构化"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 演示非结构化剪枝原理（不影响学习）。"
fi
python python/pruning.py
