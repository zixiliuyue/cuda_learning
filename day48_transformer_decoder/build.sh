#!/usr/bin/env bash
# Day 48: Transformer Decoder 层 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 48] 从零搭一个 Transformer Decoder 层"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 演示注意力核心公式（不影响学习）。"
fi
python python/transformer_decoder.py
