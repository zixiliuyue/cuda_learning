#!/usr/bin/env bash
# Day 55: 投机解码入门 — 运行脚本 (Linux / WSL2 / macOS)
set -e
echo "[Day 55] 投机解码 Speculative Decoding 演示"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。有 GPU 建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 演示等价原理（不影响学习）。"
fi
python python/speculative_decoding.py
