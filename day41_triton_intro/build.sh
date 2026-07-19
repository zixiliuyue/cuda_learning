#!/usr/bin/env bash
# Day 41: Triton 入门 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 41] Triton 入门演示"
if ! python -c "import triton" 2>/dev/null; then
    echo "[提示] 未检测到 Triton。如有 NVIDIA GPU，请: pip install triton torch"
    echo "        无 GPU 时脚本会用 NumPy 演示原理（不影响学习）。"
fi
python python/triton_intro.py
