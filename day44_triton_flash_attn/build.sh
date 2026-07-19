#!/usr/bin/env bash
# Day 44: Triton Flash Attention — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 44] Triton Flash Attention 演示"
if ! python -c "import triton" 2>/dev/null; then
    echo "[提示] 未检测到 Triton。如有 NVIDIA GPU，请: pip install triton torch"
    echo "        无 GPU 时脚本会用 NumPy 演示 tiled + online softmax 原理（不影响学习）。"
fi
python python/triton_flash_attn.py
