#!/usr/bin/env bash
# Day 43: Triton 融合算子 Fused Softmax / LayerNorm — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 43] Triton 融合算子 Fused Softmax / LayerNorm 演示"
if ! python -c "import triton" 2>/dev/null; then
    echo "[提示] 未检测到 Triton。如有 NVIDIA GPU，请: pip install triton torch"
    echo "        无 GPU 时脚本会用 NumPy 演示融合原理（不影响学习）。"
fi
python python/triton_fused.py
