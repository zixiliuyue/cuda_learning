#!/usr/bin/env bash
# Day 47: 阶段小结 Triton vs CUDA vs cuBLAS — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 47] Triton vs CUDA vs cuBLAS 阶段小结"
if ! python -c "import triton" 2>/dev/null; then
    echo "[提示] 未检测到 Triton。如有 NVIDIA GPU，请: pip install triton torch"
    echo "        无 GPU 时脚本会用 CPU/NumPy 计时演示方法论（不影响学习）。"
fi
python python/triton_summary.py
