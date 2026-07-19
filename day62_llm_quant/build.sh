#!/usr/bin/env bash
# Day 62: 大模型量化 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 62] 大模型量化：GPTQ / AWQ / SmoothQuant 原理"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。有 GPU 建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 复现 SmoothQuant/AWQ 数学（不影响学习）。"
fi
python python/llm_quant.py
