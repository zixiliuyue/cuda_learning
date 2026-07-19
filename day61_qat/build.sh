#!/usr/bin/env bash
# Day 61: QAT 量化感知训练 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 61] QAT 量化感知训练：伪量化 + STE"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。有 GPU 建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 演示 STE 原理（实验 A）。"
fi
python python/qat.py
