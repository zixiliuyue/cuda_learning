#!/usr/bin/env bash
# Day 53: 推理量化 INT8/FP16 — 运行脚本 (Linux / WSL2 / macOS)
set -e
echo "[Day 53] 推理量化 INT8/FP16 演示"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。有 GPU 建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 演示等价原理（不影响学习）。"
fi
python python/inference_quant.py
