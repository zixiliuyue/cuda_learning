#!/usr/bin/env bash
# Day 54: 算子融合在推理中的应用 — 运行脚本 (Linux / WSL2 / macOS)
set -e
echo "[Day 54] 算子融合推理演示"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。有 GPU 建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 演示等价原理（不影响学习）。"
fi
python python/fused_inference.py
