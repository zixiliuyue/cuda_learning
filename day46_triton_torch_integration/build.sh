#!/usr/bin/env bash
# Day 46: Triton 接入 PyTorch — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 46] Triton 接入 PyTorch (autograd.Function) 演示"
if ! python -c "import triton" 2>/dev/null; then
    echo "[提示] 未检测到 Triton。如有 NVIDIA GPU，请: pip install triton torch"
    echo "        无 GPU 时脚本会用 PyTorch-CPU autograd 演示梯度校验（不影响学习）。"
fi
python python/triton_torch_integration.py
