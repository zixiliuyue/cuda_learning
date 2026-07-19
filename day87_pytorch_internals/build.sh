#!/usr/bin/env bash
# Day 87: PyTorch 底层 Autograd / C10 / dispatcher — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 87] PyTorch 底层 — 迷你 Autograd 引擎 + Dispatcher 演示"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。迷你 autograd 引擎为纯 Python，会与手算梯度对拍（不影响学习）。"
    echo "        如需与工业级 autograd 交叉验证：pip install torch"
fi
python python/pytorch_internals.py
