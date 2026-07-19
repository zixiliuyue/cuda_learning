#!/usr/bin/env bash
# Day 71: 流水线并行 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 71] 流水线并行（Pipeline Parallel）演示"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。脚本会自动用 NumPy 演示原理（不影响学习）。"
fi
python python/pipeline_parallel.py
