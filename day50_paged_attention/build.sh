#!/usr/bin/env bash
# Day 50: PagedAttention 简化实现 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 50] PagedAttention（vLLM 核心思想）"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 演示分页管理器（不影响学习）。"
fi
python python/paged_attention.py
