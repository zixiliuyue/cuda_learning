#!/usr/bin/env bash
# Day 49: KV Cache 原理与实现 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 49] KV Cache 原理与实现"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 演示 KV Cache 原理（不影响学习）。"
fi
python python/kv_cache.py
