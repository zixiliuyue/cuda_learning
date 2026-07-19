#!/usr/bin/env bash
# Day 67: 分布式基础 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 67] 分布式基础：进程组与集合通信"
if ! python -c "import torch.distributed" 2>/dev/null; then
    echo "[提示] 未检测到 torch.distributed。脚本会用 numpy 单进程模拟各原语（不影响学习）。"
fi
python python/distributed_basics.py
