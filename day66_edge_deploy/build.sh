#!/usr/bin/env bash
# Day 66: 端侧部署实战 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 66] 端侧部署实战：FP16/INT8 + 精度验证（第十一阶段收官）"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。建议: pip install torch"
    echo "        无 torch 时脚本会用 NumPy 演示 INT8 量化原理（不影响学习）。"
fi
python python/edge_deploy.py
