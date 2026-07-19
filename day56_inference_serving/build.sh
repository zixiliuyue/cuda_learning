#!/usr/bin/env bash
# Day 56: 推理服务化 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 56] 推理服务化：并发调度与低延迟"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。有 GPU 建议: pip install torch"
    echo "        无 PyTorch 时脚本会用 NumPy 演示等价原理（不影响学习）。"
fi
python python/inference_serving.py
