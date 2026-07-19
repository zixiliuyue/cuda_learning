#!/usr/bin/env bash
# Day 86: GPUDirect Storage 原理 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 86] GPUDirect Storage — IO 路径分析（原理为主）"
if ! python -c "import kvikio" 2>/dev/null; then
    echo "[提示] 未检测到 kvikio（GDS Python 入口）。真实 GDS 需 Linux + NVMe + cuFile 驱动。"
    echo "        无 GDS 环境时脚本用带宽模型演示两条路径差异（不影响学习）。"
fi
python python/gpudirect_storage.py
