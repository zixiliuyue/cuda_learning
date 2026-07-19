#!/usr/bin/env bash
# Day 88: Linux 工程化 troubleshooting — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 88] GPU 问题排查 — NaN 钩子 / OOM / compute-sanitizer"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch，将用 NumPy 演示等价诊断逻辑（不影响学习）。"
fi
echo "[进阶] 有 CUDA 工具链时可试：compute-sanitizer --tool memcheck python python/troubleshooting.py"
python python/troubleshooting.py
