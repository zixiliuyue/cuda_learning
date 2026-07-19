#!/usr/bin/env bash
# Day 89: 国产 AI 芯片适配概览 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 89] 国产 AI 芯片适配概览 — 昇腾 / 寒武纪 / 壁仞（原理为主）"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。本天为纯知识梳理，任意环境均可运行（不影响学习）。"
fi
python python/domestic_chips.py
