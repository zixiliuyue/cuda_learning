#!/usr/bin/env bash
# Day 58: 迷你 LLM 推理引擎 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 58] 阶段项目：迷你 LLM 推理引擎（第十阶段收官）"
if ! python -c "import torch" 2>/dev/null; then
    echo "[提示] 未检测到 PyTorch。有 GPU 建议: pip install torch"
    echo "        无 PyTorch 时用 NumPy 等价实现（逻辑不打折）。"
fi
python python/mini_llm_engine.py
