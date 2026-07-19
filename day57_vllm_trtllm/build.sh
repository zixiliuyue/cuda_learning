#!/usr/bin/env bash
# Day 57: vLLM / TensorRT-LLM 架构 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 57] 认识 vLLM / TensorRT-LLM 架构（原理演示）"
if ! python -c "import vllm" 2>/dev/null; then
    echo "[提示] 未检测到 vLLM（正常）。本天以原理演示为主，无需 vLLM。"
    echo "        如需真实体验：Linux + NVIDIA GPU 上 pip install vllm。"
fi
python python/vllm_trtllm.py
