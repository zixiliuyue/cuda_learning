#!/usr/bin/env bash
# Day 76: 认识 Megatron-LM 架构 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 76] Megatron-LM 张量并行演示（单进程模拟多 rank）"
echo "        无 GPU 也能运行：用 numpy 验证 TP 与完整计算数值等价。"
python python/megatron.py
