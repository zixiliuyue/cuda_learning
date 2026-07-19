#!/usr/bin/env bash
# Day 81: NCCL 通信算法 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 81] NCCL 通信算法：Ring / Tree AllReduce"
echo "        本脚本无需 GPU，用 CPU 模拟 N 个 rank 的收发过程。"
echo "        真实多卡通信可用: torchrun --nproc_per_node=N ..."
python python/nccl_algorithms.py
