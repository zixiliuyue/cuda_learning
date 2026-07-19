#!/usr/bin/env bash
# Day 82: NVLink / RDMA / GPUDirect 原理 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 82] 互联原理：NVLink / RDMA / GPUDirect"
echo "        本脚本无需 GPU；有 GPU 时会探测 P2P 可达性。"
echo "        真实物理拓扑可用: nvidia-smi topo -m"
python python/interconnect.py
