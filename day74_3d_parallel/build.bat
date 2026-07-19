@echo off
chcp 65001 >nul
echo [Day 74] 混合并行 —— 3D 并行（DP + TP + PP）演示
echo [提示] 本天为纯逻辑推演（rank 坐标映射 + 通信量估算），无需 GPU。
python python\hybrid_parallel.py
