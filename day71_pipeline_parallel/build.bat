@echo off
chcp 65001 >nul
echo [Day 71] 流水线并行（Pipeline Parallel）演示
echo [提示] 真实多 stage 流水线需多卡，本脚本用单进程模拟调度与前向。
echo         无 PyTorch/GPU 时将用 NumPy 演示原理（不影响学习）。
python python\pipeline_parallel.py
