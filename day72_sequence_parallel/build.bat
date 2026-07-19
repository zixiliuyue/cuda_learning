@echo off
chcp 65001 >nul
echo [Day 72] 序列并行 / 上下文并行（Sequence / Context Parallel）演示
echo [提示] 真实序列并行需多卡，本脚本用单进程模拟 AllGather / Ring 通信。
echo         无 PyTorch/GPU 时将用 NumPy 演示原理（不影响学习）。
python python\sequence_parallel.py
