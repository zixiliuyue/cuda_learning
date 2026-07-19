@echo off
chcp 65001 >nul
echo [Day 63] 剪枝：结构化 vs 非结构化
echo [提示] 有 CUDA 用 GPU，无 CUDA 自动用 CPU，均真实训练/剪枝/微调。
echo         无 PyTorch 时将用 NumPy 演示核心原理（不影响学习）。
python python\pruning.py
