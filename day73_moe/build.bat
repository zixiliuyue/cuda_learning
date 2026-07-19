@echo off
chcp 65001 >nul
echo [Day 73] MoE 并行与专家路由（Mixture of Experts）演示
echo [提示] 真实 expert parallel 需多卡 All-to-All，本脚本用单进程模拟专家分布。
echo         无 PyTorch/GPU 时将用 NumPy 演示原理（不影响学习）。
python python\moe.py
