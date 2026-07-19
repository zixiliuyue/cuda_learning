@echo off
chcp 65001 >nul
echo [Day 42] Triton 矩阵乘与自动调优演示
echo [提示] Triton 官方主要支持 Linux，Windows 建议使用 WSL2 + NVIDIA GPU。
echo         无 Triton/GPU 时将用 NumPy 演示分块 GEMM 原理（不影响学习）。
python python\triton_matmul.py
