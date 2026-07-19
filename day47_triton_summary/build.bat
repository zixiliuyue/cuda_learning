@echo off
chcp 65001 >nul
echo [Day 47] Triton vs CUDA vs cuBLAS 阶段小结
echo [提示] Triton 官方主要支持 Linux，Windows 建议使用 WSL2 + NVIDIA GPU。
echo         无 Triton/GPU 时将用 CPU/NumPy 计时演示方法论（不影响学习）。
python python\triton_summary.py
