@echo off
chcp 65001 >nul
echo [Day 50] PagedAttention（vLLM 核心思想）
echo [提示] PyTorch 在 CPU 也能运行；有 CUDA GPU 时物理块池分配在 GPU 显存。
echo         未安装 torch 时将用 NumPy 演示分页管理器（不影响学习）。
python python\paged_attention.py
