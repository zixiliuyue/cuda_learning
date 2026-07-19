@echo off
chcp 65001 >nul
echo [Day 49] KV Cache 原理与实现
echo [提示] PyTorch 在 CPU 也能运行；有 CUDA GPU 时会自动使用 GPU。
echo         未安装 torch 时将用 NumPy 演示 KV Cache 原理（不影响学习）。
python python\kv_cache.py
