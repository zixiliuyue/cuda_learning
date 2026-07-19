@echo off
chcp 65001 >nul
echo [Day 59] 量化基础：对称/非对称 x per-tensor/per-channel
echo [提示] 有 CUDA 用 GPU，无 CUDA 用 CPU；连 PyTorch 都没有则用 NumPy 演示原理。
python python\quant_basics.py
