@echo off
chcp 65001 >nul
echo [Day 60] PTQ 训练后量化 + 三种激活校准
echo [提示] 有 CUDA 用 GPU，无 CUDA 用 CPU；连 PyTorch 都没有则用 NumPy 演示原理。
python python\ptq_calibration.py
