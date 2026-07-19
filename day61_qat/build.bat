@echo off
chcp 65001 >nul
echo [Day 61] QAT 量化感知训练：伪量化 + STE
echo [提示] 有 CUDA 用 GPU，无 CUDA 用 CPU；连 PyTorch 都没有则用 NumPy 演示 STE 原理。
python python\qat.py
