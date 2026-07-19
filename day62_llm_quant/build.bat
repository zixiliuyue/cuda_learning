@echo off
chcp 65001 >nul
echo [Day 62] 大模型量化：GPTQ / AWQ / SmoothQuant 原理
echo [提示] 有 CUDA 用 GPU，无 CUDA 用 CPU；连 PyTorch 都没有则用 NumPy 复现原理。
python python\llm_quant.py
