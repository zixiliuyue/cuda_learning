@echo off
chcp 65001 >nul
echo [Day 46] Triton 接入 PyTorch (autograd.Function) 演示
echo [提示] Triton 官方主要支持 Linux，Windows 建议使用 WSL2 + NVIDIA GPU。
echo         无 Triton/GPU 时将用 PyTorch-CPU autograd 演示梯度校验（不影响学习）。
python python\triton_torch_integration.py
