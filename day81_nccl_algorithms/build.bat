@echo off
chcp 65001 >nul
echo [Day 81] NCCL 通信算法：Ring / Tree AllReduce
echo         本脚本无需 GPU，用 CPU 模拟 N 个 rank 的收发过程。
echo         真实多卡通信需 Linux + 多张 NVIDIA GPU（torchrun + NCCL）。
python python\nccl_algorithms.py
