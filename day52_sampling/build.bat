@echo off
chcp 65001 >nul
echo [Day 52] 采样与解码策略演示
echo [提示] 有 CUDA GPU 时在 GPU 上运行，无 GPU 时在 CPU 上运行完整逻辑。
echo         若未安装 PyTorch，将用 NumPy 演示等价原理（不影响学习）。
python python\sampling.py
