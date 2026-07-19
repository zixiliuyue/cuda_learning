@echo off
chcp 65001 >nul
echo [Day 53] 推理量化 INT8/FP16 演示
echo [提示] 有 CUDA GPU 时在 GPU 上运行，无 GPU 时在 CPU 上运行完整逻辑。
echo         若未安装 PyTorch，将用 NumPy 演示等价原理（不影响学习）。
python python\inference_quant.py
