@echo off
chcp 65001 >nul
echo [Day 56] 推理服务化：并发调度与低延迟
echo [提示] 有 CUDA GPU 会自动用 GPU；无 GPU 用 CPU；无 PyTorch 用 NumPy 演示原理。
python python\inference_serving.py
