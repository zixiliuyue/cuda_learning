@echo off
chcp 65001 >nul
echo [Day 75] 梯度检查点 / 激活重计算（Gradient Checkpointing）演示
echo [提示] 有 PyTorch 时会额外用 torch.utils.checkpoint 做真实对照。
echo         无 PyTorch/GPU 时将用 NumPy 手写演示（不影响学习）。
python python\gradient_checkpoint.py
