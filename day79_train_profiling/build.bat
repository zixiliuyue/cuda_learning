@echo off
chcp 65001 >nul
echo [Day 79] 训练 profiling 演示（分阶段计时 + 瓶颈判定）
echo         有 torch 跑真实小模型，无则 numpy 模拟；无 GPU 也能运行。
python python\train_profiling.py
