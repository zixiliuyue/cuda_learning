@echo off
chcp 65001 >nul
echo [Day 64] 知识蒸馏：teacher -^> student
echo [提示] 有 CUDA 用 GPU，无 CUDA 自动用 CPU，均真实训练蒸馏。
echo         无 PyTorch 时将用 NumPy 演示温度软化机制（不影响学习）。
python python\distillation.py
