@echo off
chcp 65001 >nul
echo [Day 87] PyTorch 底层 — 迷你 Autograd 引擎 + Dispatcher 演示
echo [提示] 迷你 autograd 引擎为纯 Python，无 torch 时与手算梯度对拍（不影响学习）。
echo         如需与工业级 autograd 交叉验证：pip install torch
python python\pytorch_internals.py
