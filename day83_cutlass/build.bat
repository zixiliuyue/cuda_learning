@echo off
chcp 65001 >nul
echo [Day 83] CUTLASS 入门：分层 GEMM 模板库
echo         本脚本用 NumPy 复现分层 tiling 概念，无需 GPU。
echo         真实 CUTLASS 需 C++ 编译或 pip install nvidia-cutlass。
python python\cutlass_intro.py
