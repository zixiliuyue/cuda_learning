@echo off
chcp 65001 >nul
echo [Day 48] 从零搭一个 Transformer Decoder 层
echo [提示] PyTorch 在 CPU 也能运行；有 CUDA GPU 时会自动使用 GPU 并统计显存/耗时。
echo         未安装 torch 时将用 NumPy 演示注意力核心公式（不影响学习）。
python python\transformer_decoder.py
