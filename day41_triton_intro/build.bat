@echo off
chcp 65001 >nul
echo [Day 41] Triton 入门演示
echo [提示] Triton 官方主要支持 Linux，Windows 建议使用 WSL2 + NVIDIA GPU。
echo         无 Triton/GPU 时将用 NumPy 演示原理（不影响学习）。
python python\triton_intro.py
