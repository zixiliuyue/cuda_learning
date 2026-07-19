@echo off
chcp 65001 >nul
echo [Day 88] GPU 问题排查 — NaN 钩子 / OOM / compute-sanitizer
echo [提示] 无 torch 时用 NumPy 演示等价诊断逻辑（不影响学习）。
echo         compute-sanitizer / cuda-gdb / coredump 需 Linux + CUDA 工具链，Windows 建议 WSL2。
python python\troubleshooting.py
