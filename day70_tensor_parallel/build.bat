@echo off
chcp 65001 >nul
echo [Day 70] 张量并行：Column/Row Parallel Linear 与单卡等价性验证
echo [提示] 本天用单进程模拟 2 个 rank，验证张量并行结果 == 单卡完整 Linear。
echo         真实张量并行请在 Linux/WSL2 用 torchrun 运行脚本末尾骨架。
python python\tensor_parallel.py
