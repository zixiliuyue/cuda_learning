@echo off
chcp 65001 >nul
echo [Day 67] 分布式基础：进程组与集合通信
echo [提示] 本天用单进程 numpy 精确模拟多 rank 集合通信，稳定可退出，无需 GPU。
echo         真实多进程请在 Linux/WSL2 用 torchrun 运行脚本中给出的骨架。
python python\distributed_basics.py
