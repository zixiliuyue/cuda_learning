@echo off
chcp 65001 >nul
echo [Day 68] 数据并行 DDP：梯度同步与单卡等价性验证
echo [提示] 本天用单进程模拟多 rank，验证「平均局部梯度 == 单卡大 batch 梯度」。
echo         真实多卡 DDP 请在 Linux/WSL2 用 torchrun 运行脚本末尾骨架。
python python\ddp.py
