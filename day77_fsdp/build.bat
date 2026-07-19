@echo off
chcp 65001 >nul
echo [Day 77] FSDP 全分片数据并行演示（DDP vs FSDP 显存账本）
echo         无 GPU 也能运行：用 numpy 单进程模拟分片与参数生命周期。
python python\fsdp.py
