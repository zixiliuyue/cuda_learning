@echo off
chcp 65001 >nul
echo [Day 78] 通信与计算重叠演示（串行 vs 重叠时间线 + 分桶）
echo         纯 numpy 数值模拟，无 GPU 也能运行。
python python\overlap.py
