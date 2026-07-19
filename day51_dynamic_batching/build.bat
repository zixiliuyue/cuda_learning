@echo off
chcp 65001 >nul
echo [Day 51] Continuous / Dynamic Batching（连续/动态批处理）
echo [提示] 本天为调度事件模拟，纯 Python，有无 torch/GPU 均可运行。
python python\dynamic_batching.py
