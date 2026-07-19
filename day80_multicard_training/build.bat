@echo off
chcp 65001 >nul
echo [Day 80] 第十二阶段收官：单进程模拟多 rank DDP 端到端训练 + 扩展性分析
echo         有 torch 用 autograd，无则 numpy；无 GPU 也能稳定运行。
python python\multicard_training.py
