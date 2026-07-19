@echo off
chcp 65001 >nul
echo [Day 66] 端侧部署实战：FP16/INT8 + 精度验证（第十一阶段收官）
echo [提示] 有 CUDA 用 GPU（真半精度），无 CUDA 用 CPU（数值模拟），均真实运行。
echo         无 PyTorch 时将用 NumPy 演示 INT8 量化流程（不影响学习）。
python python\edge_deploy.py
