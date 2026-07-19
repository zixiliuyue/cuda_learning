@echo off
chcp 65001 >nul
echo [Day 86] GPUDirect Storage — IO 路径分析（原理为主）
echo [提示] 真实 GPUDirect Storage 需 Linux + 支持的 NVMe/并行文件系统 + cuFile 驱动。
echo         Windows 上以原理讲解 + 带宽模型模拟为主（不影响学习）。
python python\gpudirect_storage.py
