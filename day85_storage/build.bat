@echo off
chcp 65001 >nul
echo [Day 85] 存储：并行文件系统 / 小文件问题 / 数据加载流水线
echo         本脚本无需 GPU，用 sleep 模拟 IO/计算耗时。
python python\storage.py
