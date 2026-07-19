@echo off
chcp 65001 >nul
echo [Day 84] Checkpoint：同步 / 异步 / 分片 / 增量
echo         本脚本无需 GPU，写系统临时目录并自动清理。
python python\checkpoint.py
