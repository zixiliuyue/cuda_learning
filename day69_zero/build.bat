@echo off
chcp 65001 >nul
echo [Day 69] ZeRO-1/2/3 显存优化：状态切分与单卡显存对比
echo [提示] 本天为显存原理计算，无需 GPU 即可完整学习。
echo         真实 ZeRO 可 pip install deepspeed 后按脚本中的 ds_config 运行。
python python\zero.py
