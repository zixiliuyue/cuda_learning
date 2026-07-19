@echo off
chcp 65001 >nul
echo [Day 58] 阶段项目：迷你 LLM 推理引擎（第十阶段收官）
echo [提示] 有 CUDA GPU 会自动用 GPU；无 GPU 用 CPU；无 PyTorch 用 NumPy 等价实现。
python python\mini_llm_engine.py
