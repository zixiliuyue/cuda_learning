@echo off
chcp 65001 >nul
echo [Day 65] TensorRT 深入：ONNX -^> Engine -^> Plugin
echo [提示] 真实 ONNX 导出需 onnx 包；TensorRT 通常未安装。
echo         无 TensorRT/GPU 时以原理+代码骨架讲解完整链路（不报错）。
python python\tensorrt_deep.py
