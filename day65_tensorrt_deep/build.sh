#!/usr/bin/env bash
# Day 65: TensorRT 深入 — 运行脚本 (Linux / WSL2)
set -e
echo "[Day 65] TensorRT 深入：ONNX -> Engine -> Plugin"
if ! python -c "import onnx" 2>/dev/null; then
    echo "[提示] 未检测到 onnx 包。真实 ONNX 导出需: pip install onnx"
fi
if ! python -c "import tensorrt" 2>/dev/null; then
    echo "[提示] 未检测到 tensorrt。无 TensorRT/GPU 时以原理+骨架讲解（不报错）。"
fi
python python/tensorrt_deep.py
