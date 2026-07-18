#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 29: 推理引擎 — Python 版本

演示深度学习推理引擎的概念和工作流程。
包括模型加载、预处理、推理、后处理的完整流程。
使用 PyTorch 演示（不可用时用 NumPy 模拟）。
"""

import sys
import time
import numpy as np

# 尝试导入 PyTorch
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("[提示] 未安装 PyTorch，将使用 NumPy 概念演示。")

# 常量定义
BATCH_SIZE = 32
INPUT_SIZE = 784  # 28x28 (MNIST 风格)
NUM_CLASSES = 10
ITERATIONS = 100
TOLERANCE = 1e-3


class SimpleModel(nn.Module):
    """简单的全连接模型（演示用）"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def simulate_model_numpy(x):
    """NumPy 模拟简单模型推理"""
    w1 = np.random.randn(INPUT_SIZE, 256).astype(np.float32) * 0.01
    b1 = np.zeros(256, dtype=np.float32)
    w2 = np.random.randn(256, 128).astype(np.float32) * 0.01
    b2 = np.zeros(128, dtype=np.float32)
    w3 = np.random.randn(128, NUM_CLASSES).astype(np.float32) * 0.01
    b3 = np.zeros(NUM_CLASSES, dtype=np.float32)

    # 前向传播
    h1 = np.maximum(0, x @ w1 + b1)  # ReLU
    h2 = np.maximum(0, h1 @ w2 + b2)  # ReLU
    out = h2 @ w3 + b3
    return out


def simulate_preprocessing(raw_data):
    """模拟预处理: 归一化 + reshape"""
    # 模拟 MNIST 预处理
    normalized = raw_data / 255.0
    flattened = normalized.reshape(normalized.shape[0], -1)
    return flattened.astype(np.float32)


def simulate_postprocessing(logits):
    """模拟后处理: Softmax + argmax"""
    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    return predictions, confidences, probabilities


def main():
    print()
    print("=" * 60)
    print("     Day 29: 推理引擎 — Python 版本                          ")
    print("=" * 60)
    print()

    # 创建模拟输入
    print(f"[主机] Batch Size: {BATCH_SIZE}")
    print(f"[主机] Input Size: {INPUT_SIZE} (模拟 MNIST 28x28)")

    # 生成模拟原始图像数据
    raw_data = np.random.randint(0, 256, size=(BATCH_SIZE, 28, 28), dtype=np.uint8)

    # ---- 预处理 ----
    print()
    print("[推理流程] Stage 1: 预处理 (归一化 + Flatten)")
    start = time.time()
    for _ in range(ITERATIONS):
        input_tensor = simulate_preprocessing(raw_data)
    preprocess_time = (time.time() - start) / ITERATIONS * 1000
    print(f"  预处理耗时: {preprocess_time:.4f} ms")

    # ---- 推理 ----
    print("[推理流程] Stage 2: 模型推理")
    if PYTORCH_AVAILABLE:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleModel().to(device).eval()
        x_torch = torch.from_numpy(input_tensor).to(device)

        with torch.no_grad():
            # 预热
            _ = model(x_torch)

            start = time.time()
            for _ in range(ITERATIONS):
                logits = model(x_torch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_time = (time.time() - start) / ITERATIONS * 1000

        logits_np = logits.cpu().numpy()
        print(f"  推理耗时  : {infer_time:.4f} ms ({device})")
    else:
        start = time.time()
        for _ in range(ITERATIONS):
            logits_np = simulate_model_numpy(input_tensor)
        infer_time = (time.time() - start) / ITERATIONS * 1000
        print(f"  推理耗时  : {infer_time:.4f} ms (NumPy CPU)")

    # ---- 后处理 ----
    print("[推理流程] Stage 3: 后处理 (Softmax + Argmax)")
    start = time.time()
    for _ in range(ITERATIONS):
        predictions, confidences, probs = simulate_postprocessing(logits_np)
    postprocess_time = (time.time() - start) / ITERATIONS * 1000
    print(f"  后处理耗时: {postprocess_time:.4f} ms")

    # 展示结果
    print()
    print("[推理结果] 前 5 个样本:")
    for i in range(min(5, BATCH_SIZE)):
        print(f"  样本 {i}: 预测类别={predictions[i]}, 置信度={confidences[i]:.4f}")

    # 性能总结
    total_time = preprocess_time + infer_time + postprocess_time
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  预处理              : {preprocess_time:.4f} ms")
    print(f"  模型推理            : {infer_time:.4f} ms")
    print(f"  后处理              : {postprocess_time:.4f} ms")
    print(f"  端到端延迟          : {total_time:.4f} ms")
    print(f"  吞吐量              : {BATCH_SIZE / total_time * 1000:.0f} samples/s")
    print(f"  每样本延迟          : {total_time / BATCH_SIZE * 1000:.2f} us")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 推理流程: 预处理 -> 模型推理 -> 后处理")
    print("  2. 预处理: 归一化、resize、数据格式转换")
    print("  3. 推理: 前向传播（通常占大部分时间）")
    print("  4. 后处理: Softmax、NMS、阈值过滤")
    print("  5. 推理引擎: TensorRT, ONNX Runtime, OpenVINO")
    print("  6. 优化: 量化、蒸馏、剪枝、层融合")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
