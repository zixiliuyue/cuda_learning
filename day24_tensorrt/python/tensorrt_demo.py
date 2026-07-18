#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 24: TensorRT — Python 版本

演示 TensorRT 的概念和工作流程。
TensorRT 是 NVIDIA 的高性能深度学习推理优化器。
不可用时用 NumPy 模拟推理流程。
"""

import sys
import time
import numpy as np

# 尝试导入 TensorRT
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("[提示] 未安装 TensorRT，将使用 NumPy 概念演示。")
    print("[提示] TensorRT 通常随 CUDA Toolkit 安装")

# 常量定义
BATCH_SIZE = 8
INPUT_CHANNELS = 3
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
NUM_CLASSES = 10
ITERATIONS = 20


def simulate_conv2d_numpy(input_tensor, weight, bias, stride=1, padding=1):
    """NumPy 模拟 2D 卷积（简化版，用于演示推理流程）"""
    batch, in_c, h, w = input_tensor.shape
    out_c, _, kh, kw = weight.shape

    # Padding
    if padding > 0:
        padded = np.pad(input_tensor, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    else:
        padded = input_tensor

    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    output = np.zeros((batch, out_c, out_h, out_w), dtype=np.float32)

    # 简化的卷积（只处理第一个 batch 和少量通道用于演示）
    for b in range(min(batch, 2)):
        for oc in range(min(out_c, 4)):
            for oh in range(out_h):
                for ow in range(out_w):
                    s = 0.0
                    for ic in range(min(in_c, 3)):
                        for fh in range(kh):
                            for fw in range(kw):
                                ih = oh * stride + fh
                                iw = ow * stride + fw
                                s += padded[b, ic, ih, iw] * weight[oc, ic, fh, fw]
                    output[b, oc, oh, ow] = s + bias[oc]

    return output


def simulate_reLU_numpy(x):
    """ReLU 激活"""
    return np.maximum(0, x)


def simulate_linear_numpy(x, weight, bias):
    """全连接层"""
    # x: (batch, input_dim), weight: (output_dim, input_dim)
    return x @ weight.T + bias


def simulate_inference_pipeline(input_data):
    """模拟完整的推理流程: Conv -> ReLU -> Conv -> ReLU -> FC"""
    # 模拟第一层卷积
    w1 = np.random.randn(16, INPUT_CHANNELS, 3, 3).astype(np.float32) * 0.1
    b1 = np.random.randn(16).astype(np.float32) * 0.1
    x = simulate_conv2d_numpy(input_data, w1, b1)
    x = simulate_reLU_numpy(x)

    # 模拟第二层卷积
    w2 = np.random.randn(32, 16, 3, 3).astype(np.float32) * 0.1
    b2 = np.random.randn(32).astype(np.float32) * 0.1
    x = simulate_conv2d_numpy(x, w2, b2)
    x = simulate_reLU_numpy(x)

    # 模拟全连接
    x_flat = x.reshape(x.shape[0], -1)
    fc_w = np.random.randn(NUM_CLASSES, x_flat.shape[1]).astype(np.float32) * 0.1
    fc_b = np.random.randn(NUM_CLASSES).astype(np.float32)
    output = simulate_linear_numpy(x_flat, fc_w, fc_b)

    return output


def main():
    print()
    print("=" * 60)
    print("     Day 24: TensorRT — Python 版本                          ")
    print("=" * 60)
    print()

    if TENSORRT_AVAILABLE:
        print(f"[设备] TensorRT 版本: {trt.__version__}")
        logger = trt.Logger(trt.Logger.WARNING)
        print("[设备] TensorRT 已就绪")
        print("[提示] 完整使用需要 .onnx 或 .engine 模型文件")
        print("[提示] 这里展示推理流程概念")
    else:
        print("[设备] 使用 NumPy 模拟推理流程")

    # 创建模拟输入数据
    print(f"[主机] 输入形状: ({BATCH_SIZE}, {INPUT_CHANNELS}, {INPUT_HEIGHT}, {INPUT_WIDTH})")
    input_data = np.random.uniform(0, 1,
                                   size=(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
                                   ).astype(np.float32)

    # 推理流程演示
    print()
    print("[主机] 模拟推理流程...")
    print("  Stage 1: Conv2D (3->16, 3x3, stride=1, pad=1)")
    print("  Stage 2: ReLU")
    print("  Stage 3: Conv2D (16->32, 3x3, stride=1, pad=1)")
    print("  Stage 4: ReLU")
    print("  Stage 5: Flatten")
    print("  Stage 6: Linear (->10 classes)")
    print()

    # 计时
    start = time.time()
    for _ in range(ITERATIONS):
        output = simulate_inference_pipeline(input_data)
    total_time = (time.time() - start) / ITERATIONS * 1000

    print(f"[主机] 输出形状: {output.shape}")
    print(f"[主机] 推理延迟: {total_time:.4f} ms/batch")
    print(f"[主机] 每样本延迟: {total_time/BATCH_SIZE:.4f} ms")

    # 模拟优化对比
    print()
    print("[概念] TensorRT 优化技术:")
    print("  1. 层融合 (Layer Fusion): 合并连续操作")
    print("  2. 精度校准 (Precision Calibration): FP16/INT8 量化")
    print("  3. 内核自动调优 (Kernel Auto-Tuning)")
    print("  4. 动态张量内存 (Dynamic Tensor Memory)")
    print("  5. 多流执行 (Multi-Stream Execution)")

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  模拟推理时间  : {total_time:.4f} ms (batch={BATCH_SIZE})")
    print(f"  输出形状      : {output.shape}")
    if TENSORRT_AVAILABLE:
        print(f"  TensorRT      : 已安装 (版本 {trt.__version__})")
    else:
        print(f"  TensorRT      : 未安装（NumPy 模拟）")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. TensorRT: NVIDIA 推理优化器，降低延迟提高吞吐")
    print("  2. 工作流: ONNX -> TensorRT Engine -> 推理")
    print("  3. trt.Builder: 构建优化引擎")
    print("  4. trt.Executor: 执行推理")
    print("  5. FP16: 2x 速度提升，精度损失极小")
    print("  6. INT8: 4x 速度提升，需要校准数据集")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
