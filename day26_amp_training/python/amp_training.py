#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 26: 混合精度训练 — Python 版本

演示混合精度训练（AMP, Automatic Mixed Precision）的概念。
PyTorch 的 torch.cuda.amp 提供自动混合精度支持。
不可用时用 NumPy 模拟 FP16/FP32 的精度差异。
"""

import sys
import time
import numpy as np

# 尝试导入 PyTorch
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("[提示] 未安装 PyTorch，将使用 NumPy 概念演示。")
    print("[提示] 安装 PyTorch: pip install torch")

# 常量定义
BATCH_SIZE = 64
INPUT_DIM = 256
HIDDEN_DIM = 512
OUTPUT_DIM = 10
ITERATIONS = 50
TOLERANCE = 0.1


def simulate_layer_numpy(x, weight, bias, use_fp16=False):
    """NumPy 模拟矩阵乘法层（FP32 或 FP16）"""
    if use_fp16:
        x = x.astype(np.float16)
        weight = weight.astype(np.float16)
        bias = bias.astype(np.float16)

    output = x @ weight.T + bias

    return output.astype(np.float32)


def simulate_amp_training_numpy(num_iterations=ITERATIONS):
    """NumPy 模拟混合精度训练"""
    # 初始化参数
    w1 = np.random.randn(HIDDEN_DIM, INPUT_DIM).astype(np.float32) * 0.01
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    w2 = np.random.randn(OUTPUT_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
    b2 = np.zeros(OUTPUT_DIM, dtype=np.float32)

    # FP32 训练
    fp32_losses = []
    start = time.time()
    for i in range(num_iterations):
        x = np.random.randn(BATCH_SIZE, INPUT_DIM).astype(np.float32)
        # 前向传播 (FP32)
        h = x @ w1.T + b1
        h = np.maximum(0, h)  # ReLU
        y = h @ w2.T + b2
        loss = np.mean(y**2)
        fp32_losses.append(loss)
    fp32_time = (time.time() - start) * 1000

    # FP16 模拟训练
    fp16_losses = []
    start = time.time()
    for i in range(num_iterations):
        x = np.random.randn(BATCH_SIZE, INPUT_DIM).astype(np.float32)
        # 前向传播 (FP16)
        h = simulate_layer_numpy(x, w1, b1, use_fp16=True)
        h = np.maximum(0, h.astype(np.float16))
        y = simulate_layer_numpy(h.astype(np.float32), w2, b2, use_fp16=True)
        loss = np.mean(y**2)
        fp16_losses.append(loss)
    fp16_time = (time.time() - start) * 1000

    return fp32_time, fp16_time, fp32_losses, fp16_losses


def main():
    print()
    print("=" * 60)
    print("     Day 26: 混合精度训练 — Python 版本                      ")
    print("=" * 60)
    print()

    if not PYTORCH_AVAILABLE:
        print("[设备] 使用 NumPy 模拟混合精度训练概念")
        print()

        fp32_time, fp16_time, fp32_losses, fp16_losses = simulate_amp_training_numpy()

        print("[主机] 模拟训练完成")
        print()
        print("=" * 60)
        print("                        性能总结                            ")
        print("=" * 60)
        print(f"  FP32 训练时间 : {fp32_time:.4f} ms ({ITERATIONS} iterations)")
        print(f"  FP16 训练时间 : {fp16_time:.4f} ms ({ITERATIONS} iterations)")
        if fp16_time > 0:
            print(f"  FP16 vs FP32  : {fp32_time / fp16_time:.2f}x 速度")
        print(f"  FP32 最终 loss : {fp32_losses[-1]:.6f}")
        print(f"  FP16 最终 loss : {fp16_losses[-1]:.6f}")
        print("=" * 60)
        print()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[设备] 使用设备: {device}")

        if device.type == 'cuda':
            print(f"[设备] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[设备] 是否支持 FP16: {torch.cuda.is_bf16_supported() if hasattr(torch.cuda, 'is_bf16_supported') else 'N/A'}")

        # 创建模型
        model = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        ).to(device)

        # FP32 训练
        optimizer_fp32 = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        fp32_losses = []
        start = time.time()
        for _ in range(ITERATIONS):
            x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
            y = torch.randn(BATCH_SIZE, OUTPUT_DIM, device=device)
            loss = criterion(model(x), y)
            fp32_losses.append(loss.item())
            optimizer_fp32.zero_grad()
            loss.backward()
            optimizer_fp32.step()
        fp32_time = (time.time() - start) * 1000

        # AMP (混合精度) 训练
        if device.type == 'cuda' and hasattr(torch.cuda.amp, 'autocast'):
            model_amp = torch.nn.Sequential(
                torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
            ).to(device)
            optimizer_amp = torch.optim.SGD(model_amp.parameters(), lr=0.01)

            scaler = torch.cuda.amp.GradScaler()
            amp_losses = []

            start = time.time()
            for _ in range(ITERATIONS):
                x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
                y = torch.randn(BATCH_SIZE, OUTPUT_DIM, device=device)

                with torch.cuda.amp.autocast():
                    loss = criterion(model_amp(x), y)
                amp_losses.append(loss.item())

                optimizer_amp.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer_amp)
                scaler.update()
            amp_time = (time.time() - start) * 1000

            print()
            print("=" * 60)
            print("                        性能总结                            ")
            print("=" * 60)
            print(f"  FP32 训练时间 : {fp32_time:.4f} ms ({ITERATIONS} iterations)")
            print(f"  AMP 训练时间  : {amp_time:.4f} ms ({ITERATIONS} iterations)")
            if amp_time > 0:
                print(f"  AMP vs FP32   : {fp32_time / amp_time:.2f}x 速度")
            print(f"  FP32 最终 loss : {fp32_losses[-1]:.6f}")
            print(f"  AMP 最终 loss  : {amp_losses[-1]:.6f}")
            print("=" * 60)
            print()
        else:
            print()
            print("[提示] AMP 需要 CUDA GPU")
            print(f"  FP32 训练时间: {fp32_time:.4f} ms")
            print()

    print("知识点总结:")
    print("  1. FP32: 标准 32 位浮点，精度高但内存和计算开销大")
    print("  2. FP16: 16 位浮点，速度翻倍，显存减半")
    print("  3. BF16: Brain Float 16，范围更大（Tensor Core 优化）")
    print("  4. AMP: 自动混合精度，关键操作用 FP32，其余用 FP16")
    print("  5. Loss Scaling: 防止 FP16 梯度下溢")
    print("  6. PyTorch: torch.cuda.amp.autocast() + GradScaler()")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
