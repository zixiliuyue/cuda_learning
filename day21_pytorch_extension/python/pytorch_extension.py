#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 21: PyTorch CUDA 扩展 — Python 版本

演示使用 torch.autograd.Function 实现自定义 CUDA 算子，
包括前向传播和反向传播。
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
    print("[提示] 未安装 PyTorch，将使用 NumPy fallback。")
    print("[提示] 安装 PyTorch: pip install torch")

# 常量定义
BATCH_SIZE = 64
INPUT_SIZE = 256
OUTPUT_SIZE = 128
TOLERANCE = 1e-3
ITERATIONS = 10


class CustomLinearFunction(torch.autograd.Function):
    """
    自定义线性层: y = x @ W^T + b
    使用 PyTorch 的 autograd.Function 框架。
    在实际项目中，前向和反向可以用 CUDA Kernel 实现。
    """

    @staticmethod
    def forward(ctx, x, weight, bias):
        """
        前向传播: y = x @ W^T + b
        ctx.save_for_backward: 保存反向传播需要的数据。
        """
        # 保存反向传播需要的张量
        ctx.save_for_backward(x, weight, bias)

        # 计算线性变换
        output = torch.matmul(x, weight.t())
        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播: 计算梯度
        grad_output: 来自上层的梯度
        返回: (grad_x, grad_weight, grad_bias)
        """
        x, weight, bias = ctx.saved_tensors

        # 计算 x 的梯度
        grad_x = torch.matmul(grad_output, weight)

        # 计算 weight 的梯度
        grad_weight = torch.matmul(grad_output.t(), x)

        # 计算 bias 的梯度
        grad_bias = grad_output.sum(dim=0) if bias is not None else None

        return grad_x, grad_weight, grad_bias


def custom_linear_numpy(x_np, w_np, b_np):
    """NumPy 参考实现"""
    return x_np @ w_np.T + b_np


def main():
    print()
    print("=" * 60)
    print("     Day 21: PyTorch CUDA 扩展 — Python 版本                 ")
    print("=" * 60)
    print()

    if not PYTORCH_AVAILABLE:
        print("[提示] 使用 NumPy 模拟演示概念...")
        # NumPy 模拟
        x_np = np.random.randn(BATCH_SIZE, INPUT_SIZE).astype(np.float32)
        w_np = np.random.randn(OUTPUT_SIZE, INPUT_SIZE).astype(np.float32)
        b_np = np.random.randn(OUTPUT_SIZE).astype(np.float32)

        start = time.time()
        for _ in range(ITERATIONS):
            y = custom_linear_numpy(x_np, w_np, b_np)
        cpu_time = (time.time() - start) / ITERATIONS * 1000

        print(f"  NumPy CPU 时间: {cpu_time:.4f} ms")
        print(f"  输出形状: {y.shape}")
        print()
        print("知识点总结:")
        print("  1. torch.autograd.Function: 自定义前向/反向传播")
        print("  2. forward(): 定义前向计算逻辑")
        print("  3. backward(): 定义梯度计算逻辑")
        print("  4. ctx.save_for_backward(): 保存反向传播所需数据")
        print("  5. 实际中: 用 CUDA C++ 或 Triton 实现高性能算子")
        print("  6. PyTorch 扩展: 支持用 CUDA/ATen/Triton 编写")
        return 0

    # 检测 CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[设备] 使用设备: {device}")
    if device.type == 'cuda':
        print(f"[设备] GPU: {torch.cuda.get_device_name(0)}")

    # 创建输入数据
    x = torch.randn(BATCH_SIZE, INPUT_SIZE, device=device, requires_grad=True)
    weight = torch.randn(OUTPUT_SIZE, INPUT_SIZE, device=device, requires_grad=True)
    bias = torch.randn(OUTPUT_SIZE, device=device, requires_grad=True)

    # ---- 自定义算子 ----
    print("[主机] 使用自定义 autograd.Function...")
    start = time.time()
    for _ in range(ITERATIONS):
        y_custom = CustomLinearFunction.apply(x, weight, bias)
    custom_time = (time.time() - start) / ITERATIONS * 1000

    # 反向传播
    grad_output = torch.randn_like(y_custom)
    start = time.time()
    for _ in range(ITERATIONS):
        y_custom = CustomLinearFunction.apply(x, weight, bias)
        y_custom.backward(grad_output, retain_graph=True)
    backward_time = (time.time() - start) / ITERATIONS * 1000

    # ---- PyTorch 内置 nn.Linear ----
    print("[主机] 使用 PyTorch 内置 nn.Linear...")
    linear = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE, device=device)
    linear.weight.data = weight.data.clone()
    linear.bias.data = bias.data.clone()
    x2 = x.detach().clone().requires_grad_(True)

    start = time.time()
    for _ in range(ITERATIONS):
        y_builtin = linear(x2)
    builtin_time = (time.time() - start) / ITERATIONS * 1000

    # 验证前向结果
    y_ref = torch.matmul(x.detach(), weight.t().detach()) + bias.detach()
    forward_diff = torch.max(torch.abs(y_custom.detach() - y_ref)).item()
    forward_ok = forward_diff < TOLERANCE

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  自定义算子前向     : {custom_time:.4f} ms")
    print(f"  自定义算子反向     : {backward_time:.4f} ms")
    print(f"  PyTorch nn.Linear  : {builtin_time:.4f} ms")
    print(f"  前向结果验证       : {'通过' if forward_ok else '失败'} (误差: {forward_diff:.2e})")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. torch.autograd.Function: 自定义算子的基础")
    print("  2. @staticmethod forward/backward: 必须是静态方法")
    print("  3. 实际 CUDA 扩展: 使用 C++/CUDA + Pybind11")
    print("  4. Triton: OpenAI 的 GPU 编程语言，更简单")
    print("  5. torch.compile(): PyTorch 2.x 的 JIT 编译")
    print("  6. 典型场景: Flash Attention, Fused LayerNorm")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
