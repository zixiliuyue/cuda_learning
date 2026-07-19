#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 40: 自定义 CUDA 算子的 Python 封装层

本模块把编译好的 C++/CUDA 扩展 cuda_custom_ops 包装成：
  1) 继承 torch.autograd.Function 的 ReLUFunction / LayerNormFunction，
     让自定义算子能无缝接入 PyTorch 的自动求导（autograd）系统。
  2) 便捷调用接口 relu() / layer_norm()。

autograd.Function 的核心约定：
  - forward(ctx, *inputs)   : 前向计算，用 ctx.save_for_backward 保存反向所需张量。
  - backward(ctx, *grad_out): 输入是「对 forward 每个输出的梯度」，
                              返回是「对 forward 每个输入的梯度」（顺序、个数一一对应，
                              非张量/不需梯度的输入返回 None）。
"""

import os
import sys

import torch

# 把项目根目录加入路径，便于以脚本方式直接运行时导入
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 尝试导入编译好的 CUDA 扩展；若未编译，HAS_EXT=False，调用方可回退到原生实现
try:
    import cuda_custom_ops
    HAS_EXT = True
except ImportError:
    cuda_custom_ops = None
    HAS_EXT = False


# ============================================================
# 1. ReLU：自定义 autograd.Function
# ============================================================
class ReLUFunction(torch.autograd.Function):
    """自定义 ReLU 算子，前向/反向都调用 CUDA Kernel。"""

    @staticmethod
    def forward(ctx, input):
        # 保存 input，反向传播时用它决定门控（input>0 才让梯度通过）
        ctx.save_for_backward(input)
        return cuda_custom_ops.relu_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        # grad_output 必须连续，才能安全地传给 Kernel
        grad_input = cuda_custom_ops.relu_backward(grad_output.contiguous(), input)
        # forward 只有一个输入 input，故只返回一个梯度
        return grad_input


# ============================================================
# 2. LayerNorm：自定义 autograd.Function
# ============================================================
class LayerNormFunction(torch.autograd.Function):
    """自定义 LayerNorm 算子（对最后一维归一化）。"""

    @staticmethod
    def forward(ctx, input, gamma, beta, eps):
        # 前向 Kernel 返回 (output, mean, rstd)，后两者供反向复用
        output, mean, rstd = cuda_custom_ops.layer_norm_forward(
            input.contiguous(), gamma.contiguous(), beta.contiguous(), eps)
        ctx.save_for_backward(input, gamma, mean, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, mean, rstd = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = cuda_custom_ops.layer_norm_backward(
            grad_output.contiguous(), input, gamma, mean, rstd)
        # forward 的输入是 (input, gamma, beta, eps)，
        # eps 是普通标量、不需要梯度，因此对应位置返回 None
        return grad_input, grad_gamma, grad_beta, None


# ============================================================
# 3. 便捷调用接口
# ============================================================
def relu(input):
    """自定义 ReLU：等价于 torch.relu，但底层是我们的 CUDA Kernel。"""
    return ReLUFunction.apply(input)


def layer_norm(input, gamma, beta, eps=1e-5):
    """自定义 LayerNorm（对最后一维归一化）。

    参数:
        input : [..., D] 张量
        gamma : [D] 缩放
        beta  : [D] 偏移
        eps   : 数值稳定项
    """
    return LayerNormFunction.apply(input, gamma, beta, eps)
