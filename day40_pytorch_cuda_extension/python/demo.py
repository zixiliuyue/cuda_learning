#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 40: PyTorch CUDA Extension 完整演示

演示内容：
  1. ReLU 前向 + 反向（与 torch.relu 对比数值与梯度）
  2. LayerNorm 前向（与 torch.nn.functional.layer_norm 对比）
  3. LayerNorm 反向（梯度对比）
  4. 简单性能对比（自定义 Kernel vs PyTorch 原生）

运行方式：
    cd day40_pytorch_cuda_extension
    python setup.py install          # 先编译扩展
    python python/demo.py            # 再运行本演示

若扩展未编译或没有 CUDA，脚本会自动回退到 PyTorch 原生实现并给出提示。
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("[错误] 未安装 PyTorch，请先执行: pip install torch")
    sys.exit(1)

# 导入封装层（内部会尝试导入编译好的 cuda_custom_ops）
try:
    from custom_ops import relu, layer_norm, HAS_EXT
except ImportError:
    # 兼容从项目根目录运行的情况
    from python.custom_ops import relu, layer_norm, HAS_EXT

if not HAS_EXT:
    print("[提示] 扩展未编译，请运行: python setup.py install")
    print("[提示] 本次演示将使用 PyTorch 原生实现作为 fallback。")


def sync():
    """若在 GPU 上，等待 Kernel 完成，保证计时准确。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def demo_relu(device):
    print("\n" + "=" * 60)
    print("  1. ReLU 前向 / 反向演示")
    print("=" * 60)

    x = torch.randn(4096, 1024, device=device, requires_grad=True)

    # ---- 自定义（若扩展可用）或 fallback 到 torch.relu ----
    if HAS_EXT and device.type == 'cuda':
        y_custom = relu(x)
    else:
        y_custom = torch.relu(x)

    # 反向：构造一个上游梯度并回传
    grad_out = torch.randn_like(y_custom)
    y_custom.backward(grad_out)
    grad_custom = x.grad.clone()

    # ---- PyTorch 原生参考 ----
    x2 = x.detach().clone().requires_grad_(True)
    y_ref = torch.relu(x2)
    y_ref.backward(grad_out)
    grad_ref = x2.grad.clone()

    fwd_diff = (y_custom.detach() - y_ref.detach()).abs().max().item()
    bwd_diff = (grad_custom - grad_ref).abs().max().item()

    print(f"  前向最大误差: {fwd_diff:.2e}  ->  {'通过' if fwd_diff < 1e-4 else '失败'}")
    print(f"  反向最大误差: {bwd_diff:.2e}  ->  {'通过' if bwd_diff < 1e-4 else '失败'}")


def demo_layer_norm(device):
    print("\n" + "=" * 60)
    print("  2 & 3. LayerNorm 前向 / 反向演示")
    print("=" * 60)

    rows, D = 2048, 512
    x = torch.randn(rows, D, device=device, requires_grad=True)
    gamma = torch.randn(D, device=device, requires_grad=True)
    beta = torch.randn(D, device=device, requires_grad=True)
    eps = 1e-5

    # ---- 自定义 或 fallback ----
    if HAS_EXT and device.type == 'cuda':
        y_custom = layer_norm(x, gamma, beta, eps)
    else:
        y_custom = F.layer_norm(x, (D,), gamma, beta, eps)

    grad_out = torch.randn_like(y_custom)
    y_custom.backward(grad_out)
    gx_custom = x.grad.clone()
    gg_custom = gamma.grad.clone()
    gb_custom = beta.grad.clone()

    # ---- PyTorch 原生参考 ----
    x2 = x.detach().clone().requires_grad_(True)
    g2 = gamma.detach().clone().requires_grad_(True)
    b2 = beta.detach().clone().requires_grad_(True)
    y_ref = F.layer_norm(x2, (D,), g2, b2, eps)
    y_ref.backward(grad_out)

    fwd_diff = (y_custom.detach() - y_ref.detach()).abs().max().item()
    gx_diff = (gx_custom - x2.grad).abs().max().item()
    gg_diff = (gg_custom - g2.grad).abs().max().item()
    gb_diff = (gb_custom - b2.grad).abs().max().item()

    print(f"  前向最大误差       : {fwd_diff:.2e}  ->  {'通过' if fwd_diff < 1e-3 else '失败'}")
    print(f"  grad_input 最大误差: {gx_diff:.2e}  ->  {'通过' if gx_diff < 1e-3 else '失败'}")
    print(f"  grad_gamma 最大误差: {gg_diff:.2e}  ->  {'通过' if gg_diff < 1e-3 else '失败'}")
    print(f"  grad_beta  最大误差: {gb_diff:.2e}  ->  {'通过' if gb_diff < 1e-3 else '失败'}")


def demo_perf(device):
    print("\n" + "=" * 60)
    print("  4. 性能对比（ReLU，仅前向，取多次平均）")
    print("=" * 60)

    if not (HAS_EXT and device.type == 'cuda'):
        print("  [跳过] 扩展未编译或非 CUDA 环境，无法进行有意义的对比。")
        return

    x = torch.randn(8192, 8192, device=device)
    iters = 50

    # 预热
    for _ in range(10):
        _ = relu(x)
        _ = torch.relu(x)
    sync()

    start = time.time()
    for _ in range(iters):
        _ = relu(x)
    sync()
    custom_ms = (time.time() - start) / iters * 1000

    start = time.time()
    for _ in range(iters):
        _ = torch.relu(x)
    sync()
    native_ms = (time.time() - start) / iters * 1000

    print(f"  自定义 CUDA ReLU : {custom_ms:.4f} ms")
    print(f"  PyTorch 原生 ReLU: {native_ms:.4f} ms")
    print(f"  比值 (自定义/原生): {custom_ms / native_ms:.2f}x")
    print("  说明: 原生算子经过高度优化，自定义 Kernel 教学为主，接近即合格。")


def main():
    print("\n" + "=" * 60)
    print("     Day 40: PyTorch CUDA Extension 完整实战演示             ")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[设备] 使用: {device}")
    if device.type == 'cuda':
        print(f"[设备] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[扩展] cuda_custom_ops 已编译: {HAS_EXT}")

    demo_relu(device)
    demo_layer_norm(device)
    demo_perf(device)

    print("\n知识点总结:")
    print("  1. torch.autograd.Function: forward/backward 一一对应")
    print("  2. ctx.save_for_backward: 前向保存、反向取用")
    print("  3. backward 返回值个数 = forward 输入个数（不需梯度返回 None）")
    print("  4. setup.py + CUDAExtension + BuildExtension: 编译 .cu 为扩展")
    print("  5. LayerNorm 反向需要整行统计量，故前向保存 mean/rstd 复用")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
