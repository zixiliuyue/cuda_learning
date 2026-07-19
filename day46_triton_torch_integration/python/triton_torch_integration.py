#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 46: Triton 接入 PyTorch — torch.autograd.Function / torch.compile

前几天写的 Triton kernel 都是「一次性前向计算」。要真正用在训练里，
必须能求梯度（backward）。本天把一个 Triton GELU kernel 封装成
`torch.autograd.Function`：

    class TritonGELU(torch.autograd.Function):
        forward(ctx, x)  -> 调 Triton 前向 kernel，保存 x
        backward(ctx, g) -> 调 Triton 反向 kernel，返回 dL/dx

这样它就能像普通 PyTorch 算子一样，被自动微分链条串起来，
也能封装进 `nn.Module`、被 `torch.compile` 优化。

校验方式：拿封装后的算子与「纯 PyTorch 版 GELU」比 forward 结果，
再比 `.grad`（反向梯度），两者都应一致。

运行策略（能不回退就不回退）：
  - 有 Triton + GPU：跑真实 Triton 前向/反向 kernel，与 PyTorch autograd 对比；
  - 无 Triton/GPU：不伪造，用 PyTorch-CPU autograd 演示 Function 的
    forward/backward 契约与梯度校验流程（数学等价，逻辑完全一致）。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（优先真实 GPU，不轻易回退）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TRITON_AVAILABLE = False
if TORCH_AVAILABLE and torch.cuda.is_available():
    try:
        import triton
        import triton.language as tl
        TRITON_AVAILABLE = True
    except ImportError:
        TRITON_AVAILABLE = False

SEP = "=" * 64

# GELU(tanh 近似) 常数：sqrt(2/pi)
_GELU_C = 0.7978845608028654


# ============================================================
#  Triton 前向 / 反向 kernel
# ============================================================
if TRITON_AVAILABLE:

    @triton.jit
    def gelu_fwd_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """前向：y = GELU(x)，tanh 近似。"""
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        c = 0.7978845608028654
        u = c * (x + 0.044715 * x * x * x)
        y = 0.5 * x * (1.0 + tl.math.tanh(u))
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.jit
    def gelu_bwd_kernel(x_ptr, gout_ptr, gin_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """反向：给定 dL/dy，计算 dL/dx = dL/dy * dGELU/dx。

        令 u = c(x + 0.044715 x^3)，t = tanh(u)
          dGELU/dx = 0.5*(1+t) + 0.5*x*(1-t^2)*du/dx
          du/dx    = c*(1 + 3*0.044715*x^2)
        全部在片上完成，只需读 x 与 dL/dy，写回 dL/dx。
        """
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        gout = tl.load(gout_ptr + offs, mask=mask)
        c = 0.7978845608028654
        x2 = x * x
        u = c * (x + 0.044715 * x * x2)
        t = tl.math.tanh(u)
        du_dx = c * (1.0 + 3.0 * 0.044715 * x2)
        dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
        tl.store(gin_ptr + offs, gout * dgelu, mask=mask)

    def _launch(kernel, *tensors, n_elements):
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        kernel[grid](*tensors, n_elements, BLOCK_SIZE=1024)


# ============================================================
#  封装成 torch.autograd.Function
# ============================================================
if TORCH_AVAILABLE:

    class TritonGELU(torch.autograd.Function):
        """把 Triton GELU 接入 PyTorch 自动微分。

        - forward：调用 Triton 前向 kernel（无 GPU 时退化为 torch 等价计算）；
        - backward：调用 Triton 反向 kernel，返回对输入的梯度。
        ctx.save_for_backward 把 forward 的输入 x 存下来供反向使用。
        """

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            if TRITON_AVAILABLE and x.is_cuda:
                y = torch.empty_like(x)
                xc = x.contiguous()
                _launch(gelu_fwd_kernel, xc, y, n_elements=x.numel())
                return y
            # 无 GPU/Triton：用 torch 等价前向（保证契约可跑通、可校验）
            return torch.nn.functional.gelu(x, approximate="tanh")

        @staticmethod
        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            if TRITON_AVAILABLE and x.is_cuda:
                grad_in = torch.empty_like(x)
                _launch(gelu_bwd_kernel, x.contiguous(), grad_out.contiguous(),
                        grad_in, n_elements=x.numel())
                return grad_in
            # 无 GPU/Triton：用解析梯度公式（与 Triton 反向 kernel 完全一致）
            c = _GELU_C
            x2 = x * x
            u = c * (x + 0.044715 * x * x2)
            t = torch.tanh(u)
            du_dx = c * (1.0 + 3.0 * 0.044715 * x2)
            dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
            return grad_out * dgelu

    def triton_gelu(x):
        """函数式入口，等价于一个可微算子。"""
        return TritonGELU.apply(x)

    class TritonGELUModule(torch.nn.Module):
        """演示：把自定义算子封装进 nn.Module，可直接替换 nn.GELU。"""

        def forward(self, x):
            return TritonGELU.apply(x)


# ============================================================
#  无 torch 时的 NumPy 原理演示
# ============================================================
def numpy_gelu(x):
    c = _GELU_C
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def numpy_gelu_grad(x):
    c = _GELU_C
    x2 = x * x
    t = np.tanh(c * (x + 0.044715 * x * x2))
    du_dx = c * (1.0 + 3.0 * 0.044715 * x2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx


# ============================================================
#  讲解
# ============================================================
def explain_autograd_function():
    print(SEP)
    print("torch.autograd.Function：让 Triton kernel 可微、可训练")
    print(SEP)
    print("""
  要把自定义 kernel 接入 PyTorch 训练，需实现两个静态方法：

    class MyOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)   # 存下反向要用的张量
            return <Triton 前向 kernel>(x)

        @staticmethod
        def backward(ctx, grad_out):   # grad_out = dL/dy
            (x,) = ctx.saved_tensors
            return <Triton 反向 kernel>(x, grad_out)   # dL/dx

  使用：  y = MyOp.apply(x)      # 不是 MyOp()！
  接入：  放进 nn.Module.forward 即可当普通层用，可被 torch.compile 优化。

  链式法则：backward 收到上游 dL/dy，乘以本算子的局部导数 dGELU/dx，
  得到 dL/dx 继续往下传。这就是自动微分把你的 kernel「串进」计算图。

  校验：与纯 PyTorch autograd 比 forward 值 + .grad 梯度，两者都应一致。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_autograd_function()

    print(SEP)
    print("演示：TritonGELU 前向 + 反向 + 梯度校验")
    print(SEP)

    if not TORCH_AVAILABLE:
        # 完全没有 torch：用 NumPy 演示前向与解析梯度
        rng = np.random.default_rng(0)
        x = rng.standard_normal(8).astype(np.float32)
        y = numpy_gelu(x)
        g = numpy_gelu_grad(x)
        print("[无 PyTorch] 用 NumPy 演示 GELU 前向与解析梯度：")
        print(f"  x[:4]      = {np.round(x[:4], 4)}")
        print(f"  GELU(x)[:4]= {np.round(y[:4], 4)}")
        print(f"  dGELU/dx[:4]={np.round(g[:4], 4)}")
        print("  说明：Triton 反向 kernel 用的就是这条解析梯度公式。")
        print()
        print(SEP)
        print("环境：未安装 PyTorch。已用 NumPy 演示前向与解析梯度。")
        print(SEP)
        return

    device = "cuda" if TRITON_AVAILABLE else "cpu"
    torch.manual_seed(0)

    # 自定义算子这一路
    x1 = torch.randn(4096, device=device, dtype=torch.float32, requires_grad=True)
    y1 = triton_gelu(x1)
    loss1 = (y1 * y1).sum()
    loss1.backward()

    # PyTorch 参考这一路（同样的输入）
    x2 = x1.detach().clone().requires_grad_(True)
    y2 = torch.nn.functional.gelu(x2, approximate="tanh")
    loss2 = (y2 * y2).sum()
    loss2.backward()

    fwd_err = (y1 - y2).abs().max().item()
    grad_err = (x1.grad - x2.grad).abs().max().item()

    tag = "Triton/GPU" if TRITON_AVAILABLE else "PyTorch-CPU"
    print(f"[{tag}] 前向最大误差       = {fwd_err:.2e}")
    print(f"[{tag}] 反向(梯度)最大误差 = {grad_err:.2e}")
    ok = fwd_err < 1e-3 and grad_err < 1e-3
    print("[结果] " + ("前向 + 梯度校验通过 ✓" if ok else "校验失败 ✗"))

    # nn.Module 接入演示
    mod = TritonGELUModule().to(device)
    xm = torch.randn(16, device=device)
    _ = mod(xm)
    print("[nn.Module] TritonGELUModule 可像 nn.GELU 一样直接调用 ✓")

    print()
    print(SEP)
    if TRITON_AVAILABLE:
        print("环境：Triton + GPU 就绪，已运行真实 Triton 前向/反向 kernel。")
    else:
        print("环境：有 PyTorch 但无 CUDA GPU / 未装 Triton。")
        print("      已用 PyTorch-CPU autograd 演示 Function 的 forward/backward 契约与梯度校验。")
        print("      在 Linux + NVIDIA GPU 上 `pip install triton` 即可切换到真实 kernel。")
    print(SEP)


if __name__ == "__main__":
    main()
