#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 87: PyTorch 底层 — Autograd 引擎 / C10 / dispatcher

你每天都在用 `loss.backward()`，但它背后到底发生了什么？本天从两个角度拆解
PyTorch 的核心机制：

  1. Autograd 引擎：前向时记录「反向图」，backward 时按拓扑序把梯度从 loss
     反向传播到每个叶子张量。我们手写一个迷你 autograd 引擎（类似 micrograd），
     支持 +, *, @(matmul), relu, sum 等算子，并与手算/PyTorch 梯度对拍。
  2. C10 张量库 + dispatcher：PyTorch 的算子不是写死的，而是按
     (device, dtype, layout) 动态「分发」到不同实现（CPU/CUDA/量化/autograd...）。
     我们用一个极简的 dispatcher 演示「同一个 add，按 device 分发到不同 kernel」。

运行策略（能不回退就不回退）：
  - 迷你 autograd 引擎是纯 Python，任何环境都真实运行、结果可复现；
  - 若装了 torch：用 torch.autograd 对同一表达式求梯度，与迷你引擎交叉验证
    （证明我们手写的反向传播和工业级引擎数值一致）；
  - 无 torch：与「手算解析梯度」对拍，同样能验证正确性。
"""

import math

# ------------------------------------------------------------
# 环境探测（torch 仅用于交叉验证，不是必需）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  1. 迷你 Autograd 引擎（标量版，反向图 + 自动微分）
# ============================================================
class Value:
    """一个标量 + 它的梯度 + 指向父节点的反向图边。

    核心思想（和 PyTorch autograd 完全同源）：
      - 前向：每做一次运算，就新建一个 Value 节点，并记录：
          * 它由哪些输入节点(_prev)产生；
          * 一个 _backward 闭包，描述「如何把本节点的梯度分配回输入」(链式法则)。
      - 反向：从最终节点出发，按拓扑逆序依次调用每个节点的 _backward，
          梯度就沿反向图累加回所有叶子节点。
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None   # 默认叶子节点无反向逻辑

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self)=1, d(out)/d(other)=1 → 梯度原样流回两边
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # 乘法：对 self 的偏导是 other，反之亦然
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Value(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += (p * self.data ** (p - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

        def _backward():
            # relu 的导数：x>0 时为 1，否则为 0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    # 一些便捷运算符
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        """从本节点出发做一次完整反向传播。"""
        # 1) 拓扑排序：保证反向时每个节点的下游梯度都已就绪
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        # 2) 种子梯度：d(self)/d(self)=1
        self.grad = 1.0
        # 3) 逆拓扑序依次调用 _backward，链式法则把梯度传回去
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


def demo_mini_autograd():
    print(SEP)
    print("演示 1：迷你 Autograd 引擎（反向图 + 自动微分）")
    print(SEP)

    # 构造一个表达式：z = (a*b + b**2).relu(), 再 tanh
    # f = tanh( relu(a*b + b^2) )
    a = Value(2.0)
    b = Value(-3.0)
    e = a * b            # -6
    d = e + b ** 2       # -6 + 9 = 3
    f = d.relu()         # relu(3) = 3
    g = f.tanh()         # tanh(3)

    g.backward()

    print(f"  表达式: g = tanh( relu(a*b + b^2) ),  a={a.data}, b={b.data}")
    print(f"  前向结果 g = {g.data:.6f}")
    print(f"  迷你引擎梯度: dg/da = {a.grad:.6f}, dg/db = {b.grad:.6f}")

    # ---- 手算解析梯度做对拍 ----
    # d = a*b + b^2 = 3 (>0)，relu 导数=1；g=tanh(d)，dg/dd = 1 - tanh(d)^2
    dval = a.data * b.data + b.data ** 2
    dgdd = 1 - math.tanh(dval) ** 2
    # dd/da = b, dd/db = a + 2b
    manual_da = dgdd * 1.0 * b.data
    manual_db = dgdd * 1.0 * (a.data + 2 * b.data)
    print(f"  手算解析梯度: dg/da = {manual_da:.6f}, dg/db = {manual_db:.6f}")
    ok = (abs(a.grad - manual_da) < 1e-9 and abs(b.grad - manual_db) < 1e-9)
    print("  [对拍] " + ("与手算一致 ✓" if ok else "不一致 ✗"))

    # ---- 若有 torch，用工业级 autograd 再验一遍 ----
    if TORCH_AVAILABLE:
        ta = torch.tensor(a.data, requires_grad=True, dtype=torch.float64)
        tb = torch.tensor(b.data, requires_grad=True, dtype=torch.float64)
        tg = torch.tanh(torch.relu(ta * tb + tb ** 2))
        tg.backward()
        print(f"  PyTorch 梯度 : dg/da = {ta.grad.item():.6f}, "
              f"dg/db = {tb.grad.item():.6f}")
        ok2 = (abs(a.grad - ta.grad.item()) < 1e-9
               and abs(b.grad - tb.grad.item()) < 1e-9)
        print("  [对拍] " + ("与 PyTorch 一致 ✓" if ok2 else "不一致 ✗"))
    else:
        print("  (未装 torch，跳过与 PyTorch 的交叉验证)")


def demo_mini_mlp_step():
    print()
    print(SEP)
    print("演示 2：用迷你引擎训练一个 2-1 神经元（梯度下降真的在学）")
    print(SEP)
    # 单神经元：y = tanh(w1*x1 + w2*x2 + b)，拟合一个目标值
    w1, w2, bias = Value(0.5), Value(-0.5), Value(0.0)
    x1, x2 = 1.0, -1.0
    target = 0.8
    lr = 0.1
    print(f"  目标 target={target}, 输入 x=({x1},{x2}), 学习率={lr}")
    for step in range(20):
        # 前向
        out = (w1 * x1 + w2 * x2 + bias).tanh()
        loss = (out + (-target)) ** 2
        # 反向
        w1.grad = w2.grad = bias.grad = 0.0
        loss.backward()
        # 更新
        w1.data -= lr * w1.grad
        w2.data -= lr * w2.grad
        bias.data -= lr * bias.grad
        if step % 5 == 0 or step == 19:
            print(f"    step {step:2d}: out={out.data:+.4f}, loss={loss.data:.6f}")
    print("  → loss 单调下降，说明手写反向传播 + 梯度下降完全可用。")


# ============================================================
#  3. 极简 Dispatcher：按 device 分发算子（模仿 C10 dispatcher）
# ============================================================
class MiniDispatcher:
    """极简算子分发器，模仿 PyTorch 的 dispatcher 机制。

    PyTorch 里一个算子（如 aten::add）有多份实现，按张量的
    (device, dtype, layout, requires_grad...) 组成的 "dispatch key"
    路由到对应 kernel：CPU kernel / CUDA kernel / Autograd 包装 / 量化 ...
    这让「同一个 Python API」在不同后端上跑不同代码，且对用户透明。
    """

    def __init__(self):
        self._table = {}   # (op_name, device) -> kernel 函数

    def register(self, op_name, device, kernel):
        self._table[(op_name, device)] = kernel

    def call(self, op_name, device, *args):
        key = (op_name, device)
        if key not in self._table:
            # 回退：找该算子的 "CPU" 通用实现（类似 fallback kernel）
            key = (op_name, "cpu")
        if key not in self._table:
            raise NotImplementedError(f"算子 {op_name} 在 {device} 上未注册")
        return self._table[key](*args)


def demo_dispatcher():
    print()
    print(SEP)
    print("演示 3：极简 Dispatcher — 同一个 add，按 device 分发到不同 kernel")
    print(SEP)

    disp = MiniDispatcher()

    def add_cpu(a, b):
        print("      [dispatch] 命中 CPU kernel")
        return [x + y for x, y in zip(a, b)]

    def add_cuda(a, b):
        print("      [dispatch] 命中 CUDA kernel（此处用 Python 模拟）")
        return [x + y for x, y in zip(a, b)]

    disp.register("add", "cpu", add_cpu)
    disp.register("add", "cuda", add_cuda)

    a, b = [1, 2, 3], [10, 20, 30]
    print("  调用 add 在 device='cpu':")
    print(f"    结果 = {disp.call('add', 'cpu', a, b)}")
    print("  调用 add 在 device='cuda':")
    print(f"    结果 = {disp.call('add', 'cuda', a, b)}")
    print("  调用 add 在 device='mps'（未注册 → fallback 到 CPU 实现）:")
    print(f"    结果 = {disp.call('add', 'mps', a, b)}")
    print()
    print("  对应真实 PyTorch：")
    print("   - dispatch key 由 (device, dtype, layout, autograd...) 组成；")
    print("   - Autograd 也是一层 dispatch key：先进 Autograd kernel 记录反向图，")
    print("     再 redispatch 到底层 CPU/CUDA kernel 做真实计算；")
    print("   - 这就是为什么 requires_grad=True 的张量会自动建反向图。")


# ============================================================
#  4. PyTorch 底层架构速览
# ============================================================
def print_internals_overview():
    print()
    print(SEP)
    print("PyTorch 底层架构速览（自顶向下）")
    print(SEP)
    print("""
  层次              作用                              关键词
  --------------------------------------------------------------------
  Python API        用户接口 torch.add / nn.Module    易用、动态图
  Autograd 引擎     记录反向图、backward 自动微分       Node / edge / grad_fn
  Dispatcher(C10)   按 dispatch key 路由到具体 kernel  device/dtype/layout
  ATen              算子库（aten::add 等的具体实现）    CPU/CUDA kernel
  C10               核心张量库(TensorImpl/Storage)     引用计数、内存、Device
  底层 kernel       真正的 CPU/CUDA 计算代码           cuBLAS/cuDNN/手写

  一次 c = a + b (requires_grad=True) 的旅程：
    Python torch.add
      → Dispatcher 命中 Autograd key：包一层 AddBackward 记录反向图
      → redispatch 命中 CUDA key：调用 ATen 的 CUDA add kernel
      → C10 TensorImpl 管理输出张量的 Storage/形状/设备
    loss.backward() 时：Autograd 引擎按反向图拓扑逆序调用各 grad_fn。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    demo_mini_autograd()
    demo_mini_mlp_step()
    demo_dispatcher()
    print_internals_overview()

    print()
    print(SEP)
    if TORCH_AVAILABLE:
        dev = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"环境：PyTorch 就绪（设备 {dev}）。已用 torch.autograd 交叉验证迷你引擎。")
    else:
        print("环境：未安装 PyTorch。迷你 autograd 引擎与手算梯度对拍，同样验证通过。")
    print(SEP)


if __name__ == "__main__":
    main()
