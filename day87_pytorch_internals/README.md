# Day 87: PyTorch 底层 — Autograd 引擎 / C10 / Dispatcher

> 🔬 **第十三阶段（通信·存储·工程化）工程素养专段**。你天天用 `loss.backward()` 和 `torch.add`，本天拆开 PyTorch 的黑盒：**自动微分怎么实现的**、**算子怎么按设备分发的**。手写一个迷你 autograd 引擎，梯度与 PyTorch 对拍一致。

## 学习目标

1. **实现**一个迷你 autograd 引擎（反向图 + 拓扑排序 + 链式法则），支持 `+ * ** relu tanh`
2. **验证**迷你引擎的梯度与手算解析梯度、与 `torch.autograd` 完全一致
3. **理解** C10 dispatcher：算子如何按 `(device, dtype, layout)` 动态分发到不同 kernel
4. **说清**一次 `c = a + b`（`requires_grad=True`）从 Python API 到底层 kernel 的完整旅程

---

## 前置依赖

- **Day 40 PyTorch CUDA Extension**：写过带 forward/backward 的自定义算子，本天从原理上解释「backward 为什么能自动接上」
- **Day 21 PyTorch 扩展 / ATen**：了解 ATen 算子的概念
- **微积分链式法则**：反向传播的数学基础

---

## 核心概念

### Autograd：前向建图，反向传播

每做一次运算就新建一个节点，记录两件事：
1. **它由哪些输入产生**（反向图的边 `_prev`）；
2. **一个 `_backward` 闭包**，描述「如何用链式法则把本节点的梯度分配回输入」。

`backward()` 时：**拓扑排序** → 从 loss 设种子梯度 1.0 → **逆拓扑序**依次调用每个节点的 `_backward`，梯度就沿反向图累加回所有叶子。

```
前向:  a, b ──*──> e ──+──> d ──relu──> f ──tanh──> g
反向:  g.grad=1 ─逆序─> 依次调用 tanh/relu/+/* 的 _backward ─> a.grad, b.grad
```

这与 PyTorch 的 `grad_fn` / autograd 图**完全同源**，只是 PyTorch 作用在张量上、由 C++ 引擎驱动。

### Dispatcher（C10）：同一个算子，多份实现

PyTorch 的 `aten::add` 不是一份代码，而是按 **dispatch key**（由 `device / dtype / layout / requires_grad` 等组成）路由到不同 kernel：

| dispatch key | 命中的实现 |
|--------------|-----------|
| CPU | ATen CPU add kernel |
| CUDA | ATen CUDA add kernel |
| Autograd | 先包一层记录反向图，再 redispatch 到底层 |
| Quantized / Sparse | 对应的专用实现 |

这让「同一个 Python API」在不同后端跑不同代码，且对用户透明——也是国产芯片能「接管」PyTorch 算子的机制（见 Day 89）。

### 一次 `c = a + b` 的旅程

```
torch.add
  → Dispatcher 命中 Autograd key：包 AddBackward 记录反向图
  → redispatch 命中 CUDA key：调用 ATen CUDA add kernel
  → C10 TensorImpl / Storage 管理输出张量的内存/形状/设备
loss.backward() → Autograd 引擎按反向图拓扑逆序调用各 grad_fn
```

---

## 文件结构

```
day87_pytorch_internals/
├── python/pytorch_internals.py   # 迷你 autograd 引擎 + MLP 训练 + 迷你 dispatcher
├── build.sh                      # Linux/WSL2 运行脚本
├── build.bat                     # Windows 运行脚本
└── README.md                     # 本文件
```

---

## 编译与运行

### Linux / WSL2

```bash
cd day87_pytorch_internals
pip install torch     # 可选：用于与工业级 autograd 交叉验证
python python/pytorch_internals.py
```

### Windows

```powershell
cd day87_pytorch_internals
.\build.bat
```

> 迷你引擎是纯 Python，任意环境都能跑。有 torch 时额外与 `torch.autograd` 对拍，证明数值一致。

---

## 知识点总结

1. **反向图**：前向时用闭包记录「梯度如何回流」，是自动微分的核心数据结构。
2. **拓扑排序**：保证反向时每个节点的下游梯度都已累加完毕才计算它。
3. **梯度累加（+=）**：一个节点被多处使用时，梯度要累加而非覆盖（多元链式法则）。
4. **dispatch key**：`(device, dtype, layout, autograd...)` 决定算子路由到哪份 kernel。
5. **Autograd 是一层 dispatch**：`requires_grad=True` 让算子先进 Autograd kernel 建图，再 redispatch 到 CPU/CUDA。
6. **C10 / ATen 分工**：C10 管张量核心（TensorImpl/Storage/Device），ATen 提供算子实现。

---

## 验收标准

- 迷你引擎对 `g = tanh(relu(a*b + b^2))` 的梯度，与手算、与 PyTorch 三者一致（误差 < 1e-9）。
- 迷你 MLP 训练时 loss 单调下降，说明手写反向传播 + 梯度下降真的在学习。
- 你能解释 dispatcher 为什么让「同一个 `add` 在不同 device 上跑不同 kernel」。
- 你能复述 `c = a + b`（`requires_grad=True`）从 Python 到底层 kernel 的调用链。

---

## 动手挑战

1. **加算子**：给 `Value` 增加 `exp` / `log` / `sigmoid`，补全对应的 `_backward`，并与 torch 对拍。
2. **张量化**：把标量 `Value` 扩展成支持 numpy 数组的迷你张量（注意广播时梯度要 sum 回原形状）。
3. **dispatch dtype**：给 `MiniDispatcher` 的 key 增加 `dtype`，演示 `float32` 和 `int8` 分发到不同 kernel。

---

## 常见问题

- **Q：为什么梯度要 `+=` 而不是 `=`？**
  A：同一个节点若被多个下游使用，各条路径的梯度要按链式法则相加，直接覆盖会丢梯度。

- **Q：PyTorch 的 autograd 也是这样递归的吗？**
  A：思想一致，但 PyTorch 用 C++ 引擎按反向图的边（`Node`/`Edge`）迭代执行，性能远高于 Python 递归。

- **Q：dispatcher 和国产芯片有什么关系？**
  A：国产后端（如昇腾）本质是注册一套新的 dispatch key + kernel，让 PyTorch 算子路由到自家硬件（见 Day 89）。

---

## 延伸阅读

- [micrograd（Karpathy 的极简 autograd）](https://github.com/karpathy/micrograd)
- [PyTorch internals（ezyang 博客）](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [PyTorch Dispatcher 官方文档](https://pytorch.org/tutorials/advanced/dispatcher.html)
