# Day 46: Triton 接入 PyTorch — torch.autograd.Function / torch.compile

> 🧩 **第九阶段（现代算子编程）**。前几天的 Triton kernel 都是「一次性前向」。要真正用于**训练**，必须能求梯度。本天把 Triton GELU 封装成 `torch.autograd.Function`，实现 forward + backward，并与 PyTorch autograd 做梯度校验。

## 学习目标

1. **掌握** `torch.autograd.Function` 的 `forward` / `backward` 契约与 `ctx.save_for_backward`
2. **实现** Triton GELU 的前向 kernel 与反向 kernel（解析梯度）
3. **做** 梯度校验：与 PyTorch autograd 对比 forward 值和 `.grad`
4. **学会** 把自定义算子封装进 `nn.Module`，并了解如何被 `torch.compile` 优化

---

## 前置依赖

- **Day 41 Triton 入门**：program/BLOCK/mask 编程模型
- **Day 45 Fused MLP**：GELU 的 tanh 近似公式在本天复用
- **基础微积分**：链式法则、tanh 的导数

---

## 核心概念

### autograd.Function 的两段契约

```
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)      # 存下反向要用的张量
        return <Triton 前向 kernel>(x)

    @staticmethod
    def backward(ctx, grad_out):      # grad_out = dL/dy（上游梯度）
        (x,) = ctx.saved_tensors
        return <Triton 反向 kernel>(x, grad_out)   # 返回 dL/dx

# 调用要用 .apply，不是构造实例：
y = MyOp.apply(x)
```

### 链式法则串进计算图

```
   forward:   x ──GELU──▶ y ──...──▶ loss
   backward:  dL/dx ◀──×dGELU/dx── dL/dy ◀──...── dL/dloss=1
                        （本算子的局部导数）
```

`backward` 收到上游 `dL/dy`，乘以本算子局部导数 `dGELU/dx`，得到 `dL/dx` 继续往下传。

### GELU（tanh 近似）的解析梯度

```
令 u = c·(x + 0.044715·x³)，  c = sqrt(2/π)，  t = tanh(u)
GELU(x) = 0.5·x·(1 + t)
dGELU/dx = 0.5·(1 + t) + 0.5·x·(1 - t²)·du/dx
du/dx    = c·(1 + 3·0.044715·x²)
```

反向 kernel 就是把这条公式在片上算出来，只读 `x` 和 `dL/dy`，写回 `dL/dx`。

### forward vs backward kernel 对照

| 项 | 前向 kernel | 反向 kernel |
|----|-------------|-------------|
| 输入 | `x` | `x`、`dL/dy` |
| 输出 | `y = GELU(x)` | `dL/dx` |
| 片上计算 | tanh 近似 | 解析导数 `dGELU/dx` |
| 何时调用 | `forward()` | `backward()`（loss.backward 触发） |

---

## 文件结构

```
day46_triton_torch_integration/
├── python/triton_torch_integration.py   # TritonGELU 前向/反向 + 梯度校验 + nn.Module
├── build.sh                             # Linux/WSL2 运行脚本
├── build.bat                            # Windows 运行脚本（提示用 WSL2）
└── README.md                            # 本文件
```

> Triton 官方主要支持 **Linux**，Windows 用户建议 **WSL2 + NVIDIA GPU**。无 GPU 时脚本用 **PyTorch-CPU autograd** 演示同一套 forward/backward 契约与梯度校验（逻辑完全一致）。

---

## 编译与运行

### Linux / WSL2（推荐，有 NVIDIA GPU）

```bash
cd day46_triton_torch_integration
pip install triton torch
python python/triton_torch_integration.py
```

### Windows

```powershell
cd day46_triton_torch_integration
.\build.bat
```

> 脚本会检测环境：有 GPU+Triton 跑真实前向/反向 kernel；否则用 PyTorch-CPU autograd 演示梯度校验（不伪造结果）。

---

## 知识点总结

1. **`forward(ctx, x)`**：跑前向 kernel，用 `ctx.save_for_backward` 存反向依赖。
2. **`backward(ctx, grad_out)`**：`grad_out` 是 `dL/dy`，返回 `dL/dx`。
3. **`.apply` 调用**：`MyOp.apply(x)` 而非 `MyOp()(x)`。
4. **梯度校验**：对比 PyTorch autograd 的 forward 值与 `.grad`，两者都应一致。
5. **nn.Module 接入**：在 `Module.forward` 里调 `MyOp.apply`，即可替换标准算子。
6. **torch.compile**：自定义 Function 可被计算图捕获并进一步融合优化。

---

## 验收标准

- 程序打印 autograd.Function 的 forward/backward 契约讲解，你能复述链式法则如何串进计算图。
- 有 GPU 时：前向误差 < 1e-3 且梯度误差 < 1e-3，显示「前向 + 梯度校验通过 ✓」。
- 无 GPU 时：用 PyTorch-CPU autograd 完成同样的梯度校验并通过。
- 你能手推 GELU（tanh 近似）的解析梯度，并对应到反向 kernel 的代码。

---

## 动手挑战

1. **换算子**：把 GELU 换成 Softmax，实现其 forward/backward（提示：Softmax 反向需要 `y` 而非 `x`，用 `ctx.save_for_backward(y)`）。
2. **数值梯度检查**：用 `torch.autograd.gradcheck`（需 `double` 精度）验证你的 backward 是否正确。
3. **torch.compile**：把 `TritonGELUModule` 用 `torch.compile` 包一层，观察是否能正常运行并加速。
4. **多输入**：写一个 `y = a*x + b` 的 Function，backward 返回对 `x`、`a`、`b` 三个梯度，体会多输入的返回顺序。

---

## 常见问题

- **Q：backward 返回值个数？**
  A：必须与 forward 的**可微输入个数**一一对应；不可微输入（如整型标量）对应位置返回 `None`。

- **Q：为什么要 `save_for_backward` 而不是直接存成员变量？**
  A：`save_for_backward` 与 PyTorch 的版本计数/内存管理协同，能正确处理原地修改与释放，避免隐蔽 bug。

- **Q：无 GPU 时的演示可信吗？**
  A：可信。无 GPU 分支用的是与 Triton 反向 kernel **完全相同的解析梯度公式**，只是执行在 CPU 上，逻辑与校验流程一致。

---

## 延伸阅读

- [PyTorch — Extending torch.autograd](https://pytorch.org/docs/stable/notes/extending.html)
- [Triton 官方教程 — Fused Softmax（含反向）](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [torch.compile 简介](https://pytorch.org/docs/stable/torch.compiler.html)
