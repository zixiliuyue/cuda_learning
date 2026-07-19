# Day 21: PyTorch CUDA 扩展

> **重要说明**：本天的 `src/main.cu` 是**概念演示**——它用纯 CUDA 实现 ReLU / Softmax Kernel，
> 展示"未来可作为 PyTorch 自定义算子内核"的逻辑，但**不链接 PyTorch**。
> 真正与 PyTorch 集成的示例在 [python/pytorch_extension.py](file:///f:/code/cuda_learning/day21_pytorch_extension/python/pytorch_extension.py)，
> 通过 `torch.autograd.Function` 实现带前向/反向传播的自定义线性层。

## 学习目标

1. **实现** 一个 CUDA ReLU Kernel，并与 CPU 版本对比结果、验证正确性（教学演示层）
2. **实现** 基于 `torch.autograd.Function` 的自定义算子，编写 `forward` 与 `backward` 两个静态方法
3. **解释** `ctx.save_for_backward` 如何在前向时保存反向传播所需张量
4. **推导** 线性层 `y = x @ Wᵀ + b` 的三个梯度：`grad_x`、`grad_weight`、`grad_bias`
5. **对比** 自定义算子与 PyTorch 内置 `nn.Linear` 的前向结果与耗时，验证等价性

---

## 前置依赖

- **Day 05（激活函数 / 逐元素 Kernel）**：ReLU 的逐元素并行模式
- **Day 20（性能调优）**：预热 + 多次迭代的计时方法（Python 版沿用）
- 深度学习基础：链式法则、矩阵求导

---

## 核心概念

### 1. 两层演示，各司其职

```
day21/
├── src/main.cu                 ← 概念层：纯 CUDA ReLU/Softmax Kernel，不依赖 PyTorch
│                                  证明"这些 Kernel 就是 PyTorch 算子的底层实现"
└── python/pytorch_extension.py ← 集成层：torch.autograd.Function 自定义线性层
                                   演示 forward/backward 与 autograd 挂钩
```

### 2. 概念层：ReLU / Softmax Kernel（main.cu）

[relu_kernel](file:///f:/code/cuda_learning/day21_pytorch_extension/src/main.cu#L28-L36) 就是 `torch.nn.ReLU` 的底层逻辑：

```cpp
__global__ void relu_kernel(const float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        output[tid] = (input[tid] > 0.0f) ? input[tid] : 0.0f;  // max(0, x)
}
```

[softmax_kernel](file:///f:/code/cuda_learning/day21_pytorch_extension/src/main.cu#L42-L50) 是**简化版**（仅逐元素 `expf`，未做全局归约），源码注释明确标注"完整实现需要全局归约"。程序还会用 [relu_cpu](file:///f:/code/cuda_learning/day21_pytorch_extension/src/main.cu#L52-L58) 做交叉验证（`TOLERANCE = 1e-4`）。

### 3. 集成层：torch.autograd.Function

自定义算子的核心是实现两个静态方法（[CustomLinearFunction](file:///f:/code/cuda_learning/day21_pytorch_extension/python/pytorch_extension.py#L30-L71)）：

```python
class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)   # 保存反向需要的张量
        output = torch.matmul(x, weight.t())      # y = x @ Wᵀ
        if bias is not None:
            output = output + bias                # + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x      = torch.matmul(grad_output, weight)       # ∂L/∂x = g @ W
        grad_weight = torch.matmul(grad_output.t(), x)        # ∂L/∂W = gᵀ @ x
        grad_bias   = grad_output.sum(dim=0)                  # ∂L/∂b = Σ g
        return grad_x, grad_weight, grad_bias
```

### 4. 前向 / 反向的梯度推导

对 `y = x @ Wᵀ + b`，设上游梯度为 `g = ∂L/∂y`：

```
∂L/∂x = g @ W        # 形状与 x 相同 [B, in]
∂L/∂W = gᵀ @ x       # 形状与 W 相同 [out, in]
∂L/∂b = Σ_batch g    # 沿 batch 维求和 [out]
```

`backward` 返回值的顺序必须与 `forward` 的输入参数一一对应（`x, weight, bias`）。

---

## 文件结构

```
day21_pytorch_extension/
├── src/main.cu                  # 概念演示：CUDA ReLU/Softmax Kernel + CPU 验证
├── python/pytorch_extension.py  # 集成示例：torch.autograd.Function 自定义线性层
├── CMakeLists.txt               # 构建配置
├── build.bat                    # Windows 构建脚本（CMake + VS 2022）
├── build.sh                     # Linux/macOS 构建脚本
└── README.md                    # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）—— 概念演示层

方式一：使用 build.bat（内部调用 CMake，产物在 `build\Release\`，可执行文件名 `pytorch_extension_demo.exe`）

```powershell
cd day21_pytorch_extension
.\build.bat
.\build\Release\pytorch_extension_demo.exe
```

方式二：手动 CMake

```powershell
cd day21_pytorch_extension
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\pytorch_extension_demo.exe
```

### Linux / macOS

```bash
cd day21_pytorch_extension
bash build.sh
./build/pytorch_extension_demo
```

### Python 版本（真正的 PyTorch 集成）

```bash
cd day21_pytorch_extension/python
python pytorch_extension.py
```

> Python 版需要 **PyTorch**（`pip install torch`）。若检测到 CUDA，张量会放到 GPU 上运行；
> 若未安装 PyTorch，则自动 fallback 到 NumPy 仅演示前向计算的概念。

---

## 知识点总结

1. **PyTorch 算子的本质**：`nn.ReLU`、`nn.Linear` 等底层就是 CUDA Kernel，main.cu 展示了 ReLU 的内核逻辑。
2. **autograd.Function**：自定义可微算子的标准入口，必须提供 `@staticmethod` 的 `forward`/`backward`。
3. **ctx 上下文**：`ctx.save_for_backward(...)` 前向存、`ctx.saved_tensors` 反向取。
4. **梯度对应关系**：`backward` 返回的梯度个数/顺序要与 `forward` 输入严格对应。
5. **线性层三梯度**：`grad_x = g@W`、`grad_weight = gᵀ@x`、`grad_bias = Σg`。
6. **真实工程路线**：性能算子用 C++/CUDA + `torch.utils.cpp_extension`（或 Triton）编写，再用 `TORCH_LIBRARY` 注册；main.cu 结尾列出了这些集成要点。

---

## 验收标准

- 运行 `pytorch_extension_demo.exe`，看到 `GPU ReLU` 与 `CPU ReLU` 计时，且结果验证 `TOLERANCE = 1e-4` 内通过（返回码 0）。
- Python 版输出 `前向结果验证 : 通过`，说明自定义 `forward` 与 `torch.matmul` 参考值一致（误差 < `1e-3`）。
- Python 版能打印自定义算子的前向、反向耗时以及内置 `nn.Linear` 耗时三行对比。
- 能解释自定义算子的输出与 `nn.Linear` 为何数值一致（同样的 `y = x@Wᵀ + b`）。

---

## 动手挑战

1. **给 ReLU 加反向**：在 Python 版仿照 `CustomLinearFunction` 写一个 `CustomReLUFunction`，`backward` 中对 `x<=0` 的位置置零梯度（`grad_input = grad_output * (x > 0)`）。
2. **真编译一个 CUDA 扩展**：用 `torch.utils.cpp_extension.load(...)` 把 main.cu 的 `relu_kernel` 包成可从 Python 调用的算子，并用 `torch.autograd.gradcheck` 验证。
3. **性能对比**：把 `BATCH_SIZE/INPUT_SIZE/OUTPUT_SIZE` 放大 10 倍，比较自定义算子与 `nn.Linear` 的耗时差距，思考差距来源。

---

## 常见问题

- **Q：为什么 main.cu 不直接写完整的 PyTorch 扩展？**
  A：完整扩展需要 PyTorch 的 ATen 头文件与库，编译环境依赖重。本天让 main.cu 保持纯 CUDA 以便任何有 nvcc 的机器都能编译，把 PyTorch 集成放到 Python 层。

- **Q：`forward`/`backward` 为什么必须是 `@staticmethod`？**
  A：`autograd.Function` 不实例化，PyTorch 通过类直接调用；`apply()` 会在内部构造 `ctx` 并调用这两个静态方法。

- **Q：main.cu 的 softmax 能直接用吗？**
  A：不能当作正确 softmax，它只做了逐元素 `exp`，缺少减最大值稳定化和分母归约，仅用于演示 Kernel 骨架。

---

## 延伸阅读

- [PyTorch — Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html) — 官方 CUDA 扩展教程
- [PyTorch — Extending torch.autograd](https://pytorch.org/docs/stable/notes/extending.html) — autograd.Function 官方文档
