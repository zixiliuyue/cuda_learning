# Day 40: PyTorch CUDA Extension 完整实战

用 C++/CUDA 编写一个**完整**的 PyTorch 自定义算子，走通全流程：

> Kernel 编写 → 编译（setup.py） → Python 调用 → 自动求导（autograd）

本项目实现两个算子：**ReLU**（最简单，适合入门）和 **LayerNorm**（更复杂，涉及行内归约与统计量复用），并各自实现了前向与反向 Kernel。

---

## 目录结构

```
day40_pytorch_cuda_extension/
├── src/
│   ├── relu_forward.cu         # ReLU 前向 Kernel + 模块注册（PYBIND11_MODULE）
│   ├── relu_backward.cu        # ReLU 反向 Kernel
│   ├── layer_norm_forward.cu   # LayerNorm 前向 Kernel（行内归约求 mean/var）
│   ├── layer_norm_backward.cu  # LayerNorm 反向 Kernel（简化版）
│   └── verify_main.cu          # 纯 CUDA 验证程序（不依赖 PyTorch，供 CMake 编译）
├── python/
│   ├── custom_ops.py           # torch.autograd.Function 封装 + 便捷接口
│   └── demo.py                 # 完整演示：前向/反向/数值对比/性能对比
├── setup.py                    # PyTorch 扩展编译脚本（CUDAExtension）
├── CMakeLists.txt              # 备用：只编译 verify_main.cu（验证 Kernel 逻辑）
├── build.bat / build.sh        # 两步构建脚本
└── README.md                   # 本文件
```

### 一份 `.cu`，两种编译目标

每个 Kernel 源文件都分成两段：

- **裸 Kernel**（`__global__` 函数）：不依赖 PyTorch，永远编译。
- **PyTorch 绑定层**（`#ifdef WITH_TORCH ... #endif`）：只有编译扩展时才启用。

这样带来两个好处：

| 编译方式 | 是否定义 `WITH_TORCH` | 产物 | 用途 |
|----------|----------------------|------|------|
| `python setup.py install` | 是（由 setup.py 传入） | `cuda_custom_ops` 扩展 | Python 调用 + 自动求导 |
| `cmake` / `build.bat` 第 1 步 | 否 | `verify_kernels` 可执行文件 | 脱离 PyTorch 验证 Kernel 逻辑 |

> `PYBIND11_MODULE` 只能出现一次，因此统一放在 `relu_forward.cu`，通过前向声明引用其他文件里的封装函数。

---

## PyTorch CUDA Extension 开发流程

1. **写 Kernel**：普通的 `__global__` CUDA 函数（`relu_forward_kernel` 等）。
2. **写绑定函数**：接收 `torch::Tensor`，做检查（CUDA/连续/dtype）、分配输出、算 grid/block、启动 Kernel，返回 `torch::Tensor`。
3. **注册模块**：用 `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` + `m.def("名字", &函数)` 把 C++ 函数暴露给 Python。
4. **编译**：`setup.py` 里用 `CUDAExtension` 声明源文件，`BuildExtension` 负责调用 nvcc。
5. **封装 autograd**：Python 侧继承 `torch.autograd.Function`，把前向/反向 Kernel 串成可求导的算子。
6. **调用**：`ReLUFunction.apply(x)`，即可像普通张量运算一样参与反向传播。

---

## torch.autograd.Function 的工作原理

自定义算子接入自动求导，靠的是 `torch.autograd.Function` 的两个静态方法：

```python
class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)          # 保存反向所需张量
        return cuda_custom_ops.relu_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors          # 取回前向保存的张量
        return cuda_custom_ops.relu_backward(grad_output.contiguous(), input)
```

核心约定：

- `forward(ctx, *inputs)`：正向计算。需要在反向用到的张量，用 `ctx.save_for_backward(...)` 保存。
- `backward(ctx, *grad_outputs)`：**输入**是「对 forward 每个输出的梯度」，**返回**是「对 forward 每个输入的梯度」。
- **个数与顺序严格对应**：forward 有几个输入，backward 就返回几个梯度；不需要梯度的输入（如标量 `eps`）对应位置返回 `None`。
- 用 `.apply(...)` 调用（不要直接调用 `forward`），这样才会被 autograd 记录到计算图中。

### 前向与反向的对应关系

| 算子 | 前向 | 反向 |
|------|------|------|
| ReLU | `y = max(0, x)` | `grad_x = (x > 0) ? grad_y : 0` |
| LayerNorm | `y = (x - mean) / sqrt(var+eps) * γ + β` | `grad_x = rstd·(g - mean(g) - x̂·mean(g·x̂))`，其中 `g = grad_y·γ`，`x̂ = (x-mean)·rstd` |

LayerNorm 反向需要整行的统计量，所以**前向 Kernel 额外输出 `mean` 和 `rstd`**，通过 `ctx.save_for_backward` 传给反向，避免重复计算。

---

## setup.py 配置说明

```python
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='cuda_custom_ops',
    ext_modules=[
        CUDAExtension('cuda_custom_ops', [
            'src/relu_forward.cu', 'src/relu_backward.cu',
            'src/layer_norm_forward.cu', 'src/layer_norm_backward.cu',
        ], extra_compile_args={'cxx': ['-DWITH_TORCH'], 'nvcc': ['-DWITH_TORCH']}),
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

- `CUDAExtension`：自动带上 PyTorch/ATen 头文件与 CUDA 库路径。
- `BuildExtension`：PyTorch 定制的 `build_ext`，会把 `.cu` 交给 `nvcc`。
- `extra_compile_args`：给两端编译器（`cxx` 主机端、`nvcc` 设备端）都传 `-DWITH_TORCH`，开启绑定层。
- 扩展名 `cuda_custom_ops` 与 `PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)` 必须一致（`TORCH_EXTENSION_NAME` 由编译系统自动定义为该名字）。

---

## 编译步骤

### 方式一：两步构建脚本（推荐）

Windows：

```powershell
cd day40_pytorch_cuda_extension
.\build.bat
```

Linux / macOS：

```bash
cd day40_pytorch_cuda_extension
bash build.sh
```

脚本会依次：
1. 用 CMake 编译 `verify_kernels`（验证 Kernel 逻辑，不需要 PyTorch）；
2. 运行 `python setup.py install` 编译 PyTorch 扩展。

### 方式二：分别手动执行

只验证 Kernel 逻辑（不装 PyTorch 也能跑）：

```powershell
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\verify_kernels.exe        # Linux: ./verify_kernels
```

编译并运行 PyTorch 扩展演示：

```powershell
python setup.py install
python python\demo.py
```

---

## 预期输出（demo.py）

```
[设备] 使用: cuda
[扩展] cuda_custom_ops 已编译: True

  1. ReLU 前向 / 反向演示
  前向最大误差: 0.00e+00  ->  通过
  反向最大误差: 0.00e+00  ->  通过

  2 & 3. LayerNorm 前向 / 反向演示
  前向最大误差       : x.xxe-06  ->  通过
  grad_input 最大误差: x.xxe-05  ->  通过
  ...
```

> 若扩展未编译或没有 CUDA，`demo.py` 会自动回退到 PyTorch 原生实现并给出提示，不会报错崩溃。

---

## 常见问题（FAQ）

1. **`ImportError: No module named cuda_custom_ops`**
   扩展还没编译。先执行 `python setup.py install`（或 `build_ext --inplace`）。

2. **编译报找不到 `cl.exe` / nvcc（Windows）**
   请在「x64 Native Tools Command Prompt for VS」中运行，确保 MSVC 与 CUDA 都在 PATH。

3. **`RuntimeError: xxx 必须是 CUDA 张量`**
   传入的张量在 CPU 上。请先 `x = x.cuda()`，且保证 `float32`、`contiguous()`。

4. **CUDA 13 兼容性**
   源码只用 `cuda_runtime` 基础 API 与 ATen 张量接口；`verify_main.cu` 里的设备信息查询用 `cudaDeviceGetAttribute`（见 `common/include/cuda_utils.h`），避免使用 CUDA 13 已移除的 `cudaDeviceProp` 字段。架构列表在 `CMakeLists.txt` 中按 CUDA 版本自动选择（含 Blackwell `sm_120`）。

5. **自定义 Kernel 为什么不一定比 `torch.relu` 快？**
   PyTorch 原生算子经过高度优化（向量化、多算子融合等）。本项目以**讲清扩展开发全流程**为目标，性能接近即合格。

6. **改了 `.cu` 后重新编译不生效？**
   删除 `build/`、`dist/`、`*.egg-info/` 以及 Python 环境里已安装的旧版本，再重新 `python setup.py install`。

---

## 知识点总结

1. 一个 `.cu` 可同时服务「独立验证」与「PyTorch 扩展」，用 `#ifdef WITH_TORCH` 隔离。
2. 绑定函数三件事：**检查输入 → 分配输出 → 启动 Kernel**。
3. `PYBIND11_MODULE` 全项目只出现一次，其余文件用前向声明。
4. `autograd.Function` 的 forward/backward 输入输出**个数、顺序一一对应**。
5. LayerNorm 前向输出 `mean/rstd` 供反向复用，是减少重复计算的常见工程技巧。
6. `setup.py` 用 `CUDAExtension` + `BuildExtension` 即可把 CUDA 源码编成可导入的模块。

---

## 验收标准

完成本天学习后，你应当能够观察到并解释以下现象，才算真正掌握：

1. **裸 Kernel 逻辑独立验证通过**：运行 `verify_kernels`（CMake 产物），四项（ReLU 前向/反向、LayerNorm 前向）均打印「通过」，最后「总体结果: 全部通过」——这说明四个 `.cu` 里的 `__global__` Kernel 在不依赖 PyTorch 的情况下逻辑正确。
2. **PyTorch 扩展成功编译并被识别**：`python setup.py install` 后运行 `python python/demo.py`，打印 `[扩展] cuda_custom_ops 已编译: True`，说明 `CUDAExtension` + `PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)` 的编译链路走通。
3. **前向与 PyTorch 原生结果一致**：demo 中 ReLU「前向最大误差 0.00e+00 -> 通过」、LayerNorm「前向最大误差」在约 1e-6 量级 -> 通过，证明自定义算子与 `torch.relu` / `torch.nn.functional.layer_norm` 数值对齐。
4. **反向梯度通过 autograd 校验**：ReLU、LayerNorm 的反向「grad_input 最大误差」落在容差内并显示「通过」，说明 `torch.autograd.Function` 的 forward/backward 串联正确，自定义算子能真正参与反向传播。

## 动手挑战

以下任务基于本天真实代码（四个 Kernel + `verify_main.cu` + `custom_ops.py`）循序渐进地加深理解：

1. **验证 WITH_TORCH 的双目标机制**：先用 CMake 编译 `verify_kernels`（未定义 `WITH_TORCH`，只含裸 Kernel），再用 `setup.py` 编译扩展（定义 `WITH_TORCH`，含绑定层）。对比两者，说明为什么 `PYBIND11_MODULE` 只放在 `relu_forward.cu` 一处、其余文件用前向声明。
2. **给 LayerNorm 反向补齐 gamma/beta 的梯度**：当前反向重点演示对 `input` 的梯度。参考前向 Kernel 保存的 `mean/rstd`，在 `layer_norm_backward.cu` 中新增对 `gamma`、`beta` 的梯度计算，并在 `custom_ops.py` 的 `backward` 中按输入顺序返回对应梯度（无梯度项返回 `None`），用 `torch.autograd.gradcheck` 验证。
3. **新增一个自定义算子走完整流程**：仿照 ReLU，实现一个 `GELU` 或 `SiLU` 算子的前向/反向 Kernel，在 `relu_forward.cu` 的 `PYBIND11_MODULE` 中 `m.def` 注册，编译后在 `custom_ops.py` 里用 `torch.autograd.Function` 封装，最后在 `demo.py` 与 PyTorch 原生实现做数值对比，走通「写 Kernel → 绑定 → 编译 → autograd 封装 → 验证」全链路。
