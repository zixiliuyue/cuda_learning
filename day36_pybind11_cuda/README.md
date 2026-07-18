# Day 36: pybind11 CUDA 封装调用

## 学习目标

1. 理解 **pybind11** 的作用：把 C++/CUDA 代码封装成 Python 可直接 `import` 的扩展模块
2. 掌握 `PYBIND11_MODULE` 宏导出函数的写法
3. 学会用 `py::array_t<float>` 与 NumPy 数组进行零拷贝互操作
4. 编写主机端包装函数：NumPy 指针 → GPU 计算 → 返回 NumPy 数组
5. 使用 CMake `FetchContent` 自动拉取 pybind11，免去手动安装
6. 对比三种 CUDA + Python 绑定方式（Numba / PyTorch / pybind11）

## pybind11 的工作原理

pybind11 是一个轻量级的头文件库，用于在 C++ 与 Python 之间建立桥梁：

```
┌─────────────┐   pybind11    ┌──────────────┐   nvcc    ┌─────────────┐
│  Python 端   │  <──────────> │  C++ 包装函数  │ <───────> │  CUDA Kernel │
│ import 模块  │   自动类型转换  │ py::array_t   │  编译     │   GPU 执行   │
└─────────────┘               └──────────────┘           └─────────────┘
```

关键点：

- **`PYBIND11_MODULE(cuda_pybind, m)`**：定义一个名为 `cuda_pybind` 的 Python 模块，
  这个名字必须和 CMake 里 `pybind11_add_module(cuda_pybind ...)` 的目标名一致。
- **`m.def("vector_add", &vector_add, ...)`**：把 C++ 函数注册成 Python 可调用对象。
- **`py::array_t<float>`**：pybind11 提供的 NumPy 数组包装类型。通过 `.request()`
  拿到 `buffer_info`（含底层指针、维度、元素数），即可直接读写 NumPy 内存，无需额外拷贝。
- **异常自动转换**：C++ 里 `throw std::runtime_error(...)` 会被自动转成 Python 的
  `RuntimeError`，因此 Python 端可以正常 `try/except`，而不会让解释器崩溃。

编译产物是一个动态库，但带有 Python 扩展的特殊命名：
- Windows：`cuda_pybind.cp3XX-win_amd64.pyd`
- Linux：`cuda_pybind.cpython-3XX-x86_64-linux-gnu.so`

Python 在 `import cuda_pybind` 时会自动识别这些后缀。

## 目录结构

```
day36_pybind11_cuda/
├── src/
│   └── main.cu          # CUDA Kernel + pybind11 模块绑定
├── python/
│   └── demo.py          # Python 调用演示（含 NumPy fallback）
├── CMakeLists.txt       # FetchContent 拉取 pybind11 并编译 .pyd
├── build.bat            # Windows 编译脚本
├── build.sh             # Linux 编译脚本
└── README.md            # 本说明
```

## 编译和安装步骤

### Windows

```powershell
cd day36_pybind11_cuda
.\build.bat
```

### Linux

```bash
cd day36_pybind11_cuda
chmod +x build.sh
./build.sh
```

> 首次编译会通过 `FetchContent` 从 GitHub 克隆 pybind11 v2.13.6，需要联网。
> 编译完成后，扩展模块会输出到 `build/` 目录。

## 如何在 Python 中调用

`demo.py` 已把 `build/` 加入 `sys.path`，因此可直接运行：

```bash
python python/demo.py
```

核心用法示例：

```python
import sys, os
sys.path.append(os.path.join('..', 'build'))
import numpy as np
import cuda_pybind as cuda

# 向量加法
a = np.random.rand(1024).astype(np.float32)
b = np.random.rand(1024).astype(np.float32)
c = cuda.vector_add(a, b)          # 返回 NumPy 数组

# 矩阵乘法（输入按行主序展平，返回展平结果）
w = 512
A = np.random.rand(w * w).astype(np.float32)
B = np.random.rand(w * w).astype(np.float32)
C = cuda.matmul(A, B, w).reshape(w, w)

# 查询设备
print(cuda.device_name())
```

> 注意：传入的 NumPy 数组 **dtype 必须是 `np.float32`**，
> 与 C++ 端的 `float` 对应，否则 pybind11 会自动做一次类型转换拷贝。

如果模块未编译，`demo.py` 会自动退回到纯 NumPy 实现，方便在无 GPU 环境理解接口语义。

## 三种绑定方式对比总结

| 特性 | Numba (`@cuda.jit`) | PyTorch `cpp_extension` | pybind11（本项目） |
|------|---------------------|-------------------------|--------------------|
| 语言 | 纯 Python | C++/CUDA + Python | C++/CUDA + Python |
| 上手难度 | 最低 | 中 | 中 |
| 编译方式 | JIT 即时编译 | setuptools / JIT | CMake / setuptools |
| 数据类型 | NumPy 数组 | torch.Tensor | NumPy 数组（`py::array_t`）|
| 自动求导 | 不支持 | 支持（autograd） | 不支持（需手写） |
| 依赖框架 | Numba | PyTorch | 仅 pybind11（可 FetchContent）|
| 适用场景 | 快速原型、教学 | 深度学习算子 | 通用高性能库封装 |
| 性能上限 | 中（受 JIT 限制） | 高 | 高（原生 CUDA） |

**选型建议：**

- 想快速验证想法、不想写 C++ → **Numba**（见 Day 22）
- 要给 PyTorch 写自定义算子、需要自动求导 → **PyTorch cpp_extension**（见 Day 21）
- 要把 CUDA 库封装给任意 Python 项目用、追求最大灵活性和性能 → **pybind11**（本项目）

## 知识点回顾

1. `PYBIND11_MODULE`：定义可被 Python `import` 的模块
2. `py::array_t<float>`：与 NumPy 数组零拷贝互操作
3. `buffer_info.ptr`：获取 NumPy 底层数据指针，直接喂给 `cudaMemcpy`
4. C++ 抛异常 → 自动转 Python `RuntimeError`，比 `exit()` 更安全
5. `FetchContent`：CMake 自动拉取 pybind11，用户零配置
6. 编译产物 `.pyd` / `.so` 就是普通 Python 模块，可 `pip install` 或直接 `import`
