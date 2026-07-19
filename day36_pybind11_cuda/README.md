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

---

## 验收标准

完成本天学习后，运行 `python python/demo.py`，你应当能够观察到并解释以下现象，才算真正掌握：

1. **扩展模块可被 import**：编译后 `build/` 目录出现 `cuda_pybind.cp3XX-win_amd64.pyd`（Linux 为 `.so`），`import cuda_pybind` 不报错，且 `cuda_pybind.__version__` 为 `1.0.0`、`cuda_pybind.device_name()` 打印出你的 GPU 名称。
2. **向量加法结果与 NumPy 一致**：`cuda.vector_add(a, b)` 返回的 NumPy 数组与 `a + b` 逐元素相等（误差为 0 或接近机器精度），证明 `py::array_t` 指针取用与 H2D/D2H 拷贝正确。
3. **矩阵乘法与 NumPy 交叉验证通过**：`cuda.matmul(A, B, w).reshape(w, w)` 与 `A.reshape(w,w) @ B.reshape(w,w)` 的最大误差在浮点容差内（float32 累加通常约 1e-3 量级），说明 Tiled + 共享内存的 `matmul_kernel` 计算正确。
4. **错误能被 Python 捕获而非崩溃**：故意传入长度不一致的两个向量时，得到的是 Python `RuntimeError`（可被 `try/except` 捕获），而不是解释器直接退出——验证了 C++ 抛异常自动转 Python 异常的机制。

## 动手挑战

以下任务基于本天 `src/main.cu` 的真实实现，循序渐进地加深理解：

1. **触发并捕获输入校验异常**：在 Python 里分别传入二维数组给 `vector_add`、以及元素数不等于 `width*width` 的数组给 `matmul`，用 `try/except RuntimeError` 捕获并打印 C++ 端抛出的 `[输入错误]` 消息，理解校验分支的价值。
2. **新增一个导出函数**：仿照 `vector_add`，实现并用 `m.def` 导出一个 `vector_scale(a, s)`（`c[i] = a[i] * s`）的 Kernel 与包装函数，重新编译后在 Python 里调用验证，走通「加 Kernel → 加包装 → 加绑定 → 重编译 → import」的完整链路。
3. **量化封装开销**：在 Python 端用 `time.perf_counter` 分别测量 `cuda.matmul` 与纯 NumPy `@` 的耗时，对比不同 `width`（如 256/512/1024）下的差异，思考 H2D/D2H 拷贝在总耗时中的占比与 GPU 加速的收益临界点。
