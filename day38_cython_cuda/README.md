# Day 38: Cython + CUDA 混合编程

本项目演示如何用 **Cython** 将 CUDA Kernel 封装为可直接 `import` 的 Python 扩展模块，
让 Python 代码以近乎原生的方式调用手写的 GPU 计算函数。

## 学习目标

1. 理解 Cython 的工作原理与编译流水线
2. 掌握在 `.pyx` 中声明并调用外部 C/CUDA 函数的方法
3. 学会用 `setup.py` 把 `.pyx` 与 `.cu` 一起编译成扩展模块
4. 对比 Cython、pybind11、ctypes 三种 Python/C 互操作方案的取舍

## 目录结构

```
day38_cython_cuda/
├── src/
│   └── main.cu              # 纯 CUDA Kernel + extern "C" 包装函数 + 自检 main()
├── python/
│   ├── cuda_wrapper.h       # 供 Cython 声明使用的 C 接口头文件
│   ├── cython_wrapper.pyx   # Cython 封装层(核心)
│   ├── setup.py             # 扩展编译脚本(自定义 nvcc 编译 .cu)
│   └── demo.py              # 调用演示 + 性能对比
├── CMakeLists.txt           # 备用方案：仅编译独立可执行文件验证 Kernel
├── build.bat / build.sh     # 一键编译脚本
└── README.md
```

## Cython 的工作原理

Cython 是 Python 的**超集**：在 Python 语法基础上，允许用 `cdef` 声明 C 级静态类型变量，
并通过 `cdef extern from` 直接调用 C/C++ 函数。它的编译流程是：

```
cython_wrapper.pyx
      │  (1) cython 转译
      ▼
cython_wrapper.cpp   ←── 生成标准 C/C++ 源码(包含 CPython API 调用)
      │  (2) C++ 编译器编译
      │      main.cu ──(nvcc 编译)──► main.obj
      ▼
链接 (cudart) ──► cuda_cython.pyd (Windows) / cuda_cython.so (Linux)
      │  (3) 直接 import
      ▼
import cuda_cython   ←── Python 侧像普通模块一样调用
```

关键点：
- `import numpy` 是**运行期**导入；`cimport numpy` 是**编译期**导入 NumPy 的 C-API。
- `<float*> cnp.PyArray_DATA(arr)` 取得 NumPy 数组底层连续内存指针，零拷贝传给 CUDA。
- `extern "C"` 使 CUDA 包装函数名不被 C++ 修饰(mangling)，链接期能按名匹配。

## 与 pybind11 和 ctypes 的对比

| 特性 | Cython | pybind11 | ctypes |
|------|--------|----------|--------|
| 语言 | Python 超集(.pyx) | 纯 C++ 头文件库 | 纯 Python |
| 编译 | 需要(cython + 编译器) | 需要(C++ 编译器) | 不需要(运行期加载 .dll/.so) |
| 学习曲线 | 中等(接近 Python) | 中等偏上(需 C++) | 低(纯 Python) |
| 调用开销 | 极低(编译为 C) | 极低 | 较高(每次调用有封送开销) |
| 类型转换 | 手动/半自动 | 自动(STL、NumPy 支持好) | 手动(需声明 argtypes) |
| NumPy 集成 | 好(内建 buffer 协议) | 很好(pybind11/numpy.h) | 一般(需 np.ctypeslib) |
| 适用场景 | 加速已有 Python + 封装 C | 现代 C++ 库暴露给 Python | 快速调用现成动态库 |

一句话选型：
- 已有 Python 项目想局部提速、并顺带封装 C/CUDA → **Cython**
- 从零写现代 C++ 库要暴露给 Python → **pybind11**
- 只想快速调一个已编译好的 `.dll/.so`，不想编译 → **ctypes**

## 编译步骤

### 前置依赖

```bash
pip install numpy cython
```
并确保已安装 CUDA Toolkit（`nvcc` 在 PATH 中），MSVC / GCC 可用。

### 方式一：一键脚本（推荐）

Windows：
```powershell
cd day38_cython_cuda
.\build.bat
```

Linux / macOS：
```bash
cd day38_cython_cuda
bash build.sh
```

### 方式二：仅编译 Cython 扩展

```bash
cd day38_cython_cuda/python
python setup.py build_ext --inplace
```
成功后当前目录会生成 `cuda_cython.*.pyd`(Windows) 或 `cuda_cython.*.so`(Linux)。

> 注意：`setup.py` 中 `NVCC_ARCH` 默认 Windows 用 `sm_120`(Blackwell)，
> 其它平台用 `sm_75`(Turing)。如与你的 GPU 不符，请修改为对应架构。

## Python 调用示例

```python
import numpy as np
import cuda_cython

# 向量加法
a = np.random.rand(1 << 20).astype(np.float32)
b = np.random.rand(1 << 20).astype(np.float32)
c = cuda_cython.vector_add(a, b)      # GPU 计算
assert np.allclose(c, a + b, atol=1e-4)

# 矩阵乘法(方阵)
A = np.random.rand(512, 512).astype(np.float32)
B = np.random.rand(512, 512).astype(np.float32)
C = cuda_cython.matmul(A, B)          # GPU 计算
assert np.allclose(C, A @ B, atol=1e-2)
```

运行完整演示（含性能对比表格）：
```bash
cd python
python demo.py
```

## Cython 的优缺点

优点：
- 语法接近 Python，上手成本低于纯 C++ 绑定
- 编译为 C 后调用开销极低，可给热点循环带来数量级提速
- 与 NumPy 深度集成，能直接操作数组底层内存，适合数值计算
- 支持渐进式改造：先跑通 Python，再逐步为关键部分加类型标注

缺点：
- 需要额外的编译步骤和 C/CUDA 工具链，分发比纯 Python 复杂
- 编译期错误信息有时晦涩，调试门槛高于普通 Python
- 手动管理指针/内存时容易出错（段错误），需要小心边界与连续性
- 跨平台构建脚本（尤其牵扯 nvcc）配置相对繁琐，如本项目的自定义 `build_ext`

## 备用方案：独立可执行验证

若暂不编译 Python 扩展，可用 CMake 直接编译 `src/main.cu` 生成可执行文件，
它会独立运行 Kernel 自检（不依赖 Python）：

```powershell
.\build.bat                       # 已包含 CMake 步骤
.\build\Release\cython_cuda_demo.exe
```

---

## 验收标准

完成本天学习后，你应当能够观察到并解释以下现象，才算真正掌握：

1. **独立 Kernel 自检全部通过**：运行 `cython_cuda_demo.exe`，「Kernel 正确性汇总」表格中向量加法（1M 元素）与矩阵乘法（256x256）两项均显示「通过」，程序以 `[主机] 程序正常退出。` 结束，说明裸 CUDA Kernel 逻辑正确。
2. **Cython 扩展编译成功**：执行 `python setup.py build_ext --inplace` 后，`python/` 目录生成 `cuda_cython.*.pyd`（Windows）或 `cuda_cython.*.so`（Linux），`import cuda_cython` 不报错，证明 `.pyx` 转译、`nvcc` 编译 `.cu`、链接 cudart 三步流水线全部走通。
3. **Python 调用结果与 NumPy 一致**：`cuda_cython.vector_add(a, b)` 满足 `np.allclose(c, a+b, atol=1e-4)`，`cuda_cython.matmul(A, B)` 满足 `np.allclose(C, A@B, atol=1e-2)`，说明 `PyArray_DATA` 取指针零拷贝传给 CUDA 的链路正确。
4. **能解释 extern "C" 的必要性**：你能说明如果去掉 `extern "C"`，C++ 名称修饰会导致 Cython 链接期找不到 `vector_add_wrapper` / `matmul_wrapper` 符号。

## 动手挑战

以下任务基于本天 `src/main.cu` 的真实实现，循序渐进地加深理解：

1. **验证 CYTHON_BUILD 宏的隔离作用**：观察 `main()` 被 `#ifndef CYTHON_BUILD` 包裹的写法，理解为什么 Cython 编译时要定义该宏——尝试在 setup.py 中去掉对应定义，看是否会因 `main` 符号重复而链接失败。
2. **新增一个包装函数并暴露到 Cython**：仿照 `vector_add_wrapper`，在 `extern "C"` 块中实现 `vector_scale_wrapper(const float* a, float* out, float s, int n)`，在 `cuda_wrapper.h` 声明、在 `cython_wrapper.pyx` 用 `cdef extern` 引入并写 Python 包装，重新编译后在 `demo.py` 里调用验证。
3. **量化调用开销对比三种方案**：用 `demo.py` 里的计时思路，测量同一 `matmul` 在 Cython、以及 Day36 的 pybind11、Day37 的 ctypes 三种绑定下的往返耗时，结合 README 的对比表，实测验证「Cython/pybind11 调用开销低于 ctypes」的结论。
