# Day 37: ctypes 调用 CUDA 动态库

## 学习目标

1. 理解如何把 CUDA Kernel 封装成标准动态库（`.dll` / `.so`）
2. 掌握 `extern "C"` 关闭 C++ 名称修饰的作用与原理
3. 掌握 `__declspec(dllexport)`（Windows）与 `visibility("default")`（Linux）的跨平台导出写法
4. 学会用 Python 标准库 **ctypes** 加载动态库并调用 GPU 函数（零外部依赖）
5. 理解「稳定 C ABI 接口」的设计原则：主机指针进出、返回错误码、调用方管理内存

## 核心原理

### 为什么用 ctypes 调用 CUDA？

常见的 Python 调 GPU 方案（PyCUDA、Numba、CuPy、PyTorch 扩展）都需要安装体积庞大的第三方库。而本项目的思路是：

```
CUDA 源码 (main.cu)
      │  nvcc 编译 + 导出 C 符号
      ▼
动态库 (cuda_ctypes.dll / libcuda_ctypes.so)
      │  Python 标准库 ctypes 加载
      ▼
Python 直接调用 GPU 函数（无需任何第三方 GPU 库）
```

只要目标机器装了 NVIDIA 驱动 + CUDA 运行时，Python 端仅靠内置的 `ctypes` 即可驱动 GPU 计算，非常适合分发和部署。

### `extern "C"` 的作用

C++ 编译器为了支持函数重载，会对函数名做**名称修饰**（name mangling）：

```
C++ 原名:   vector_add_cuda
修饰后(MSVC): ?vector_add_cuda@@YAHPEBM0PEAMH@Z   ← ctypes 无法用原名找到它
```

用 `extern "C"` 包裹后，编译器按 C 规则生成符号，函数名保持原样 `vector_add_cuda`，`ctypes` 才能通过 `lib.vector_add_cuda` 正确定位。

### `__declspec(dllexport)` 与 `visibility("default")`

| 平台 | 导出关键字 | 说明 |
|------|-----------|------|
| Windows (MSVC) | `__declspec(dllexport)` | 把符号写入 DLL 的导出表，否则外部找不到 |
| Linux (GCC/Clang) | `__attribute__((visibility("default")))` | 配合 `-fvisibility=hidden` 显式导出，符号表更干净 |

本项目用宏统一处理：

```c
#ifdef _WIN32
    #define EXPORT_API __declspec(dllexport)
#else
    #define EXPORT_API __attribute__((visibility("default")))
#endif
```

### 接口设计原则

导出的 C 函数遵循「稳定 ABI」的最佳实践：

- **只传主机指针 + 长度**：GPU 内存分配/拷贝/释放全部在库内部完成，对调用方透明。
- **返回错误码而非抛异常**：`0=成功, -1=失败`，跨语言最稳妥。
- **绝不在库里 `exit()`**：出错只打印并返回，避免连带杀死 Python 解释器。
- **缓冲区由调用方分配**（`get_device_info`）：避免跨模块 `malloc/free` 不匹配。

## 导出的 API

```c
extern "C" {
    // GPU 向量加法 out = a + b，返回 0=成功 / -1=失败
    EXPORT_API int  vector_add_cuda(const float* a, const float* b, float* out, int n);

    // GPU 方阵乘法 C = A * B（width x width）
    EXPORT_API int  matmul_cuda(const float* a, const float* b, float* c, int width);

    // 查询设备信息，写入调用方提供的缓冲区
    EXPORT_API int  get_device_info(char* buffer, int buffer_size);

    // 释放由本库分配的 GPU 内存（预留接口）
    EXPORT_API void free_cuda_mem(void* ptr);
}
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `src/main.cu` | CUDA Kernel + `extern "C"` 导出的 C 风格 API |
| `python/demo.py` | 用 ctypes 加载动态库并调用的演示脚本 |
| `CMakeLists.txt` | 编译为共享库（SHARED）的构建配置 |
| `build.bat` | Windows 一键编译 |
| `build.sh` | Linux 一键编译 |

## 编译与运行

### Windows (有 NVIDIA GPU)

```powershell
cd day37_ctypes_cuda
.\build.bat
python .\python\demo.py
```

### Linux

```bash
cd day37_ctypes_cuda
chmod +x build.sh
./build.sh
python3 python/demo.py
```

编译产物：
- Windows: `build\Release\cuda_ctypes.dll`（或 `build\cuda_ctypes.dll`）
- Linux: `build/libcuda_ctypes.so`

`demo.py` 会自动在 `build/`、`build/Release/`、`build/Debug/` 中探测动态库。

## Python 调用要点

```python
import ctypes
import numpy as np

# 1) 加载动态库
lib = ctypes.CDLL("build/cuda_ctypes.dll")   # Linux 换成 libcuda_ctypes.so

# 2) 声明参数/返回类型（关键！否则 64 位指针会被截断）
f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
lib.vector_add_cuda.argtypes = [f32, f32, f32, ctypes.c_int]
lib.vector_add_cuda.restype = ctypes.c_int

# 3) 准备连续的 float32 数组
n = 1024 * 1024
a = np.ascontiguousarray(np.random.uniform(-1, 1, n), dtype=np.float32)
b = np.ascontiguousarray(np.random.uniform(-1, 1, n), dtype=np.float32)
out = np.zeros(n, dtype=np.float32)

# 4) 调用并验证
ret = lib.vector_add_cuda(a, b, out, n)
print("返回码:", ret, "最大误差:", np.max(np.abs(out - (a + b))))
```

**常见坑：**
- 数组必须是 **C 连续 + float32**，否则指针传递语义错误 → 用 `np.ascontiguousarray`。
- 必须设置 `argtypes`，否则在 64 位系统上指针被当作 32 位 int 截断 → 崩溃。
- `get_device_info` 的缓冲区用 `ctypes.create_string_buffer(size)` 分配，C 端用 `snprintf` 填充。

## 预期输出

```
============================================================
     Day 37: ctypes 调用 CUDA 动态库 — Python 演示
============================================================
[加载] 找到动态库: ...\build\Release\cuda_ctypes.dll

[演示 1] 查询 GPU 设备信息 (get_device_info)
------------------------------------------------------------
  设备名称=NVIDIA GeForce RTX 5060 Ti | 计算能力=12.0 | 显存=15.99 GB | SM数=36 | Warp大小=32

[演示 2] GPU 向量加法 (vector_add_cuda)
------------------------------------------------------------
  向量长度: 1048576 个元素 (4.00 MB/数组)
  调用返回码: 0 (0=成功)
  ...
  最大误差  : 0.00e+00  ->  通过

[演示 3] GPU 矩阵乘法 (matmul_cuda)
------------------------------------------------------------
  矩阵规模: 512 x 512
  调用返回码: 0 (0=成功)
  ...
  最大误差  : ...  ->  通过

============================================================
                        性能总结
============================================================
  任务             GPU(ms)     CPU(ms)      结果
  ----------------------------------------------
  向量加法          ...         ...         通过
  矩阵乘法          ...         ...         通过
============================================================
```

## 优缺点分析

### 优点

- **零外部依赖**：Python 端只用标准库 `ctypes`，部署极轻量。
- **语言无关**：生成的动态库同样能被 C / C++ / Go / Rust / C# 等调用。
- **封装彻底**：GPU 内存管理全在库内部，调用方无需理解 CUDA。
- **编译一次、随处调用**：适合把成熟算法打包分发给非 CUDA 用户。

### 缺点

- **数据拷贝开销**：每次调用都走 H2D + D2H，小数据或频繁调用时开销明显；GPU 数据无法在多次调用间常驻。
- **手写 ABI 繁琐**：每个函数都要手动声明 `argtypes/restype`，参数一多易出错。
- **缺乏类型安全**：ctypes 是运行时绑定，类型不匹配可能直接段错误而非编译报错。
- **异常/错误传递弱**：只能靠返回码，无法传递丰富的错误信息与堆栈。

> 若需要 GPU 数据常驻、复杂对象交互，建议改用 PyTorch C++ 扩展（见 Day 21）或 CuPy；
> 若只是把一个成熟 CUDA 算法轻量分发，ctypes 动态库是最省心的方案。

## 进阶阅读

- [Python ctypes 官方文档](https://docs.python.org/zh-cn/3/library/ctypes.html)
- [NumPy ctypeslib.ndpointer](https://numpy.org/doc/stable/reference/routines.ctypeslib.html)
- [Creating Shared Libraries with nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
