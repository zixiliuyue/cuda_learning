# Day 22: Numba CUDA 编程

> **重要说明**：本天的 `src/main.cu` 用 CUDA C++ 实现 K-Means 分配 Kernel，作为**对照参考**；
> 本天的主角是 [python/numba_advanced.py](file:///f:/code/cuda_learning/day22_numba/python/numba_advanced.py)，
> 用 Numba `@cuda.jit` 演示设备函数、CUDA 事件计时、数学内建函数等高级特性。
> 两者并排阅读，能直观看到 Numba 与原生 CUDA 的一一对应关系。

## 学习目标

1. **对应** Numba `@cuda.jit` 语法与原生 CUDA C++ 关键字（`__global__`、`threadIdx` 等）
2. **实现** Numba 设备函数（`@cuda.jit(device=True)`）并在 Kernel 中调用（sigmoid / relu / softplus）
3. **使用** CUDA 事件（`cuda.event()` + `elapsed_time`）进行精确的 GPU 计时
4. **实现** 使用 CUDA 数学内建函数（`math.sin/cos/sqrt/exp` 等）的逐元素 Kernel
5. **对比** 相同 K-Means 分配逻辑在 CUDA C++ 与 Numba 中的写法差异，评估 Numba 的优缺点

---

## 前置依赖

- **Day 03/04（线程与索引）**：`blockIdx/blockDim/threadIdx` 的映射
- **Day 05（逐元素 Kernel / 激活函数）**：sigmoid/relu 的逐元素计算
- **Day 19（Numba 首次登场）**：Numba `@cuda.jit` 与 `cuda.to_device` 基础

---

## 核心概念

### 1. Numba 与 CUDA C++ 的关键字对应

源码 main.cu 开头就打印了这张对照表（[main.cu](file:///f:/code/cuda_learning/day22_numba/src/main.cu#L57-L62)）：

```
@cuda.jit          ->  __global__
cuda.threadIdx.x   ->  threadIdx.x
cuda.blockIdx.x    ->  blockIdx.x
cuda.blockDim.x    ->  blockDim.x
cuda.syncthreads() ->  __syncthreads()
```

Numba 用 Python 装饰器与即时编译（JIT），把 Python 函数编译成 PTX，免去手写 C++ 与显式编译的步骤。

### 2. 对照参考：CUDA C++ 版 K-Means 分配

[kmeans_assign_kernel](file:///f:/code/cuda_learning/day22_numba/src/main.cu#L25-L49)：每个线程处理一个 2D 点，遍历 `K=5` 个质心，计算平方欧氏距离，取最近质心作为标签：

```cpp
for (int c = 0; c < k; c++) {
    float dist = (px-cx)*(px-cx) + (py-cy)*(py-cy);
    if (dist < min_dist) { min_dist = dist; best_label = c; }
}
d_labels[tid] = best_label;
```

这正是 K-Means 的"分配步骤"（assignment step），是可完全并行的逐点操作。

### 3. Numba 设备函数（device function）

设备函数是"在 GPU 上被 Kernel 调用"的辅助函数，相当于 CUDA 的 `__device__`（[numba_advanced.py](file:///f:/code/cuda_learning/day22_numba/python/numba_advanced.py#L35-L60)）：

```python
@cuda.jit(device=True)
def device_sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)      # 数值稳定的分段写法
```

Kernel [activation_kernel](file:///f:/code/cuda_learning/day22_numba/python/numba_advanced.py#L66-L80) 根据 `mode` 参数选择调用 sigmoid / relu / softplus 设备函数。

### 4. CUDA 事件精确计时

普通 `time.time()` 计的是 CPU 墙钟，含启动/同步噪声；CUDA 事件直接在 GPU 时间线上打点（[event_timing_demo](file:///f:/code/cuda_learning/day22_numba/python/numba_advanced.py#L102-L114)）：

```python
start_event = cuda.event(); end_event = cuda.event()
start_event.record()
activation_kernel[grid, block](d_data, d_result, n, 1)
end_event.record(); end_event.synchronize()
elapsed = start_event.elapsed_time(end_event)   # 毫秒
```

### 5. CUDA 数学内建函数

[math_ops_kernel](file:///f:/code/cuda_learning/day22_numba/python/numba_advanced.py#L86-L96) 演示 Numba 支持的一批 `math` 内建函数，直接编译成 GPU 上的快速数学指令：

```python
result[tid] = math.sqrt(math.fabs(x)) + math.sin(x) * math.cos(x)
```

Kernel 结果与 NumPy 逐元素计算对比，误差应小于 `TOLERANCE = 1e-5`。

---

## 文件结构

```
day22_numba/
├── src/main.cu               # 对照参考：CUDA C++ K-Means 分配 Kernel + 关键字对照表
├── python/numba_advanced.py  # 主角：设备函数 / CUDA 事件 / 数学内建 / 激活函数
├── CMakeLists.txt            # 构建配置
├── build.bat                 # Windows 构建脚本（CMake + VS 2022）
├── build.sh                  # Linux/macOS 构建脚本
└── README.md                 # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）—— CUDA C++ 对照版

方式一：使用 build.bat（内部调用 CMake，产物在 `build\Release\`，可执行文件名 `numba_demo.exe`）

```powershell
cd day22_numba
.\build.bat
.\build\Release\numba_demo.exe
```

方式二：手动 CMake

```powershell
cd day22_numba
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\numba_demo.exe
```

### Linux / macOS

```bash
cd day22_numba
bash build.sh
./build/numba_demo
```

### Python 版本（Numba 主角）

```bash
cd day22_numba/python
python numba_advanced.py
```

> Python 版需要 **Numba**（`pip install numba`）；未安装或无 GPU 时会给出提示。
> 它会验证 ReLU、链式数学运算的 GPU/CPU 结果一致性，并输出 CUDA 事件计时与 GPU vs CPU 数学运算加速比。

---

## 知识点总结

1. **关键字一一对应**：Numba `@cuda.jit` 与 CUDA C++ 语义几乎逐条映射，学会一个即通另一个。
2. **JIT 即时编译**：Numba 无需显式编译链接，首次调用触发编译（因此计时要预热）。
3. **设备函数**：`@cuda.jit(device=True)`，只能在 GPU 端被 Kernel 调用，等价 `__device__`。
4. **CUDA 事件计时**：`cuda.event()` + `elapsed_time()` 是测 GPU 内核耗时的准确方式。
5. **数学内建**：`math.sin/cos/sqrt/exp/log/pow/fabs` 会被编译成 GPU 快速数学指令。
6. **Warp/原子操作**：Numba 也提供 `cuda.shfl_sync`、`cuda.atomic.add/min/max/cas`。
7. **Numba 权衡**：开发快、语法简单，但调试信息有限、峰值性能通常不及手写 CUDA C++（main.cu 结尾列出）。

---

## 验收标准

- 运行 `numba_demo.exe`，看到关键字对照表、`K-Means 分配` 计时和 Numba 优缺点说明，程序正常退出。
- Python 版输出 `ReLU 验证 : 通过` 与 `数学运算验证 : 通过`（误差 < `1e-5`）。
- Python 版打印出 `CUDA 事件计时`、`GPU 数学运算`、`CPU 数学运算` 三行，以及 `GPU vs CPU 数学` 加速比。
- 能对照 main.cu 与 numba_advanced.py，说出 `threadIdx.x` 在两种写法中的对应关系。

---

## 动手挑战

1. **把 K-Means 分配移植到 Numba**：仿照 main.cu 写一个 `@cuda.jit` 的 `kmeans_assign`，用 Numba 复现 C++ 版的分类结果。
2. **补全 K-Means 更新步骤**：分配之后，用 `cuda.atomic.add` 统计每个簇的点坐标和与计数，在 GPU 上重算质心，实现完整一轮迭代。
3. **Warp Shuffle 归约**：用 `cuda.shfl_down_sync` 在 warp 内做求和归约，替代共享内存归约，比较代码量与耗时。

---

## 常见问题

- **Q：为什么 Python 版第一次调用 Kernel 特别慢？**
  A：Numba 是 JIT，首次调用要编译成 PTX/机器码；所以计时前都会先"预热"跑一次，把编译开销排除。

- **Q：Numba Kernel 里能用普通 Python 库（如 numpy 函数）吗？**
  A：不能任意用。Kernel 内只支持受限子集（`math` 内建、部分数组操作、`cuda.*` API）；复杂 Python 对象会编译失败。

- **Q：Numba 和 CuPy、PyCUDA 有什么区别？**
  A：Numba 编译你写的 Python Kernel；CuPy 提供 NumPy 兼容的 GPU 数组与现成算子；PyCUDA 更底层地暴露 CUDA Driver API。

---

## 延伸阅读

- [Numba CUDA 官方文档](https://numba.readthedocs.io/en/stable/cuda/index.html) — `@cuda.jit`、设备函数、内建 API
- [Numba CUDA — Writing Device Functions](https://numba.readthedocs.io/en/stable/cuda/device-functions.html) — 设备函数与 Warp 操作
