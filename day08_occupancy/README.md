# Day 8: 寄存器、局部内存与占用率优化（Occupancy）

## 学习目标

1. **解释**寄存器文件的分配策略，以及寄存器溢出（spill）到局部内存的性能代价
2. **计算**理论占用率（Occupancy）：每 SM 实际驻留 Warp 数 / 最大可驻留 Warp 数
3. **测量**三个寄存器用量不同的 Kernel 的执行时间，对比性能差异
4. **实现**用 `__launch_bounds__` 限制寄存器分配，观察对占用率的影响
5. **分析**为什么"高占用率有助于隐藏内存延迟"，但并非越高越好

---

## 前置依赖

- **Day 3 线程层级**：理解 Grid / Block / Warp / Thread 的组织
- **Day 7 合并访问**：理解访存延迟是需要被"隐藏"的对象
- **SM（流多处理器）概念**：寄存器文件、Warp 调度器都属于 SM 的有限资源

---

## 核心概念

### 占用率是什么

占用率 = 每个 SM 上**实际驻留的 Warp 数** / **硬件允许的最大 Warp 数**。SM 通过在多个 Warp 间快速切换来隐藏访存/指令延迟——驻留的 Warp 越多，越有机会"填满"等待的空档。而**每个线程使用的寄存器数**是限制驻留 Block 数的关键因素之一：寄存器用得越多，能同时驻留的 Block/Warp 越少。

```
SM 寄存器文件（有限）
┌──────────────────────────────────────────────┐
│  少寄存器 Kernel:  Block Block Block Block ... │  → 高占用率，延迟隐藏好
│  多寄存器 Kernel:  Block Block                 │  → 占用率下降
│  溢出到局部内存:   Block Block  + spill 访存    │  → 额外访存开销
└──────────────────────────────────────────────┘
```

### 本天的三个对照 Kernel

`src/main.cu`（`ARRAY_SIZE = 1M`，`BLOCK_SIZE = 256`）：

| Kernel | 特点 |
|--------|------|
| `simple_kernel` | 只做 `val*val`，寄存器用量少 |
| `heavy_register_kernel` | 连续 16 个中间变量 `a1..a16`，制造大量寄存器压力 |
| `bounded_register_kernel` | 与 heavy 逻辑相同，但加了 `__launch_bounds__(256, 2)` |

`__launch_bounds__` 的两个参数含义：

```c
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
//                每 Block 最多 256 线程,  每 SM 至少驻留 2 个 Block
```

它提示编译器**限制每线程寄存器数**，以便至少能驻留指定数量的 Block，从而提高占用率（代价可能是溢出到局部内存）。

### 占用率的计算逻辑

程序用 `cudaFuncGetAttributes` 查询每个 Kernel 的实际寄存器数，再结合设备属性估算：

```c
int max_warps_per_sm  = prop.maxThreadsPerMultiProcessor / prop.warpSize;
int warps_per_block   = (block_size + prop.warpSize - 1) / prop.warpSize;
int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
int max_blocks_by_regs= prop.regsPerMultiprocessor / (attr.numRegs * block_size);
// 取二者较小值 -> 实际驻留 Block -> 实际 Warp -> 占用率百分比
```

### Python 版的对照实验

`python/occupancy.py`（Numba）通过遍历 `BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]`，对向量加法 / 向量乘法 Kernel 计时，展示 block size 对性能（间接反映占用率）的影响。

---

## 文件结构

```
day08_occupancy/
├── src/main.cu          # simple / heavy_register / bounded_register 三个 Kernel + 占用率分析
├── python/occupancy.py  # Python/Numba 版（遍历不同 block size 测速）
├── CMakeLists.txt       # CMake 构建配置
├── build.bat            # Windows 编译脚本（调用 CMake + VS2022）
├── build.sh             # Linux/macOS 编译脚本
└── README.md            # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat

```powershell
cd day08_occupancy
.\build.bat
.\build\Release\occupancy.exe
```

方式二：手动使用 CMake

```powershell
cd day08_occupancy
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\occupancy.exe
```

### Linux / macOS

```bash
cd day08_occupancy
bash build.sh
./build/occupancy
```

### Python 版本

```bash
cd day08_occupancy/python
python occupancy.py
```

> 需要安装 Numba：`pip install numba`。

---

## 知识点总结

1. **寄存器**：GPU 上最快的存储，但每 SM 总量有限，是限制占用率的核心资源之一。
2. **局部内存与溢出**：寄存器不够时，变量溢出到 local memory（实际位于全局内存），访问慢很多。
3. **占用率公式**：实际驻留 Warp / 最大可驻留 Warp，受寄存器、共享内存、block size 共同约束。
4. **`__launch_bounds__`**：向编译器承诺 block 大小与最少驻留 block 数，促使其限制寄存器分配、提升占用率。
5. **高占用率隐藏延迟**：更多驻留 Warp 让调度器在访存等待时切换到就绪 Warp。
6. **占用率非越高越好**：过度压低寄存器会引起溢出，反而变慢；需权衡寄存器压力与并发度。
7. **`cudaFuncGetAttributes`**：可在运行时查询 Kernel 的 `numRegs` 与 `sharedSizeBytes`，用于占用率估算。

---

## 验收标准

- 程序打印三个 Kernel 的"寄存器/线程、共享内存/Block、理论占用率"。
- 能观察到 `heavy_register_kernel` 的寄存器数明显高于 `simple_kernel`，占用率相应更低。
- 性能对比表中，`simple_kernel` 通常最快；`bounded_register_kernel` 的寄存器数受 `__launch_bounds__` 约束而低于纯 heavy 版本。
- 你能解释：为什么本例中并非"占用率越高一定越快"。

---

## 动手挑战

1. **改变 launch_bounds 参数**：把 `__launch_bounds__(256, 2)` 改为 `(256, 4)` 或 `(128, 8)`，重新编译观察寄存器数与占用率如何变化。
2. **加长/缩短寄存器链**：把 `heavy_register_kernel` 的 `a1..a16` 扩展到 `a32`，观察寄存器数是否被编译器封顶（触发溢出），并对比耗时。
3. **用 nvcc 报告寄存器**：编译时加 `-Xptxas -v`（或在 CMake 中添加该选项）打印每个 Kernel 的寄存器/局部内存使用量，与运行时 `numRegs` 相互印证。

---

## 常见问题

- **Q：为什么 `heavy_register_kernel` 寄存器多却不一定慢很多？**
  A：本例是访存主导的简单运算，编译器可能复用寄存器且缓存命中良好；寄存器压力的性能影响在计算密集或临界占用率场景才明显。

- **Q：`__launch_bounds__` 一定能提升性能吗？**
  A：不一定。它提高占用率的同时可能引入寄存器溢出。是否更快取决于 Kernel 是延迟受限还是资源受限，应实测。

- **Q：占用率 100% 是目标吗？**
  A：不是。占用率只是隐藏延迟的手段之一。很多高性能 Kernel 在 50%~75% 占用率下反而最快，因为它们靠指令级并行（ILP）和缓存也能隐藏延迟。

---

## 延伸阅读

- [CUDA C++ Programming Guide — Occupancy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#occupancy)
- [CUDA C++ Best Practices Guide — Register Pressure & `__launch_bounds__`](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)
