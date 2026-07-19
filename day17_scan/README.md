# Day 17: 并行前缀和（Scan / Prefix Sum）

## 学习目标

1. **实现** Blelloch 工作高效（work-efficient）并行 Scan 算法，包含上扫（Up-sweep）与下扫（Down-sweep）两阶段
2. **理解** Exclusive Scan（不含当前元素）与 Inclusive Scan（含当前元素）的区别
3. **掌握**上扫如何原地构建部分和树、下扫如何把和分配回各位置，并解释中间「将最后一个元素置零」的作用
4. **测量并对比** GPU Blelloch Scan 与 CPU 串行 Scan 在 1M 元素上的耗时
5. **验证** GPU 前缀和结果与 CPU 串行实现一致（容差 `1e-4`）

---

## 前置依赖

- **一维线程索引**与多轮 Kernel 启动（每个 stride 启一次）
- **归约（Reduce）思想**：上扫阶段本质上是并行归约的变体
- **计时工具** `common/include/timer.h` 的 `GpuTimer`

---

## 核心概念

### 1. 什么是前缀和

前缀和（Scan）把输入数组累积成部分和序列：

```
Inclusive:  out[i] = in[0] + in[1] + ... + in[i]
Exclusive:  out[i] = in[0] + in[1] + ... + in[i-1]   （out[0] = 0）
```

它是流压缩、直方图、基数排序等众多并行算法的基础构件。

### 2. Blelloch 两阶段算法

Blelloch Scan 是**工作高效**的：总工作量 O(N)，步数 O(log N)。它把数组看成二叉树，分两趟原地处理：

```
Up-sweep（上扫 / 归约）：从叶到根，不断把左孩子加到右孩子
    stride = 1, 2, 4, ... :  d[idx] += d[idx - stride]
    结束后根节点（最后一个元素）= 全体元素总和

置零：把最后一个元素设为 0  →  为 Exclusive Scan 做准备

Down-sweep（下扫）：从根到叶，边交换边累加
    stride = n/2, ..., 2, 1 :
        temp = d[idx - stride]
        d[idx - stride] = d[idx]
        d[idx] += temp
    结束后整个数组变成 Exclusive 前缀和
```

### 3. 上扫 Kernel

每个线程按 `stride` 计算负责的右节点下标 `idx = (tid+1)*stride*2 - 1`，把左侧 stride 处的值加过来：

```c
__global__ void scan_up_sweep_kernel(float* d_data, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < n) d_data[idx] += d_data[idx - stride];
}
```

### 4. 下扫 Kernel

下扫在同样的配对上做「交换 + 累加」，把部分和分配回正确位置：

```c
__global__ void scan_down_sweep_kernel(float* d_data, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < n) {
        float temp = d_data[idx - stride];
        d_data[idx - stride] = d_data[idx];
        d_data[idx] += temp;
    }
}
```

### 5. 主机端驱动与置零

主机端先递增 stride 跑上扫，再把最后一个元素置零（Exclusive Scan 的关键一步），最后递减 stride 跑下扫：

```c
for (int stride = 1; stride < n; stride *= 2)
    scan_up_sweep_kernel<<<blocks, BLOCK_SIZE>>>(d_data, n, stride);

float zero = 0.0f;                                    // 置零
cudaMemcpy(&d_data[n-1], &zero, sizeof(float), cudaMemcpyHostToDevice);

for (int stride = n/2; stride >= 1; stride /= 2)
    scan_down_sweep_kernel<<<blocks, BLOCK_SIZE>>>(d_data, n, stride);
```

CPU 参考实现 `scan_cpu` 是简单的串行累加（Inclusive），用于对照验证。

---

## 文件结构

```
day17_scan/
├── src/main.cu          # Blelloch 上扫/下扫 Kernel + CPU 串行 Scan 对照
├── python/scan.py       # Python 版（Numba CUDA Blelloch Scan，CPU/NumPy 对照）
├── CMakeLists.txt       # 版本感知架构选择 + /utf-8（无需额外库）
├── build.bat            # Windows 编译脚本（生成 scan_demo.exe）
├── build.sh             # Linux/macOS 编译脚本
└── README.md            # 本文件
```

关键参数：`ARRAY_SIZE = 1024 * 1024`（1M）、`BLOCK_SIZE = 512`。输入为 `[0,10)` 的随机整数（存为 float）。

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat

```powershell
cd day17_scan
.\build.bat
.\build\Release\scan_demo.exe
```

方式二：手动使用 CMake

```powershell
cd day17_scan
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\scan_demo.exe
```

### Linux / macOS

```bash
cd day17_scan
bash build.sh
./build/scan_demo
```

> 本天只用 CUDA Runtime，不依赖额外库。

### Python 版本

```bash
cd day17_scan/python
python scan.py
```

> 使用 Numba CUDA 实现 Blelloch Scan（1M 元素，自动 padding 到 2 的幂），并与 CPU 串行 Scan 及 `np.cumsum`（转 Exclusive）对照，容差 `1e-3`。需 `pip install numba`。

---

## 知识点总结

1. **前缀和**：`out[i]` 为前 i 个（或前 i-1 个）元素之和，是众多并行算法的构件。
2. **Exclusive vs Inclusive**：是否包含当前元素；Blelloch 天然产出 Exclusive Scan。
3. **Blelloch 工作高效**：总工作量 O(N)，步数 O(log N)，优于朴素 Scan 的 O(N log N) 工作量。
4. **Up-sweep**：自底向上构建部分和，根节点为总和。
5. **置零**：上扫后把最后一个元素设为 0，是转成 Exclusive Scan 的关键步骤。
6. **Down-sweep**：自顶向下「交换+累加」，把部分和分配回各位置。
7. **多轮 Kernel 启动**：每个 stride 启动一次 Kernel，共 O(log N) 轮。
8. **应用场景**：流压缩、直方图、基数排序、稀疏矩阵等。

---

## 验收标准

- 程序打印设备信息与数组大小（1048576 个元素）后正常运行
- 分别输出「GPU Blelloch Scan」与「CPU 串行 Scan」的耗时（ms）
- 「验证结果」显示**通过**（逐元素误差 < `1e-4`，无「不匹配」打印）
- 能解释上扫、置零、下扫三步各自的作用

---

## 动手挑战

1. **Inclusive 版本**：在 Exclusive 结果基础上加回原始输入，得到 Inclusive Scan，并与 `out[i]=out[i-1]+in[i]` 对照验证。
2. **改变运算符**：把加法换成 `max`（前缀最大值）或乘法（前缀积），观察 Scan 框架的通用性（注意置零应改为对应的单位元）。
3. **共享内存分块 Scan**：当前实现全程操作全局内存。尝试在每个 block 内用共享内存做局部 Scan，再对 block 和做二级 Scan，减少全局内存往返。

---

## 常见问题

- **Q：为什么上扫后要把最后一个元素置零？**
  A：这是 Blelloch 算法生成 **Exclusive** Scan 的标准步骤。上扫后最后一个元素是全体总和，置零后下扫会把它当作"根的初始前缀"向下分配，最终 `out[0]=0`，每个位置得到其左侧所有元素之和。

- **Q：GPU Scan 一定比 CPU 快吗？**
  A：不一定。Scan 是访存密集型且步数为 O(log N)，需多次 Kernel 启动；小规模时启动开销可能掩盖并行收益。大规模、结合共享内存优化后才充分体现优势。

- **Q：为什么 CPU 参考是 Inclusive 而 GPU 是 Exclusive，还能直接比对？**
  A：本天源码的 CPU 是 Inclusive 累加。若严格比对需注意两者错位一位；Python 版则显式构造了 Exclusive 参考再比对。理解两种定义的差异是本天重点。

---

## 延伸阅读

- [GPU Gems 3, Chapter 39: Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [维基百科：Prefix sum](https://en.wikipedia.org/wiki/Prefix_sum)
