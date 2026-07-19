# Day 14: cuFFT - GPU 加速的快速傅里叶变换

## 学习目标

1. **学会调用** cuFFT 库完成 1D 复数 FFT，掌握 `cufftPlan1d` / `cufftExecC2C` / `cufftDestroy` 完整流程
2. **理解** FFT 与逆 FFT（IFFT）的关系，解释为何逆变换后需要**除以 N** 才能恢复原始幅度
3. **实现**一个由两个正弦波（10 Hz + 50 Hz）叠加而成的测试信号，并对其做正/逆变换
4. **测量** cuFFT 在 1M（1024×1024）点上的正向与逆向变换耗时
5. **验证**「正变换→逆变换→除以 N」后能否重建原信号（最大误差 < `1e-4`）

---

## 前置依赖

- **CUDA 内存模型**：`cudaMalloc` / `cudaMemcpy` 主机-设备数据搬运
- **计时工具**：复用 `common/include/timer.h` 的 `GpuTimer`
- **复数概念**：cuFFT 使用 `cufftComplex`（含 `.x` 实部、`.y` 虚部）

---

## 核心概念

### 1. 什么是 FFT

傅里叶变换把信号从**时域**转换到**频域**，揭示信号中含有哪些频率分量。FFT 是离散傅里叶变换（DFT）的快速算法，把 O(N²) 复杂度降到 O(N log N)。cuFFT 是 NVIDIA 高度优化的 GPU 实现。

```
时域信号 x[n]  ──FFT──►  频域 X[k]（复数，表示各频率的幅度与相位）
                ◄─IFFT─
```

### 2. C2C 变换与 Plan

本天使用**复数到复数**（Complex-to-Complex, `CUFFT_C2C`）变换。cuFFT 的用法是「先建计划，再执行」：

```c
cufftHandle plan;
cufftPlan1d(&plan, n, CUFFT_C2C, 1);   // n 点、C2C、批量数=1

// 正向变换（就地：输入输出同一缓冲区）
cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
// 逆向变换
cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE);

cufftDestroy(plan);
```

Plan 会为特定尺寸预先规划好最优的分解与线程配置，可反复复用。

### 3. 为什么逆变换要除以 N

cuFFT 的正/逆变换是**未归一化**的：连续做一次 FFT 再一次 IFFT，结果会被放大 N 倍。因此重建原信号时要手动除以 N：

```c
float reconstructed = h_result[i].x / (float)n;   // 归一化恢复幅度
float error = fabsf(reconstructed - h_signal[i].x);
```

### 4. 测试信号构造

源码用两个正弦波叠加生成实数信号（虚部置 0）：

```c
float t = (float)i / (float)n;
float real = sinf(2*PI*10*t) + 0.5f * sinf(2*PI*50*t);
h_signal[i].x = real;   // 实部
h_signal[i].y = 0.0f;   // 虚部
```

### 5. 错误检查宏

源码定义 `CUFFT_SAFE_CALL`，检查返回的 `cufftResult` 是否为 `CUFFT_SUCCESS`。

---

## 文件结构

```
day14_cufft/
├── src/main.cu          # cuFFT 1D C2C 正向 + 逆向变换 + 重建验证
├── python/cufft.py      # Python 版（CuPy 封装 cuFFT，否则 NumPy FFT fallback）
├── CMakeLists.txt       # 版本感知架构选择 + 链接 cufft + /utf-8
├── build.bat            # Windows 编译脚本（生成 cufft_demo.exe）
├── build.sh             # Linux/macOS 编译脚本
└── README.md            # 本文件
```

关键参数：`SIGNAL_SIZE = 1024 * 1024`（1M 点）、验证阈值 `1e-4`。

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat

```powershell
cd day14_cufft
.\build.bat
.\build\Release\cufft_demo.exe
```

方式二：手动使用 CMake

```powershell
cd day14_cufft
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\cufft_demo.exe
```

### Linux / macOS

```bash
cd day14_cufft
bash build.sh
./build/cufft_demo
```

> CMakeLists 已通过 `target_link_libraries(cufft_demo cufft)` 链接 cuFFT，无需手动指定 `-lcufft`。

### Python 版本

```bash
cd day14_cufft/python
python cufft.py
```

> 若安装了 CuPy，会用 `cp.fft.fft` / `cp.fft.ifft`（底层调用 cuFFT）对 1M 点信号计时，并做频率分析找出主要频率分量；否则 fallback 到 NumPy FFT。

---

## 知识点总结

1. **FFT 作用**：时域↔频域转换，O(N log N) 复杂度，是信号处理核心算法。
2. **`cufftPlan1d`**：创建 1D FFT 计划，参数为点数、变换类型（`CUFFT_C2C`）、批量数。
3. **`cufftExecC2C`**：执行复数到复数变换，方向由 `CUFFT_FORWARD` / `CUFFT_INVERSE` 指定。
4. **未归一化**：FFT→IFFT 会放大 N 倍，需手动除以 N 恢复原幅度。
5. **`cufftComplex`**：结构体，`.x` 为实部、`.y` 为虚部；实数信号将虚部置 0。
6. **就地变换**：输入输出可用同一缓冲区（`d_signal, d_signal`）。
7. **性能特征**：大规模 FFT（≥1M 点）GPU 优势显著，且计划可复用降低开销。

---

## 验收标准

- 程序打印设备信息与信号长度（1048576 点）后正常运行
- 分别输出「cuFFT 正向变换」与「cuFFT 逆变换」耗时（ms）
- 打印「最大重建误差」，且「验证结果」显示**通过**（误差 < `1e-4`）
- 理解并能口头解释：重建时除以 N 的原因

---

## 动手挑战

1. **频谱观察**：在正向 FFT 后把结果拷回主机，打印幅度谱 `sqrt(x²+y²)`，确认在第 10 与第 50 个频率 bin 附近出现峰值（对应输入的两个正弦频率）。
2. **改信号 / 改尺寸**：增加第三个正弦分量或改变频率，并把 `SIGNAL_SIZE` 改成 `256*1024`、`4*1024*1024`，观察耗时随规模的变化。
3. **批量 FFT**：把 `cufftPlan1d` 的 batch 参数从 1 改为多个，一次变换多条信号，体会批处理对吞吐的提升。

---

## 常见问题

- **Q：逆变换后结果为什么和原信号差了 N 倍？**
  A：cuFFT 的正/逆变换均未做归一化，需要在 IFFT 后手动除以 N。这是 FFT 库常见约定（NumPy 则把归一化放在 `ifft` 内部）。

- **Q：`CUFFT_C2C` 和 `CUFFT_R2C` 有何区别？**
  A：C2C 是复数到复数；R2C 是实数到复数（利用共轭对称性只输出一半频谱，更省内存）。本例为演示简单用了 C2C，实数信号把虚部设 0。

- **Q：链接报错找不到 cuFFT 符号？**
  A：需链接 cuFFT 库。本项目 CMake 已自动链接；手动用 nvcc 时加 `-lcufft`。

---

## 延伸阅读

- [cuFFT 官方文档](https://docs.nvidia.com/cuda/cufft/index.html)
- [维基百科：快速傅里叶变换（FFT）](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
