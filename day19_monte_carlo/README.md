# Day 19: 蒙特卡洛方法（估算 Pi 与期权定价）

## 学习目标

1. **实现** 蒙特卡洛估算圆周率 Pi：在单位正方形内随机投点，用落入单位圆的比例估算 π
2. **使用** `curand` 设备端随机数库，为每个线程独立初始化随机状态（`curand_init`）
3. **实现** 跨度循环（grid-stride loop），让有限线程处理远大于线程数的样本量
4. **实现** 基于几何布朗运动的 Black-Scholes 蒙特卡洛期权定价 Kernel
5. **测量** 16M 样本下 GPU 蒙特卡洛的耗时与估算误差，理解"样本越多误差越小"的收敛规律

---

## 前置依赖

- **Day 03/04（线程与索引）**：一维全局线程索引与网格跨度
- **Day 09/10（原子操作）**：`atomicAdd` 累加各线程局部计数
- **Day 18（前沿 BFS）**：同样使用原子累加聚合并行结果

---

## 核心概念

### 1. 蒙特卡洛估算 Pi 的几何原理

单位正方形 `[0,1]×[0,1]` 面积为 1，其中四分之一单位圆面积为 `π/4`。在正方形内均匀撒点，落入圆内（`x²+y² ≤ 1`）的比例约等于 `π/4`：

```
π ≈ 4 × (圆内点数 / 总点数)

  y
  1 +--------+       随机点落在圆内 (x²+y²≤1) 计数
    | .  . ./|       比例 → π/4
    |. . ./  |       样本越多，估计越准（误差 ∝ 1/sqrt(N)）
    |. ./ .  |
  0 +--------+ x
    0        1
```

对应 [mc_pi_kernel](file:///f:/code/cuda_learning/day19_monte_carlo/src/main.cu#L22-L44)：每个线程用 `curand_uniform` 生成 `(x, y)`，累加局部命中数，最后 `atomicAdd(d_count, local_count)`。

### 2. curand 设备端随机数

每个线程需要**独立、互不相关**的随机序列：

```cpp
curandState state;
curand_init(seed, tid, 0, &state);   // seed 全局种子，tid 作为 subsequence 保证线程间独立
float x = curand_uniform(&state);    // 均匀分布 (0,1]
float z = curand_normal(&state);     // 标准正态分布，用于期权定价
```

- `curand_init(seed, sequence, offset, &state)`：同一 seed 下用不同 `sequence`（这里传 `tid`）保证各线程序列不重叠。
- Pi Kernel 用 `curand_uniform`，期权定价用 `curand_normal`。

### 3. 跨度循环（grid-stride loop）

样本数 `NUM_SAMPLES = 16M` 远超线程总数，每个线程循环处理多个样本：

```cpp
int total_threads = gridDim.x * blockDim.x;
for (int i = tid; i < n; i += total_threads) {
    // 处理第 i 个样本
}
```

这样无论启动多少线程都能覆盖全部样本，且访问步长等于线程总数，天然合并。

### 4. Black-Scholes 蒙特卡洛期权定价

对欧式看涨期权，模拟标的资产在到期日 `T` 的价格 `S_T`（几何布朗运动），取 payoff 均值再贴现：

```
S_T   = S0 * exp((r - 0.5*σ²)*T + σ*sqrt(T)*z),   z ~ N(0,1)
payoff = max(S_T - K, 0)
price  = e^(-rT) * mean(payoff)
```

对应 [mc_option_kernel](file:///f:/code/cuda_learning/day19_monte_carlo/src/main.cu#L49-L66)。参数：`S0=100, K=100, T=1, r=0.05, σ=0.2`。每个 payoff 写入 `d_results`，主机端累加求均值并乘以 `expf(-r*T)` 贴现。

---

## 文件结构

```
day19_monte_carlo/
├── src/main.cu            # CUDA 实现（mc_pi_kernel + mc_option_kernel，curand）
├── python/monte_carlo.py  # Python/Numba 版：原子版 vs 共享内存归约版 估算 Pi
├── CMakeLists.txt         # 构建配置
├── build.bat              # Windows 构建脚本（CMake + VS 2022）
├── build.sh               # Linux/macOS 构建脚本
└── README.md              # 本文件
```

### 两个 Kernel 说明

| Kernel | 作用 | 随机数 | 结果聚合 |
|--------|------|--------|----------|
| `mc_pi_kernel` | 估算 Pi | `curand_uniform` | `atomicAdd` 局部命中数 |
| `mc_option_kernel` | 期权 payoff | `curand_normal` | 写数组，主机端求均值贴现 |

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat（内部调用 CMake，产物在 `build\Release\`）

```powershell
cd day19_monte_carlo
.\build.bat
.\build\Release\monte_carlo.exe
```

方式二：手动 CMake

```powershell
cd day19_monte_carlo
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\monte_carlo.exe
```

### Linux / macOS

```bash
cd day19_monte_carlo
bash build.sh
./build/monte_carlo
```

### Python 版本

```bash
cd day19_monte_carlo/python
python monte_carlo.py
```

> Python 版基于 **Numba** `@cuda.jit`（需 `pip install numba`），用 xorshift32 伪随机数实现两种归约策略：原子操作版与共享内存归约版，并与 CPU/NumPy 结果对比 Pi 误差与耗时。

---

## 知识点总结

1. **蒙特卡洛思想**：用大量随机采样估算确定性问题的解，误差随样本数以 `1/sqrt(N)` 速度下降。
2. **curand 库**：GPU 端高质量随机数生成；`curand_init` 用 `tid` 作 subsequence 保证线程独立。
3. **uniform vs normal**：`curand_uniform` 用于 Pi，`curand_normal` 用于服从正态的资产收益率。
4. **跨度循环**：`i += gridDim.x*blockDim.x`，用固定线程数覆盖任意样本规模。
5. **结果聚合两法**：原子累加（Pi）与写数组后主机归约（期权）；C 版期权用主机 CPU 求和以简化。
6. **金融应用**：Black-Scholes 蒙特卡洛可推广到无解析解的路径依赖期权。

---

## 验收标准

- 运行 `monte_carlo.exe`，Pi 估算值接近 `3.14159`，`误差` 一般小于 `0.001`（16M 样本）。
- 期权定价输出 `期权价格估算` 约在 `10.4` 附近（该组参数的 Black-Scholes 解析价约为 10.45）。
- 两个 Kernel 的计时都能正常打印，程序 `程序正常退出。`。
- Python 版输出 `GPU 原子误差` 与 `GPU 共享误差` 均在 1% 以内，并给出 `共享 vs 原子` 的加速比。

---

## 动手挑战

1. **观察收敛**：把 `NUM_SAMPLES` 从 1M 逐步加到 64M，记录 Pi 误差，验证误差约按 `1/sqrt(N)` 下降。
2. **GPU 端归约期权价**：当前 C 版把整份 `d_results` 拷回主机求和，试着用 Day 里的归约 Kernel 或 Thrust `reduce` 在 GPU 上完成求均值。
3. **对比 PRNG 质量**：把 Python 版的 xorshift32 换成 `numba.cuda.random` 的 xoroshiro128+，比较 Pi 误差差异。

---

## 常见问题

- **Q：为什么每个线程要用不同的 `curand_init` sequence（这里用 tid）？**
  A：若所有线程用相同序列会产生高度相关的随机数，破坏采样独立性，导致估计偏差；用 `tid` 作 subsequence 保证互不重叠。

- **Q：期权定价为何在主机端 CPU 累加而不是 GPU？**
  A：这是教学简化，突出随机数生成部分；`n=16M` 的 float 求和拷回主机可接受。生产环境应在 GPU 上归约以省去大数组拷贝。

- **Q：Pi 估算值每次运行略有不同正常吗？**
  A：正常。蒙特卡洛是随机算法，固定种子可复现，但更换种子/样本数会得到不同的估计值，只要落在误差范围内即可。

---

## 延伸阅读

- [cuRAND Library Programming Guide](https://docs.nvidia.com/cuda/curand/index.html) — 官方随机数库文档
- [NVIDIA Developer Blog — Monte Carlo Simulation with cuRAND](https://developer.nvidia.com/blog/tag/monte-carlo/) — 蒙特卡洛与金融模拟实战
