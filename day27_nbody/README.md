# Day 27: N-Body 物理模拟（万有引力全对力计算）

> 🏗️ **综合大项目 · 建议 2-3 天完成**
> 本项目综合运用前面学到的多项 CUDA 优化技术（`float4` 向量化、`rsqrtf` 快速倒数平方根、
> 寄存器累加、双缓冲），完整实现一个 GPU 上的 N 体引力模拟器。建议按下方「动手挑战」的
> 里程碑分阶段推进：**MVP（跑通）→ 优化（共享内存 Tiling）→ 可视化（轨迹动画）**。

## 学习目标

1. **实现** GPU 上 O(N²) 的 N-Body 全对引力计算 Kernel，并用欧拉积分更新速度与位置
2. **解释**万有引力力分量公式 `F_x = G·m₁·m₂·dx / r³` 中为何是 `r³` 而非 `r²`
3. 掌握用 `float4` 把「位置 (x,y,z) + 质量 (w)」打包，一次 128-bit 对齐加载
4. 使用 `rsqrtf()` 快速倒数平方根**优化**除法，并理解 `SOFTENING` 软化因子防止除零奇点
5. **测量** 4096 天体模拟的每步耗时，并**对比** GPU 与 CPU 参考实现（Python 版）的正确性与加速比

---

## 前置依赖

- **Day 02 向量加法**：一维线程映射（每线程处理一个天体）
- **Day 06 共享内存**：优化阶段用于 Tiling 缓存天体块
- **Day 07 合并访问**：分析 O(N²) 读取的访存模式
- **Day 20 性能调优**：向量化、循环展开、快速数学函数等综合优化

---

## 核心概念

### 1. 物理模型

牛顿万有引力：`F = G·m₁·m₂ / r²`，加速度 `a = F/m`。力沿方向向量投影到坐标轴时需再除以 `r`，因此**力的分量含 1/r³**：

```
|F| = G·m₁·m₂ / r²           （引力大小）
方向分量 = dx / r             （单位方向）
F_x = |F| · (dx/r) = G·m₁·m₂·dx / r³   ← 关键：r³
```

### 2. 数据打包：float4

每个天体用一个 `float4` 存储 `位置(x,y,z) + 质量(w)`，一次访存拿到全部信息（见 [main.cu 初始化](file:///f:/code/cuda_learning/day27_nbody/src/main.cu#L199-L209)）：

```cpp
h_pos[i].x/y/z = 随机 [-1,1];
h_pos[i].w     = 1.0f / n;    // 质量：总质量归一化为 1
```

### 3. 核心 Kernel 算法

[nbody_kernel](file:///f:/code/cuda_learning/day27_nbody/src/main.cu#L116-L181) 中每个线程处理一个天体，遍历所有其他天体累加引力：

```
每个线程（一个天体 i）:
    my_pos = d_pos[i]          # 一次 float4 加载位置+质量
    force  = {0,0,0}
    for j in 0..n-1:
        other = d_pos[j]
        d     = other.xyz - my_pos.xyz
        dist² = d·d + SOFTENING        # 软化因子防除零
        inv   = rsqrtf(dist²)          # 快速倒数平方根
        inv³  = inv*inv*inv            # 1/r³
        f     = other.w * inv³         # 省略 G，用质量*1/r³
        force += d * f
    my_vel += force * DT               # 欧拉积分：更新速度
    my_pos += my_vel * DT              # 更新位置
    d_out_pos[i] = my_pos; d_vel[i] = my_vel
```

关键参数（见 [main.cu 宏定义](file:///f:/code/cuda_learning/day27_nbody/src/main.cu#L45-L81)）：
`NUM_BODIES=4096`、`BLOCK_SIZE=256`、`SOFTENING=1e-9f`、`DT=0.01f`、`NUM_STEPS=10`。

### 4. 双缓冲（ping-pong）

主循环用两个缓冲区 `d_pos` / `d_out_pos`，每步计算完**交换指针**而非重新分配显存（见 [main.cu 主循环](file:///f:/code/cuda_learning/day27_nbody/src/main.cu#L226-L237)）：

```cpp
for (step ...) {
    nbody_kernel<<<grid, BLOCK_SIZE>>>(d_pos, d_vel, d_out_pos, n);
    float4* temp = d_pos; d_pos = d_out_pos; d_out_pos = temp;  // 交换
}
cudaDeviceSynchronize();   // Kernel 异步，计时前必须同步
```

---

## 文件结构

```
day27_nbody/
├── src/main.cu       # float4 + rsqrtf + 双缓冲的 O(N^2) N-Body Kernel
├── python/nbody.py   # Numba CUDA 版（GPU kernel + CPU 参考实现做正确性验证）
├── CMakeLists.txt    # 构建配置
├── build.bat         # Windows 编译脚本（生成 nbody.exe）
├── build.sh          # Linux/macOS 编译脚本
└── README.md         # 本文件
```

> 说明：CUDA 版把质量存于 `float4.w`、用「省略 G、乘 1/r³」的加速度写法；Python/Numba 版把质量单独存数组、用 `inv_mass = 1/m` 显式除质量。两者物理等价，参数（`SOFTENING`、`DT`）取值略有不同（CUDA `SOFTENING=1e-9`，Python `1e-3`）。

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：build.bat

```powershell
cd day27_nbody
.\build.bat
.\build\Release\nbody.exe
```

方式二：手动 CMake

```powershell
cd day27_nbody
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\nbody.exe
```

### Linux / macOS

```bash
cd day27_nbody
bash build.sh
./build/nbody
```

### Python 版本

```bash
cd day27_nbody/python
python nbody.py    # 需要 numba；GPU 跑 4096 粒子，CPU 跑 256 粒子做验证
```

---

## 知识点总结

1. **N-Body 复杂度**：全对力计算是 O(N²)，4096 天体每步约 1670 万对相互作用。
2. **力分量 1/r³**：大小 1/r² 再乘单位方向 dx/r，合起来是 1/r³。
3. **float4 向量化**：位置+质量打包成 128-bit，一次对齐加载，提升带宽利用率。
4. **rsqrtf() 快速倒数平方根**：硬件加速，比 `1.0/sqrtf()` 快，相对误差约 2⁻²³，物理模拟足够。
5. **SOFTENING 软化因子**：`dist² += SOFTENING` 避免近距离时 r→0 导致力爆炸（数值奇点）。
6. **欧拉积分**：`v += a·dt; x += v·dt`，简单但 DT 过大会导致能量不守恒、系统发散。
7. **双缓冲**：交换输入/输出指针避免每步重分配显存。
8. **访存瓶颈**：所有线程广播读取 `d_pos[0..n-1]`，是共享内存 Tiling 优化的切入点。
9. **进阶算法**：Barnes-Hut 将远处天体用四极矩近似，把 O(N²) 降为 O(N log N)。

---

## 验收标准

- 程序打印粒子数 4096 与复杂度 `O(N^2)`。
- 打印 `N-Body 模拟 (10步)` 的耗时（参考量级：RTX 5060 Ti 约 50-100 ms，视硬件而定）。
- 程序无 CUDA 错误、正常退出，末尾打印五条优化方向。
- Python 版打印 GPU（4096 粒子）与 CPU（256 粒子）耗时，并显示 `结果验证: 通过 / 差异可接受`（阈值 `1e-3`）。

---

## 动手挑战（分阶段里程碑）

### 里程碑 1 — MVP：跑通并观察能量

- 编译运行 CUDA 版，把 `NUM_STEPS` 提高到 1000，每隔若干步把位置拷回主机，计算系统**总动能**并打印，观察是否随时间稳定（DT 合理时应大致守恒）。

### 里程碑 2 — 优化：共享内存 Tiling

- 改造 `nbody_kernel`：把天体分成 `BLOCK_SIZE` 大小的 tile，每个 block 协作把一个 tile 载入 `__shared__ float4 shPos[]`，`__syncthreads()` 后从共享内存读取计算。
- 对比优化前后每步耗时，解释为什么减少了全局内存访问（每个天体数据只从全局内存读一次而非 N 次）。

### 里程碑 3 — 可视化：轨迹动画

- 把每步位置写出为 CSV / 二进制，用 Python `matplotlib` 的散点动画（或 `matplotlib.animation`）播放天体运动轨迹；也可用 Python/Numba 版直接在内存中收集轨迹绘图。
- 进阶：尝试给天体不同初速度（如切向速度）形成旋转星系状结构。

---

## 常见问题

- **Q：为什么必须 `cudaDeviceSynchronize()` 后再停计时？** A：Kernel 启动是异步的，不同步的话计时器只记录了「提交」时间而非实际计算时间，会得到虚低的耗时。
- **Q：SOFTENING 取多大合适？** A：需匹配系统尺度。太小则近距离数值不稳定；太大则近距离物理精度下降。本例空间在 [-1,1]，CUDA 版用 `1e-9`。
- **Q：模拟几步后粒子「炸开」了？** A：多半是 `DT` 太大导致欧拉积分不稳定（能量不守恒），减小 `DT` 或改用更高阶积分（如 Leapfrog/Verlet）。

---

## 延伸阅读

- [GPU Gems 3, Chapter 31: Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)
- [Barnes-Hut 树算法（O(N log N) N-Body）](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation)
