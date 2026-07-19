# Day 20: 性能分析与调优综合

## 学习目标

1. **建立** 基准测试（benchmark）方法：预热 + 多次迭代取平均 + 计算有效带宽（GB/s）
2. **实现** 同一向量加法的四个优化版本（朴素 / 循环展开 / `__restrict__` / `float4` 向量化），并对比性能
3. **判断** 一个 Kernel 是"内存受限"还是"计算受限"（Roofline 视角）
4. **测量** Block Size 对占用率与吞吐的影响（Python 版扫描 64~1024）
5. **对比** 朴素矩阵乘法与共享内存 Tiled 版本的加速比，总结调优方法论

---

## 前置依赖

- **Day 07/08（共享内存 / Tiling）**：Tiled 矩阵乘法的分块思想（Python 版用到）
- **Day 11/12（内存合并访问）**：合并访问与向量化加载的收益来源
- **Day 31（float4 向量化）**：`float4` 一次搬 128 bit（本天首次引入向量化加载）

---

## 核心概念

### 1. 调优方法论（本天主线）

性能优化不是随手改代码，而是**测量 → 定位瓶颈 → 针对性优化 → 再测量验证**的闭环。源码结尾打印了这套方法（[main.cu](file:///f:/code/cuda_learning/day20_performance_tuning/src/main.cu#L147-L156)）：

```
1. 基准测试     : 先测量原始性能，作为对比基线
2. Roofline 分析: 判断计算受限(compute-bound)还是内存受限(memory-bound)
3. 内存优化     : 合并访问、使用共享内存、向量化加载
4. 计算优化     : 循环展开、减少分支、快速数学函数
5. 占用率优化   : 调整 Block 大小、控制寄存器/共享内存用量
6. 迭代验证     : 每次优化后都要验证结果正确性
```

### 2. 正确的基准测试写法

见 [benchmark_kernel](file:///f:/code/cuda_learning/day20_performance_tuning/src/main.cu#L79-L105)，三个关键点：

```cpp
// 1) 预热一次：触发 JIT/首次启动开销，不计入计时
kernel<<<grid, block>>>(...); cudaDeviceSynchronize();

// 2) 多次迭代取平均（这里 50 次），摊薄启动抖动
for (int i = 0; i < 50; i++) kernel<<<grid, block>>>(...);
cudaDeviceSynchronize();

// 3) 计算有效带宽：读 a、读 b、写 c 共 3 份数据
double bandwidth = (3.0 * n * sizeof(float) / GB) / (avg_ms / 1000.0);
```

向量加法是典型的**内存受限**算子（每元素只做一次加法，却要 2 读 1 写），因此用 **GB/s 有效带宽** 而非 FLOPS 衡量最贴切。

### 3. 四个优化版本对比

| 版本 | 函数 | 优化手段 | 网格配置 |
|------|------|----------|----------|
| V1 朴素 | [v1_naive](file:///f:/code/cuda_learning/day20_performance_tuning/src/main.cu#L22-L29) | 每线程处理 1 个元素 | `grid1 = (n+255)/256` |
| V2 循环展开 | [v2_unroll](file:///f:/code/cuda_learning/day20_performance_tuning/src/main.cu#L34-L44) | 每线程处理 4 个元素 | `grid1/4 + 1` |
| V3 restrict | [v3_restrict](file:///f:/code/cuda_learning/day20_performance_tuning/src/main.cu#L49-L58) | `const __restrict__` 别名提示 | `grid1` |
| V4 float4 | [v4_float4](file:///f:/code/cuda_learning/day20_performance_tuning/src/main.cu#L63-L77) | 一次加载/存储 128 bit | 每线程 1 个 float4 |

- **`__restrict__`**：告诉编译器 `a/b/c` 指针不重叠，可放心重排/合并访存、启用只读缓存。
- **`float4`**：把 4 个 float 打包成一次 128 bit 事务，减少访存指令数、提升带宽利用率。

> 注意：`main` 中真正跑 benchmark 的是 V1/V2/V3（[main.cu](file:///f:/code/cuda_learning/day20_performance_tuning/src/main.cu#L139-L142)）；V4 因为参数类型是 `float4*` 与 benchmark 的函数签名不兼容，源码里仅打印说明（"需要类型转换"），把它的实测留作练习。

### 4. Block Size 与占用率（Python 版）

Python/Numba 版扫描 `block_sizes = [64,128,256,512,1024]`（[performance_tuning.py](file:///f:/code/cuda_learning/day20_performance_tuning/python/performance_tuning.py#L141-L154)），演示 Block 大小如何影响 SM 占用率与访存效率，并对比朴素 vs Tiled 矩阵乘法（`512×512`，`TILE_SIZE=16`）的加速比。

---

## 文件结构

```
day20_performance_tuning/
├── src/main.cu                  # 4 个向量加法版本 + benchmark_kernel 计时框架
├── python/performance_tuning.py # Numba: 朴素/Tiled 矩阵乘 + Block Size 扫描
├── CMakeLists.txt               # 构建配置
├── build.bat                    # Windows 构建脚本（CMake + VS 2022）
├── build.sh                     # Linux/macOS 构建脚本
└── README.md                    # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat（内部调用 CMake，产物在 `build\Release\`）

```powershell
cd day20_performance_tuning
.\build.bat
.\build\Release\performance_tuning.exe
```

方式二：手动 CMake

```powershell
cd day20_performance_tuning
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\performance_tuning.exe
```

### Linux / macOS

```bash
cd day20_performance_tuning
bash build.sh
./build/performance_tuning
```

### Python 版本

```bash
cd day20_performance_tuning/python
python performance_tuning.py
```

> Python 版依赖 **Numba**（`pip install numba`），对比矩阵乘法朴素 vs Tiled 加速比、并扫描不同 Block Size 的向量加法耗时。

### 用 Nsight 深入分析（可选）

```powershell
ncu --set full .\build\Release\performance_tuning.exe   # Nsight Compute 逐 Kernel 剖析
nsys profile .\build\Release\performance_tuning.exe      # Nsight Systems 时间线
```

---

## 知识点总结

1. **先测量后优化**：任何优化都必须以基准数据为前提，凭直觉改代码常常适得其反。
2. **预热 + 多次迭代**：排除首次启动开销与抖动，得到稳定的平均耗时。
3. **选对指标**：内存受限算子看 **GB/s**，计算受限算子看 **GFLOPS**；向量加法属前者。
4. **Roofline 模型**：用算术强度（FLOP/Byte）判断瓶颈在计算峰值还是带宽峰值。
5. **循环展开**：每线程多算几个元素，减少启动线程数与循环/边界开销。
6. **`__restrict__` + const**：消除指针别名假设，允许编译器更激进地优化访存。
7. **float4 向量化**：128 bit 事务提升带宽利用率，要求地址对齐、元素数为 4 的倍数。
8. **Block Size 调参**：影响占用率与寄存器分配，需实测选最优（常见 128/256）。

---

## 验收标准

- 运行 `performance_tuning.exe`，看到 V1/V2/V3 各自的 `时间 (ms)` 与 `带宽 (GB/s)` 表格。
- 能读懂带宽数字：例如若显卡理论带宽 X GB/s，实测接近 X 即说明该访存密集 Kernel 已接近上限。
- Python 版输出 `Tiled 加速比 > 1`（共享内存版快于朴素矩阵乘），并给出最佳/最差 Block Size 及其差异倍数。
- 能说出为什么向量加法很难通过"计算优化"提速——它本就受带宽限制。

---

## 动手挑战

1. **补全 V4 实测**：为 `float4` 版单独写一个 benchmark（把 `d_a/d_b/d_c` 重解释为 `float4*`，`n/4` 个元素），对比它与 V3 的带宽。
2. **改 Block Size 重测**：把 C 版的 `256` 换成 `128` 和 `512` 各跑一遍，观察带宽是否变化，与 Python 版结论对照。
3. **Roofline 定位**：用 `ncu` 采集 V1 的 `dram__throughput.avg.pct_of_peak_sustained` 指标，确认它是否已到内存带宽上限。

---

## 常见问题

- **Q：为什么 V2/V3 相比 V1 提速往往不明显？**
  A：向量加法是纯内存受限，V1 在合并访问下已接近带宽上限，展开/restrict 主要减少指令开销，对带宽帮助有限；真正能再提一点的是 `float4` 减少事务数。

- **Q：`benchmark_kernel` 里为什么要先做一次预热再计时？**
  A：首次启动含上下文创建、模块加载等一次性开销；不预热会把这些成本算进第一次迭代，导致平均值虚高。

- **Q：Nsight Compute 与 Nsight Systems 有何区别？**
  A：Nsight Compute（`ncu`）做单 Kernel 微观剖析（占用率、访存、指令）；Nsight Systems（`nsys`）看整体时间线（Kernel/拷贝/CPU 的重叠）。

---

## 延伸阅读

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) — 官方性能优化权威指南
- [NVIDIA Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/) — Roofline 与 Kernel 剖析工具
