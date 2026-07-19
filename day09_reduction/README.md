# Day 9: 原子操作与并行归约（Parallel Reduction）

## 学习目标

1. **实现**三种数组求和归约：原子操作、共享内存树形归约、Warp Shuffle 归约
2. **解释**为什么 `atomicAdd` 到同一地址会因线程竞争而严重串行化
3. **对比**三种归约的执行时间，理解 O(N) 串行到 O(log N) 并行的算法改进
4. **测量**并验证 GPU 归约结果与 CPU 参考值的误差在可接受范围内
5. **分析**两阶段归约（Block 级 + Grid 级）如何减少全局内存竞争

---

## 前置依赖

- **Day 6 共享内存**：树形归约依赖 `__shared__` 数组与 `__syncthreads()`
- **Day 8 占用率**：`BLOCK_SIZE` 选取影响并发度
- **Warp 概念**：Warp Shuffle 归约以 32 线程为单位交换数据

---

## 核心概念

### 归约的本质

把 N 个元素两两相加，形成二叉树，深度 `log2(N)`。串行求和是 O(N)，并行树形归约把每层的加法并行化，关键路径缩短到 O(log N)。

```
树形归约（reduce_shared_kernel 内 stride 折半）:
  s[0] s[1] s[2] s[3] s[4] s[5] s[6] s[7]
   │____│    │____│    │____│    │____│      stride=4? 实际从 blockDim/2 开始折半
   s0        s1        s2        s3
   │_________│         │_________│
   s0                  s1
   │___________________│
   s0  = 全部之和
```

### 本天的三个归约 Kernel

`src/main.cu`（`ARRAY_SIZE = 4M` 个 float，`BLOCK_SIZE = 256`）：

| Kernel | 思路 | 特点 |
|--------|------|------|
| `reduce_atomic_kernel` | 每线程 `atomicAdd(d_out, d_in[tid])` | 最简单，但所有线程争用同一地址，最慢 |
| `reduce_shared_kernel` | 共享内存内 `stride` 折半树形归约 | 每 Block 归约出部分和，需第二轮再归约 |
| `reduce_warp_shuffle_kernel` | `__shfl_down_sync` Warp 内归约 | 无需共享内存同步，效率高 |

共享内存树形归约的核心循环：

```c
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local_tid < stride)
        sdata[local_tid] += sdata[local_tid + stride];
    __syncthreads();
}
```

Warp Shuffle 归约的核心（先 Warp 内，再由第一个 Warp 汇总各 Warp 结果）：

```c
for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
```

### 两阶段归约

共享内存 / shuffle 版本都是**两阶段**：第一轮每个 Block 产出一个部分和写入 `d_block_sums`，第二轮再用同一 Kernel（`<<<1, BLOCK_SIZE>>>`）把这些部分和归约成最终结果，避免所有线程直接竞争一个全局地址。

### Python 版

`python/reduction.py`（Numba）实现 `reduction_atomic_kernel`（原子加）与 `reduction_tree_kernel`（共享内存树形归约 + CPU 汇总部分和），并对比性能。

---

## 文件结构

```
day09_reduction/
├── src/main.cu          # atomic / shared / warp_shuffle 三种归约 + CPU 参考对比
├── python/reduction.py  # Python/Numba 版（原子归约 vs 树形归约）
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
cd day09_reduction
.\build.bat
.\build\Release\reduction.exe
```

方式二：手动使用 CMake

```powershell
cd day09_reduction
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\reduction.exe
```

### Linux / macOS

```bash
cd day09_reduction
bash build.sh
./build/reduction
```

### Python 版本

```bash
cd day09_reduction/python
python reduction.py
```

> 需要安装 Numba：`pip install numba`。

---

## 知识点总结

1. **atomicAdd**：保证线程安全，但对同一地址的高并发原子操作会串行化，性能最低。
2. **共享内存归约**：先把数据搬入片上共享内存，在 Block 内做树形折半归约，大幅减少全局竞争。
3. **`__syncthreads()`**：树形归约每一步后必须同步，确保上一层写入对下一层可见。
4. **`__shfl_down_sync`**：Warp 内寄存器直接交换数据，无需共享内存与同步，是最高效的归约原语。
5. **两阶段归约**：Block 级部分和 + Grid 级再归约，是处理超大数组的标准范式。
6. **复杂度**：并行树形归约把关键路径从 O(N) 降到 O(log N)。
7. **浮点误差**：GPU 累加顺序与 CPU 不同，求和存在浮点误差，需用容差（Python 版 `TOLERANCE=1e-1`）判定正确性。

---

## 验收标准

- 程序打印 CPU 归约结果，以及三种 GPU 归约的耗时与相对 CPU 的误差。
- 三种 GPU 结果与 CPU 参考值误差都很小（属浮点累加正常范围）。
- 能观察到**共享内存 / Warp Shuffle 归约明显快于原子操作归约**。
- 你能解释：为什么原子归约慢，以及两阶段归约如何缓解竞争。

---

## 动手挑战

1. **首次加载即相加**：修改 `reduce_shared_kernel`，让每个线程在载入共享内存时就累加两个元素（grid-stride 的一半规模），减少一半 Block 数并观察提速。
2. **消除 bank conflict / 分支发散**：把折半归约中的 `if (local_tid < stride)` 与索引方式优化为顺序寻址版本，对比是否更快。
3. **改数据规模测速**：把 `ARRAY_SIZE` 从 4M 改为 16M / 64M，记录三种方法耗时随规模的变化趋势，找出原子版本随规模恶化的现象。

---

## 常见问题

- **Q：为什么归约要求元素数是 2 的幂？**
  A：树形折半 `stride >>= 1` 假设 Block 内线程数为 2 的幂，能对称折半。非 2 的幂需要额外的边界处理（把越界元素补 0，如源码中 `sdata[local_tid] = 0.0f`）。

- **Q：`__shfl_down_sync` 的 `0xFFFFFFFF` 掩码是什么意思？**
  A：它表示参与本次 shuffle 的活跃线程掩码，`0xFFFFFFFF` 代表 Warp 内全部 32 个线程都参与。

- **Q：为什么 GPU 求和结果和 CPU 略有差异？**
  A：浮点加法不满足结合律，GPU 的并行累加顺序与 CPU 串行不同，导致舍入误差累积不同，属正常现象。

---

## 延伸阅读

- [NVIDIA — Optimizing Parallel Reduction in CUDA (Mark Harris)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [CUDA C++ Programming Guide — Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
