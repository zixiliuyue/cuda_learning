# Day 7: 全局内存合并访问（Coalesced Memory Access）

## 学习目标

1. **解释**什么是全局内存的"合并访问"，以及 Warp（32 线程）如何把连续地址的访问合并为少数几个内存事务
2. **测量**顺序访问、跨步访问（stride=2 / stride=32）三种模式下的有效带宽（GB/s）
3. **对比**行优先与列优先内存布局对访问模式的影响（Python 版用矩阵行/列求和演示）
4. **实现**符合合并访问原则的 Kernel：让相邻线程访问相邻内存地址
5. **分析**跨步越大为何触发越多独立的缓存行读取，导致带宽显著下降

---

## 前置依赖

- **Day 5 全局内存基础**：理解 `cudaMalloc` / `cudaMemcpy`、global memory 的访问延迟
- **Day 6 共享内存**：了解片上存储与全局内存的层级差异（本天专注全局内存的访问模式）
- **Warp 概念**：一个 Warp 包含 32 个线程，是硬件调度与访存合并的基本单位

---

## 核心概念

### 什么是合并访问

GPU 的全局内存（DRAM）以固定大小的**事务/缓存行**（通常 32B、64B、128B）为单位搬运数据。当一个 Warp 内 32 个线程访问**连续且对齐**的地址时，硬件可以把它们合并为 1~2 个内存事务；反之，访问越分散，需要的事务越多，有效带宽越低。

```
合并访问（coalesced_read_kernel: d_out[tid] = d_in[tid]）:
  Warp0 线程 0..31  --->  地址 [0][1][2]...[31]   连续
  ┌───────────────────────────────────────────┐
  │  一个 128B 事务即可覆盖 32 个 float          │   高带宽
  └───────────────────────────────────────────┘

跨步访问 stride=32（strided_read_kernel_32: idx = tid*32）:
  Warp0 线程 0..31  --->  地址 [0][32][64]...    每个间隔 128B
  ┌────┐   ┌────┐   ┌────┐  ...
  │事务│   │事务│   │事务│                        每线程几乎独占一个事务
  └────┘   └────┘   └────┘                        带宽严重浪费
```

### 本天实现的三个基准 Kernel

`src/main.cu` 中定义（数组 16M 个 float ≈ 64MB，`BLOCK_SIZE = 256`）：

| Kernel | 访问模式 | 说明 |
|--------|----------|------|
| `coalesced_read_kernel` | `d_out[tid] = d_in[tid]` | 完美合并，Warp 内访问连续地址 |
| `strided_read_kernel_2` | `idx = tid * 2` | 跨步 2，影响较小 |
| `strided_read_kernel_32` | `idx = tid * 32` | 跨步 32（128 字节），严重非合并 |

> 注：源码中还声明了 `random_access_kernel`（随机索引访问）作为"最差情况"示例，`main` 中主要运行前三个模式做带宽对比。

有效带宽的计算方式（读 + 写各 `n` 个 float）：

```c
double bytes_moved = 2.0 * n * sizeof(float);
double bandwidth_gbps = (bytes_moved / (1024.0*1024.0*1024.0)) / (avg_ms / 1000.0);
```

### Python 版的对照实验

`python/coalesced_access.py`（Numba）用 2048×2048 矩阵演示：

- `sum_rows_kernel`：行优先求和，线程沿行方向读取连续地址 → 合并访问
- `sum_cols_kernel`：列优先求和，线程沿列方向读取间隔 `cols` 的地址 → 非合并

并打印"合并访问加速比"（列优先耗时 / 行优先耗时）。

---

## 文件结构

```
day07_coalesced_access/
├── src/main.cu               # CUDA 实现（顺序 / stride=2 / stride=32 三种访问模式 + 带宽测量）
├── python/coalesced_access.py# Python/Numba 版（行优先 vs 列优先矩阵求和）
├── CMakeLists.txt            # CMake 构建配置
├── build.bat                 # Windows 编译脚本（调用 CMake + VS2022）
├── build.sh                  # Linux/macOS 编译脚本
└── README.md                 # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat（内部调用 CMake + Visual Studio 2022）

```powershell
cd day07_coalesced_access
.\build.bat
.\build\Release\coalesced_access.exe
```

方式二：手动使用 CMake

```powershell
cd day07_coalesced_access
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\coalesced_access.exe
```

### Linux / macOS

```bash
cd day07_coalesced_access
bash build.sh
./build/coalesced_access
```

### Python 版本

```bash
cd day07_coalesced_access/python
python coalesced_access.py
```

> 需要安装 Numba：`pip install numba`。

---

## 知识点总结

1. **合并访问**：Warp 内 32 个线程访问连续、对齐的地址，可合并为极少数内存事务，带宽利用率最高。
2. **非合并访问**：线程访问分散/跨步地址时，事务数量激增，带宽可下降数倍到数十倍。
3. **跨步影响**：stride=2 影响较小，stride=32（128 字节）时几乎每个线程独占一个事务，性能最差。
4. **行优先 vs 列优先**：C/C++ 数组行内连续，按行处理易合并，按列处理是跨步访问。
5. **优化原则**：让**相邻线程访问相邻内存地址**，是编写高带宽 Kernel 的第一准则。
6. **带宽度量**：用 `搬运字节数 / 时间` 估算有效带宽，并与设备理论带宽对比，判断是否为访存瓶颈。

---

## 验收标准

- 程序正常打印设备信息与三种访问模式的耗时/带宽表格。
- **合并访问（顺序）的带宽明显高于跨步访问**：通常 stride=32 的有效带宽显著低于顺序访问（数倍差距），能观察到"跨步越大、带宽越低"的趋势。
- Python 版能看到"合并访问加速比 > 1"，且行/列求和验证均显示"通过"。
- 你能用自己的话解释：为什么 stride=32 会退化到近似"每线程一个内存事务"。

---

## 动手挑战

1. **新增 stride=16 / stride=64 变体**：仿照 `strided_read_kernel_32` 增加不同跨步的 Kernel，绘制"跨步大小 vs 有效带宽"的趋势，观察拐点。
2. **修改 BLOCK_SIZE 测速**：把 `BLOCK_SIZE` 从 256 改为 128 / 512，观察合并访问带宽是否变化，思考为什么访问模式（而非 block size）是这里的主要决定因素。
3. **实现列优先转置读取**：写一个二维矩阵 Kernel，分别按行、按列读取并计时，复现 Python 版中行优先与列优先的差距，再尝试用共享内存做转置以恢复合并访问。

---

## 常见问题

- **Q：跨步 Kernel 的元素数量为什么要除以 stride（如 `ARRAY_SIZE / 32`）？**
  A：因为线程访问 `tid * stride`，若仍用完整长度会越界。除以 stride 保证访问范围落在数组内，同时公平对比"搬运相同规模数据"时的带宽。

- **Q：为什么现代 GPU 上跨步访问不像老架构那样慢到几十倍？**
  A：L1/L2 缓存会吸收一部分邻近访问，缓解了非合并的惩罚；但访问越分散、数据集越大（超出缓存），惩罚越明显。

- **Q：合并访问只和地址连续有关吗？**
  A：还与**对齐**有关。起始地址最好按事务大小（如 128 字节）对齐；未对齐会让一个 Warp 的访问跨越两个缓存行，多出一个事务。

---

## 延伸阅读

- [CUDA C++ Best Practices Guide — Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
- [CUDA C++ Programming Guide — Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
