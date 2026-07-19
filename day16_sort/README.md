# Day 16: 并行排序 - Bitonic Sort（双调排序）

## 学习目标

1. **实现** Bitonic Sort（双调排序）——一种天然适合 GPU 并行的排序网络
2. **理解** Bitonic Sort 的两层循环结构（`k` 控制阶段、`j` 控制比较距离）与 `tid ^ j` 索引配对
3. **掌握**比较-交换（compare-exchange）操作，理解其为何**无分支发散**、方向如何由 `tid & k` 决定
4. **测量并对比** GPU Bitonic Sort 与 CPU `qsort` 在 1M 元素上的耗时
5. **验证**排序结果与 CPU `qsort` 完全一致

---

## 前置依赖

- **一维线程索引**：`tid = blockIdx.x * blockDim.x + threadIdx.x`
- **位运算基础**：理解异或 `^` 与按位与 `&` 在索引配对与方向判断中的作用
- **多轮 Kernel 启动**：本天在主机端用双重循环反复启动同一 Kernel

---

## 核心概念

### 1. 为什么 GPU 用 Bitonic Sort

快速排序等比较排序依赖数据相关的分支与递归，难以高效并行。Bitonic Sort 是一种**排序网络**：比较-交换的位置在编译期就完全确定，与数据无关，因此每一步中大量比较可并行执行，非常契合 GPU。代价是总比较次数偏多，复杂度 O(N log²N)。

### 2. 双重循环结构

主机端用两层循环驱动，`k` 是外层阶段（子序列长度翻倍），`j` 是内层比较距离（不断减半）：

```c
for (int k = 2; k <= n; k *= 2)          // 阶段：双调序列长度
    for (int j = k / 2; j > 0; j /= 2)   // 比较距离
        bitonic_sort_kernel<<<blocks, BLOCK_SIZE>>>(d_data, j, k, n);
```

### 3. Kernel 内的配对与方向

每个线程用 `tid ^ j` 找到自己的比较伙伴 `ixj`，只让 `ixj > tid` 的线程执行（避免重复交换）；交换方向由 `tid & k` 决定，从而构建"先升后降"的双调序列：

```c
int ixj = tid ^ j;
if (ixj > tid && tid < n) {
    if ((tid & k) == 0) {                 // 该段应升序
        if (d_data[tid] > d_data[ixj]) swap(tid, ixj);
    } else {                              // 该段应降序
        if (d_data[tid] < d_data[ixj]) swap(tid, ixj);
    }
}
```

关键点：所有线程执行相同的比较-交换结构，**没有数据相关的复杂分支**，warp 内不易发散。

### 4. 交换本身

比较-交换用临时变量完成经典三步交换：

```c
float temp = d_data[tid];
d_data[tid] = d_data[ixj];
d_data[ixj] = temp;
```

### 5. 与 CPU 对照

主机端用标准库 `qsort`（配合自定义 `compare_float`）排序同一份数据作为参考，逐元素比对（容差 `1e-4`）验证 GPU 结果正确性。

---

## 文件结构

```
day16_sort/
├── src/main.cu          # Bitonic Sort Kernel + CPU qsort 对照验证
├── python/sort.py       # Python 版（Numba CUDA Bitonic Sort，NumPy/CPU 对照）
├── CMakeLists.txt       # 版本感知架构选择 + /utf-8（无需额外库）
├── build.bat            # Windows 编译脚本（生成 sort_demo.exe）
├── build.sh             # Linux/macOS 编译脚本
└── README.md            # 本文件
```

关键参数：`ARRAY_SIZE = 1024 * 1024`（1M，2 的幂）、`BLOCK_SIZE = 512`。数据为 `[0, 1000)` 的随机浮点。

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat

```powershell
cd day16_sort
.\build.bat
.\build\Release\sort_demo.exe
```

方式二：手动使用 CMake

```powershell
cd day16_sort
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\sort_demo.exe
```

### Linux / macOS

```bash
cd day16_sort
bash build.sh
./build/sort_demo
```

> 本天只用 CUDA Runtime，不依赖额外库。

### Python 版本

```bash
cd day16_sort/python
python sort.py
```

> 使用 Numba CUDA 实现 Bitonic Merge Kernel（数组大小 65536，2 的幂），与 `np.sort` 及 CPU 版对照。需 `pip install numba`。

---

## 知识点总结

1. **排序网络**：比较-交换位置与数据无关，编译期确定，天然并行。
2. **复杂度**：Bitonic Sort 为 O(N log²N)，比较次数多但并行度极高。
3. **双重循环**：外层 `k`（阶段）× 内层 `j`（比较距离），每个 `(k,j)` 启动一次 Kernel。
4. **索引配对**：`ixj = tid ^ j`，仅 `ixj > tid` 的线程执行，避免重复交换。
5. **方向控制**：`tid & k` 决定该段升序或降序，构建双调序列。
6. **无分支发散**：所有线程执行相同结构的比较-交换，适合 GPU。
7. **输入限制**：Bitonic Sort 要求元素个数为 2 的幂（本例 1M = 2²⁰）。
8. **实际应用**：Thrust 库的 `thrust::sort`（基于 Radix Sort）通常更高效。

---

## 验收标准

- 程序打印设备信息与数组大小（1048576 个元素）后正常运行
- 分别输出「GPU Bitonic Sort」与「CPU qsort」的耗时（ms）
- 「排序结果验证」显示**通过**（GPU 结果与 CPU qsort 逐元素一致，容差 `1e-4`）
- 能解释 `tid ^ j` 配对与 `tid & k` 方向判断的作用

---

## 动手挑战

1. **规模扫描**：将 `ARRAY_SIZE` 改为 2¹⁶、2¹⁸、2²²，记录 GPU 与 CPU 耗时，观察 GPU 加速比随规模的变化。
2. **降序排序**：修改 Kernel 中的比较方向（交换两个分支的比较符），实现降序排序并验证。
3. **对比 Thrust**：用 `thrust::sort` 排序同一数组，比较其与手写 Bitonic Sort 的性能差距，体会成熟库的优化。

---

## 常见问题

- **Q：为什么数组大小必须是 2 的幂？**
  A：Bitonic Sort 的网络结构基于二分递归构建双调序列，`k` 从 2 翻倍到 N、`j` 减半的循环要求 N 为 2 的幂。非 2 的幂需先 padding 到最近的 2 的幂。

- **Q：为什么用 `ixj > tid` 判断？**
  A：`tid` 与 `ixj = tid ^ j` 是一对比较伙伴，两者会互相计算出对方。只让较小的一方执行交换，避免同一对被交换两次（相当于没交换）。

- **Q：Bitonic Sort 比 qsort 慢正常吗？**
  A：Bitonic Sort 总比较次数是 O(N log²N)，比串行 qsort 的 O(N log N) 更多；它的价值在于高并行度。在足够大规模且 GPU 空闲带宽充足时才体现优势，教学版规模下不一定更快。

---

## 延伸阅读

- [维基百科：Bitonic sorter（双调排序网络）](https://en.wikipedia.org/wiki/Bitonic_sorter)
- [Thrust 官方文档（thrust::sort）](https://nvidia.github.io/cccl/thrust/)
