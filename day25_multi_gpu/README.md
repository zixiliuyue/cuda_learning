# Day 25: 多 GPU 编程基础

## 学习目标

1. 掌握 `cudaGetDeviceCount` / `cudaSetDevice` **实现**在多个 GPU 之间切换上下文
2. **实现**数据并行：把一个大数组按 GPU 数量切分，每块在各自 GPU 上独立计算后合并
3. **测量**多 GPU 数据并行版本向量加法的总耗时，理解「计算 + 数据传输」的整体开销
4. **解释**单 GPU 环境下代码为何仍能正确运行（自动退化为 `num_gpus=1`）
5. **对比** Peer-to-Peer、NCCL、NVLink 等多 GPU 通信/互连方案的适用场景

---

## 前置依赖

- **Day 02 向量加法**：本例的计算内核就是最基础的逐元素加法
- **Day 03 内存模型**：`cudaMalloc` / `cudaMemcpy` 主机-设备数据搬运
- **Day 10 CUDA 流**：多 GPU 常与流配合实现计算/传输重叠（进阶方向）

---

## 核心概念

### 1. 设备枚举与选择

程序先查询系统 GPU 数量，并据此决定切分份数（见 [main.cu](file:///f:/code/cuda_learning/day25_multi_gpu/src/main.cu#L29-L41)）：

```cpp
int device_count = 0;
cudaGetDeviceCount(&device_count);        // 系统有几块 GPU
int num_gpus = (device_count < 2) ? 1 : 2; // 至多用 2 块做演示
int n_per_gpu = ARRAY_SIZE / num_gpus;     // 每块 GPU 分到的元素数
```

`ARRAY_SIZE = 1024*1024*4`（约 400 万元素），`BLOCK_SIZE = 256`。

### 2. 数据并行主循环

核心是一个「按 GPU 循环」的结构：每次 `cudaSetDevice(gpu)` 切换上下文，在该 GPU 上独立分配显存、拷入本块数据、启动 Kernel、拷回结果（见 [main.cu](file:///f:/code/cuda_learning/day25_multi_gpu/src/main.cu#L60-L79)）：

```
      h_a / h_b (主机大数组)
      ┌──────────────┬──────────────┐
      │   前一半      │   后一半      │
      ▼              ▼
   cudaSetDevice(0)   cudaSetDevice(1)
   d_a/d_b/d_c(GPU0)  d_a/d_b/d_c(GPU1)
   vector_add_kernel  vector_add_kernel
      │              │
      └──────┬───────┘
             ▼
        h_c (合并结果，按 offset 写回)
```

```cpp
for (int gpu = 0; gpu < num_gpus; gpu++) {
    cudaSetDevice(gpu);                       // 切换当前 GPU
    cudaMalloc(&d_a, bytes_per_gpu); ...       // 在该 GPU 上分配
    int offset = gpu * n_per_gpu;              // 该 GPU 负责的数据区间
    cudaMemcpy(d_a, h_a + offset, ...);        // 只拷贝本块
    vector_add_kernel<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, n_per_gpu);
    cudaMemcpy(h_c + offset, d_c, ...);        // 结果写回对应区间
}
```

### 3. 计算 Kernel

[vector_add_kernel](file:///f:/code/cuda_learning/day25_multi_gpu/src/main.cu#L17-L21) 就是标准逐元素加法，`c[tid] = a[tid] + b[tid]`。多 GPU 的关键不在 Kernel 本身，而在于**数据划分与设备上下文管理**。

---

## 文件结构

```
day25_multi_gpu/
├── src/main.cu           # 多 GPU 数据并行向量加法
├── python/multi_gpu.py   # Python 版（检测 Numba/PyTorch 设备 + 模拟多 GPU 数据并行）
├── CMakeLists.txt        # 构建配置
├── build.bat             # Windows 编译脚本（生成 multi_gpu.exe）
├── build.sh              # Linux/macOS 编译脚本
└── README.md             # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：build.bat

```powershell
cd day25_multi_gpu
.\build.bat
.\build\Release\multi_gpu.exe
```

方式二：手动 CMake

```powershell
cd day25_multi_gpu
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\multi_gpu.exe
```

### Linux / macOS

```bash
cd day25_multi_gpu
bash build.sh
./build/multi_gpu
```

### Python 版本

```bash
cd day25_multi_gpu/python
python multi_gpu.py
```

> Python 版会用 Numba 与 PyTorch 分别枚举可用 GPU，并用 NumPy 模拟「切分到多 GPU → 分别计算 → 合并」的数据并行流程，最后与单线程结果做误差校验（阈值 `1e-5`）。

---

## 知识点总结

1. **`cudaSetDevice(i)`**：切换当前线程的活动 GPU，之后的 `cudaMalloc` / Kernel 启动都作用在该设备上。
2. **数据并行**：把数据切块分给不同 GPU 各自计算，是最常见、最易实现的多 GPU 扩展方式。
3. **每 GPU 独立显存**：每块 GPU 有独立地址空间，必须各自 `cudaMalloc` 并按 `offset` 拷贝对应数据段。
4. **优雅退化**：只有 1 块 GPU 时 `num_gpus=1`，逻辑照常正确，只是没有并行加速。
5. **Peer-to-Peer**：支持的 GPU 间可通过 `cudaDeviceEnablePeerAccess` 直接互访显存，省去经主机中转。
6. **NCCL**：NVIDIA 的多 GPU 集合通信库，提供 AllReduce/Broadcast 等，用于分布式训练（见 Day 39）。
7. **NVLink**：GPU 间高速互连，带宽远超 PCIe，缓解多 GPU 通信瓶颈。

---

## 验收标准

- 程序打印 `检测到 N 个 GPU 设备`；单 GPU 时打印退化提示但仍正常运行。
- 打印 `数据分割: <n_per_gpu> 元素 / GPU`。
- 打印 `多 GPU 向量加法` 的耗时（毫秒级）。
- 末尾打印五条多 GPU 编程要点，程序正常退出。
- Python 版打印检测到的设备列表，并显示 `结果验证: 通过`。

---

## 动手挑战

1. **加正确性校验**：在 CUDA 版中补一段 CPU 参考计算 `h_ref[i]=h_a[i]+h_b[i]`，逐元素对比 `h_c`，打印最大误差（应为 0）。
2. **重叠计算与传输**：为每个 GPU 创建独立 stream，用 `cudaMemcpyAsync` + 固定内存（pinned memory）让拷贝与计算重叠，测量耗时变化。
3. **非均匀切分**：当 `ARRAY_SIZE` 不能被 `num_gpus` 整除时，让最后一块 GPU 处理余数元素，保证不漏算、不越界。

---

## 常见问题

- **Q：我只有一块 GPU，这个例子还有意义吗？** A：有。它演示的是「数据划分 + 设备上下文切换」的编程范式，逻辑在多 GPU 机器上原样成立，单卡时只是没有并行收益。
- **Q：为什么每个 GPU 都要单独 `cudaMalloc`？** A：每块 GPU 有独立显存地址空间，一块 GPU 的指针不能直接给另一块用（除非启用 P2P 且在同一地址域）。
- **Q：多 GPU 一定更快吗？** A：不一定。数据传输（H2D/D2H）与负载不均可能抵消计算收益，需权衡通信开销。

---

## 延伸阅读

- [CUDA C++ Programming Guide — Multi-Device System](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system)
- [NVIDIA NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
