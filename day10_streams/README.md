# Day 10: CUDA 流与异步执行（Streams）

## 学习目标

1. **解释**CUDA 流（Stream）的概念：流是 GPU 操作的 FIFO 队列，同流内串行、跨流可并发
2. **实现**用多个流把数据分段，重叠"主机↔设备传输"与"Kernel 计算"
3. **对比**串行（默认流）与多流并行执行的总耗时，计算加速比
4. **测量**页锁定内存（pinned memory）配合 `cudaMemcpyAsync` 实现真正异步传输
5. **分析**默认流与显式流的同步语义差异

---

## 前置依赖

- **Day 5 内存管理**：`cudaMalloc` / `cudaMemcpy` 的同步语义
- **Day 9 计时**：使用 GpuTimer（基于 CUDA event）精确计时
- **异步概念**：理解 CPU 提交任务后不阻塞、GPU 异步执行的模型

---

## 核心概念

### 流与重叠执行

一个 Kernel 执行时，如果 GPU 的拷贝引擎（Copy Engine）空闲，就可以同时为另一段数据做传输。把工作切分到多个流，让"传输"和"计算"在时间轴上错开重叠，从而缩短总时间。

```
串行（默认流）:
  段0: [H2D][Kernel][D2H]  段1: [H2D][Kernel][D2H]  段2: ...   (首尾相接)

多流并行:
  流0: [H2D][Kernel][D2H]
  流1:      [H2D][Kernel][D2H]
  流2:           [H2D][Kernel][D2H]     ← 传输与计算重叠
  流3:                [H2D][Kernel][D2H]
              时间轴 ───────────────►      总时间显著缩短
```

### 本天实现要点

`src/main.cu`（`SEGMENT_SIZE = 1M`，`NUM_SEGMENTS = 4`，`BLOCK_SIZE = 256`）：

- 唯一的计算 Kernel：`vector_scale_kernel`，做 `d_out[tid] = d_in[tid] * scale`（scale=2.0）
- **页锁定内存**：用 `cudaMallocHost` 分配 `h_in` / `h_out`，这是异步拷贝的前提
- **串行版**：循环内用同步 `cudaMemcpy` + Kernel + `cudaMemcpy`
- **多流版**：为每段创建独立 `cudaStreamCreate`，用 `cudaMemcpyAsync(..., streams[i])` 和 `kernel<<<grid, block, 0, streams[i]>>>` 提交，最后 `cudaStreamSynchronize` 逐流同步

关键差异：

```c
// 异步传输必须使用 pinned 内存 + Async 版本 + 指定流
cudaMemcpyAsync(d_in_seg, h_in_seg, segment_bytes,
                cudaMemcpyHostToDevice, streams[i]);
vector_scale_kernel<<<grid_size, BLOCK_SIZE, 0, streams[i]>>>(
    d_in_seg, d_out_seg, 2.0f, SEGMENT_SIZE);
cudaMemcpyAsync(h_out_seg, d_out_seg, segment_bytes,
                cudaMemcpyDeviceToHost, streams[i]);
```

### Python 版

`python/streams.py`（Numba）用 `NUM_STREAMS = 4` 演示 `vec_add_kernel` 与 `vec_scale_kernel` 在多流下的重叠执行，CPU 参考为 `(a + b + 1.0) * 2.0`。

---

## 文件结构

```
day10_streams/
├── src/main.cu        # vector_scale_kernel + 串行 vs 多流并行对比（pinned 内存 + Async）
├── python/streams.py  # Python/Numba 版（cuda.stream() 多流演示）
├── CMakeLists.txt     # CMake 构建配置
├── build.bat          # Windows 编译脚本（调用 CMake + VS2022）
├── build.sh           # Linux/macOS 编译脚本
└── README.md          # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat

```powershell
cd day10_streams
.\build.bat
.\build\Release\streams.exe
```

方式二：手动使用 CMake

```powershell
cd day10_streams
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\streams.exe
```

### Linux / macOS

```bash
cd day10_streams
bash build.sh
./build/streams
```

### Python 版本

```bash
cd day10_streams/python
python streams.py
```

> 需要安装 Numba：`pip install numba`。

---

## 知识点总结

1. **流（Stream）**：GPU 操作的 FIFO 队列；同一流内顺序执行，不同流之间可并发。
2. **默认流（Stream 0）**：具有隐式同步语义，通常会与其他流串行化，因此对比实验用显式流。
3. **页锁定内存**：`cudaMallocHost` 分配的 pinned 内存不可被换页，是 `cudaMemcpyAsync` 真正异步的前提。
4. **异步 API**：`cudaMemcpyAsync` 与带流参数的 Kernel 启动会立即返回，靠流排队执行。
5. **重叠**：多流让 H2D / 计算 / D2H 在时间上错开，隐藏传输开销，缩短总时间。
6. **同步**：用 `cudaStreamSynchronize`（或 event）确保所有流的工作完成后再读取结果。
7. **资源清理**：流用完需 `cudaStreamDestroy`，pinned 内存需 `cudaFreeHost`。

---

## 验收标准

- 程序打印串行执行与多流并行执行的耗时以及加速比。
- **多流并行时间通常短于串行时间**（加速比 > 1），加速幅度取决于 GPU 拷贝引擎数量与传输/计算比例。
- 程序正常创建/销毁 4 个流，无 CUDA 错误。
- 你能解释：为什么必须用 pinned 内存 + `cudaMemcpyAsync` 才能实现真正的重叠。

---

## 动手挑战

1. **增加段数测速**：把 `NUM_SEGMENTS` 从 4 改为 8 / 16，观察加速比是否继续提升，找到重叠收益的边际点。
2. **增大计算量**：把 `vector_scale_kernel` 换成多次幂运算的计算密集 Kernel，观察"计算受限"时多流加速比的变化。
3. **用 event 做细粒度计时**：用 `cudaEventRecord` 在每个流的 H2D / Kernel / D2H 之间打点，画出各段的时间线，验证重叠是否真的发生。

---

## 常见问题

- **Q：为什么用普通 `malloc` 的内存做 `cudaMemcpyAsync` 不会真正异步？**
  A：可分页内存需要驱动先暂存到内部 pinned 缓冲区，实际退化为同步拷贝。只有 `cudaMallocHost` 的页锁定内存能被 DMA 引擎直接异步搬运。

- **Q：多流一定更快吗？**
  A：不一定。若 GPU 只有一个拷贝引擎，或计算时间远大于传输时间，重叠收益有限；加速比取决于传输与计算的比例。

- **Q：默认流为什么会破坏并发？**
  A：传统默认流具有全局同步语义，会等待/阻塞其他流。要与显式流并发，需使用非默认流（或 per-thread 默认流编译选项）。

---

## 延伸阅读

- [NVIDIA Developer Blog — How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
- [CUDA C++ Programming Guide — Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
