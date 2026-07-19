# Day 11: 常量内存与纹理内存（Constant & Texture Memory）

## 学习目标

1. **解释**常量内存（Constant Memory）的广播机制：Warp 内所有线程读同一地址时一次广播完成
2. **实现**用 `__constant__` 声明查找表（LUT），并用 `cudaMemcpyToSymbol` 从主机拷贝数据
3. **对比**常量内存 LUT 与全局内存 LUT 的查表性能，计算加速比
4. **测量**只读数据缓存优化带来的收益
5. **描述**纹理内存的适用场景（2D 空间局部性、硬件插值与缓存）

---

## 前置依赖

- **Day 5 全局内存**：作为对照的 `lut_global_kernel` 从全局内存读表
- **Day 7 合并访问**：理解不同内存空间的访问模式与缓存差异
- **Warp 概念**：常量内存的广播以 Warp 为单位

---

## 核心概念

### 常量内存的广播机制

常量内存是位于设备端、带专用常量缓存的**只读**空间（大小上限 64KB）。它最擅长的场景是：**一个 Warp 内的所有线程读取同一个地址**——此时硬件只需一次读取，然后广播给全 Warp，效率极高。若同一 Warp 的线程读取不同地址，则会被串行化（多次访问）。

```
常量内存广播（理想场景：全 Warp 读同一地址）:
  c_lut[idx] ──┬──► 线程0
               ├──► 线程1
               ├──► ...        一次读取，广播 32 份
               └──► 线程31

全局内存读同一表（无广播，依赖 L1/L2 缓存命中）:
  d_lut[idx] ──► 每线程各自发起访问（缓存可能命中，但语义上非广播）
```

### 本天实现要点

`src/main.cu`（`LUT_SIZE = 256`，`ARRAY_SIZE = 4M`，`BLOCK_SIZE = 256`）：

- **常量内存声明**（全局作用域）：`__constant__ float c_lut[LUT_SIZE];`
- **拷贝到常量内存**：`cudaMemcpyToSymbol(c_lut, h_lut, LUT_SIZE * sizeof(float));`
- 两个对照 Kernel：

| Kernel | 表来源 | 说明 |
|--------|--------|------|
| `lut_constant_kernel` | `c_lut`（常量内存） | 通过广播/常量缓存查表 |
| `lut_global_kernel` | `d_lut`（全局内存指针） | 从全局内存查表，作对照 |

查表索引计算（两者一致）：

```c
int idx = (int)(d_in[tid] * (LUT_SIZE - 1)) & (LUT_SIZE - 1);
d_out[tid] = c_lut[idx];   // 或 d_lut[idx]
```

程序对两种 Kernel 计时并输出"加速比 = 全局内存耗时 / 常量内存耗时"。

### 关于纹理内存

源码在知识点里说明了纹理内存的定位：**适合 2D 空间局部性访问，带硬件缓存与插值能力**（本天代码主要以常量内存做可运行的对照实验；纹理内存作为概念延伸）。

### Python 版

`python/constant_texture.py`（Numba）用 `cuda.declare_device` 声明常量内存数组（`CONSTANT_SIZE = 256`），对比 `kernel_with_constant` 与 `kernel_with_global` 的读取性能。

---

## 文件结构

```
day11_constant_texture/
├── src/main.cu                 # __constant__ LUT + cudaMemcpyToSymbol，常量 vs 全局内存查表对比
├── python/constant_texture.py  # Python/Numba 版（常量内存 vs 全局内存）
├── CMakeLists.txt              # CMake 构建配置
├── build.bat                   # Windows 编译脚本（调用 CMake + VS2022）
├── build.sh                    # Linux/macOS 编译脚本
└── README.md                   # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat

```powershell
cd day11_constant_texture
.\build.bat
.\build\Release\constant_texture.exe
```

方式二：手动使用 CMake

```powershell
cd day11_constant_texture
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\constant_texture.exe
```

### Linux / macOS

```bash
cd day11_constant_texture
bash build.sh
./build/constant_texture
```

### Python 版本

```bash
cd day11_constant_texture/python
python constant_texture.py
```

> 需要安装 Numba：`pip install numba`。

---

## 知识点总结

1. **`__constant__`**：在全局作用域声明常量内存变量，对所有 Kernel 可见，只读。
2. **`cudaMemcpyToSymbol`**：把主机数据拷贝到设备常量内存（注意是符号名而非普通指针）。
3. **广播机制**：Warp 内线程读同一地址时，常量缓存一次读取广播给全 Warp，效率极高。
4. **发散惩罚**：若同一 Warp 读不同常量地址，会退化为串行多次访问。
5. **容量限制**：常量内存上限 64KB，适合小而频繁复用的只读数据（系数、LUT、卷积核等）。
6. **纹理内存**：适合 2D/3D 空间局部性访问，具备硬件缓存与线性插值能力，常用于图像处理。
7. **只读缓存优化**：合理使用常量/纹理内存，可为只读数据提供专用缓存路径，减轻全局内存压力。

---

## 验收标准

- 程序打印全局内存 LUT 与常量内存 LUT 的耗时以及加速比。
- 常量内存/全局内存两种查表得到一致的输出结果（无 CUDA 错误）。
- 你能解释：本例查表在什么访问分布下常量内存最有优势（Warp 内读同一/邻近地址）。

> 说明：由于现代 GPU 全局内存也有强大的 L1/L2 缓存，且本例 LUT 很小，常量内存的加速比可能不大甚至接近 1；重点在于理解**广播机制的适用场景**，而非追求某个固定加速倍数。

---

## 动手挑战

1. **制造广播场景**：修改索引，让同一个 Block/Warp 内所有线程访问**同一个** LUT 元素（如 `idx = blockIdx.x & (LUT_SIZE-1)`），重新对比常量内存与全局内存，观察广播优势是否放大。
2. **增大 LUT 逼近上限**：把 `LUT_SIZE` 提高（注意常量内存 64KB 上限，float 最多约 16K 个），观察容量增大后的表现。
3. **接入纹理对象**：用现代 `cudaTextureObject_t` API 实现一个纹理版查表 Kernel，与常量/全局版本三方对比，体会纹理缓存对局部性访问的作用。

---

## 常见问题

- **Q：常量内存为什么有 64KB 限制？**
  A：这是硬件常量内存空间的容量约束。它面向"小而热"的只读数据设计，靠专用常量缓存加速，因此容量小但访问高效。

- **Q：为什么我的常量内存版本没比全局内存快多少？**
  A：现代 GPU 的 L1/L2 缓存对小 LUT 命中率很高，全局内存已经很快；常量内存优势主要在"全 Warp 读同一地址的广播"场景，普通分散索引下差距不大。

- **Q：`cudaMemcpyToSymbol` 和 `cudaMemcpy` 有何区别？**
  A：`cudaMemcpyToSymbol` 面向设备端的 `__constant__` / `__device__` 符号，传入的是符号名；`cudaMemcpy` 面向用 `cudaMalloc` 得到的普通设备指针。

---

## 延伸阅读

- [CUDA C++ Programming Guide — Constant Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constant-memory)
- [CUDA C++ Programming Guide — Texture and Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)
