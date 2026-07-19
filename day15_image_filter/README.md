# Day 15: GPU 图像滤波 - 卷积与边缘检测

## 学习目标

1. **实现** 2D 卷积在 GPU 上的并行化：每个线程独立计算一个输出像素
2. **实现**两种经典滤波算子——5×5 高斯模糊（平滑去噪）与 Sobel 边缘检测（提取边缘）
3. **掌握**使用 `__constant__` 常量内存存储高斯核，并解释其 broadcast 优势
4. **掌握** clamp 边界处理技巧，用 `min(max(...))` 避免分支发散
5. **解释**卷积的访存瓶颈（输入被重复读取 K² 次），并说明共享内存 tile 的优化方向

---

## 前置依赖

- **Day 6 Tiled MatMul**：共享内存 tile 思想是本天卷积优化方向的基础
- **2D Grid/Block 索引**：本天用 `dim3(16,16)` 的 2D block 直接映射图像坐标
- **常量内存概念**：`__constant__` 修饰的只读数据及其缓存机制

---

## 核心概念

### 1. 2D 卷积原理

卷积就是**滑动窗口加权求和**，每个输出像素独立可并行：

```
输出像素(i,j) = Σ_{fx,fy} 输入像素(i+fx, j+fy) * 核(fx, fy)
```

对 5×5 核，每个输出像素需要 25 次乘加；整幅图复杂度为 O(W·H·K²)。

### 2. 高斯模糊与常量内存

源码把 5×5 高斯核（sigma=1.0，归一化，和为 1.0）放入常量内存：

```c
__constant__ float gaussian_kernel[FILTER_SIZE * FILTER_SIZE] = {
    0.0037f, 0.0147f, 0.0256f, 0.0147f, 0.0037f,
    0.0147f, 0.0585f, 0.0952f, 0.0585f, 0.0147f,
    0.0256f, 0.0952f, 0.1502f, 0.0952f, 0.0256f,
    ...
};
```

为什么用 `__constant__`？所有线程读取相同的核数据（只读），完美匹配常量内存的 **broadcast** 机制——一个 warp 读同一地址只需一次事务，且有专用 L1 缓存，延迟低。

### 3. clamp 边界处理

当卷积窗口越出图像边界时，用 clamp（钳制到边界像素）而非补零，避免黑边伪影，且无分支：

```c
int img_x = min(max(col + fx - half, 0), width - 1);
int img_y = min(max(row + fy - half, 0), height - 1);
float pixel = d_in[img_y * width + img_x];
float weight = gaussian_kernel[fy * FILTER_SIZE + fx];
sum += pixel * weight;
```

### 4. Sobel 边缘检测

Sobel 用两个 3×3 核分别求水平梯度 Gx 与垂直梯度 Gy，梯度幅值 = √(Gx²+Gy²)：

```
Gx（检测水平边缘）        Gy（检测垂直边缘）
[-1  0 +1]                [-1 -2 -1]
[-2  0 +2]                [ 0  0  0]
[-1  0 +1]                [+1 +2 +1]
```

源码中 Sobel 核是**硬编码**的（3×3 固定，不用 `FILTER_SIZE`），边界像素（最外一圈）直接置 0；结果用 `fminf(magnitude, 1.0f)` 钳制到 [0,1]。

### 5. 访存瓶颈

相邻线程访问相邻像素满足合并访问，但每个线程重复读取 K² 个邻域像素，输入图像被重复读取——这是性能瓶颈。优化方向：用共享内存 tile 缓存 block 邻域数据（类似 Day 6）。

---

## 文件结构

```
day15_image_filter/
├── src/main.cu              # 高斯模糊 Kernel + Sobel 边缘检测 Kernel
├── python/image_filter.py   # Python 版（Numba CUDA 2D 卷积，CPU 对照）
├── CMakeLists.txt           # 版本感知架构选择 + /utf-8（无需额外库）
├── build.bat                # Windows 编译脚本（生成 image_filter.exe）
├── build.sh                 # Linux/macOS 编译脚本
└── README.md                # 本文件
```

关键参数：`IMAGE_WIDTH = IMAGE_HEIGHT = 1024`、`FILTER_SIZE = 5`、`BLOCK_SIZE = 16`。测试图为 64 像素方块的黑白棋盘格（0.2/0.8 灰度），富含边缘，便于观察 Sobel 效果。

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat

```powershell
cd day15_image_filter
.\build.bat
.\build\Release\image_filter.exe
```

方式二：手动使用 CMake

```powershell
cd day15_image_filter
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\image_filter.exe
```

### Linux / macOS

```bash
cd day15_image_filter
bash build.sh
./build/image_filter
```

> 本天只用 CUDA Runtime，不依赖 cuBLAS/cuFFT 等额外库。

### Python 版本

```bash
cd day15_image_filter/python
python image_filter.py
```

> 使用 Numba CUDA JIT 实现 2D 卷积（3×3 高斯核），并与 CPU 版对比加速比与结果一致性（容差 `1e-3`）。需 `pip install numba`。

---

## 知识点总结

1. **2D 卷积**：滑动窗口加权求和，每个输出像素独立可并行，天然适合 GPU。
2. **高斯模糊**：核由二维高斯函数生成并归一化（和为 1），保证亮度不变、平滑去噪。
3. **Sobel 算子**：Gx 检测水平边缘、Gy 检测垂直边缘，幅值 = √(Gx²+Gy²)。
4. **常量内存**：`__constant__` 适合所有线程读同一只读数据，broadcast 机制高效。
5. **边界处理**：clamp 用 `min(max())` 实现，无分支发散，避免黑边伪影。
6. **2D 索引**：`dim3(16,16)` block 直接映射图像坐标，无需手动一维转二维。
7. **优化方向**：共享内存 tile 缓存邻域，减少全局内存的 K² 倍重复读取。

---

## 验收标准

- 程序打印设备信息与图像尺寸（1024×1024，约 4 MB）后正常运行
- 分别输出「高斯模糊」与「Sobel 边缘检测」两个 Kernel 的耗时（ms）
- 程序无 CUDA 错误正常退出（`CHECK_KERNEL_ERROR` 通过）
- 能解释高斯核为何要归一化、Sobel 为何对边缘敏感

---

## 动手挑战

1. **共享内存优化**：为 `gaussian_blur_kernel` 增加共享内存 tile，把 block 需要的 (16+4)×(16+4) 邻域一次载入，比较优化前后耗时。
2. **换核**：把高斯核换成锐化核（Laplacian）或均值核，观察不同滤波器对棋盘格的效果差异。
3. **可分离卷积**：高斯核是可分离的，尝试将 5×5 卷积拆成先横向 1×5 再纵向 5×1 两次一维卷积，把每像素乘加从 25 降到 10。

---

## 常见问题

- **Q：为什么高斯核放常量内存而 Sobel 核硬编码？**
  A：高斯核是 25 个可复用的只读浮点数，适合常量内存 broadcast；Sobel 核固定 3×3 且系数简单（±1、±2），直接写进表达式反而更省一次内存读取。

- **Q：clamp 边界和补零边界有什么区别？**
  A：clamp 用最近的边界像素"外扩"，避免边缘出现黑边伪影；补零（zero padding）会让边缘变暗。本例用 clamp。

- **Q：Sobel 输出为何要 `fminf(magnitude, 1.0f)`？**
  A：梯度幅值可能超过 1，钳制到 [0,1] 便于按灰度显示，避免溢出。

---

## 延伸阅读

- [CUDA C++ Programming Guide - 常量内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constant-memory)
- [维基百科：Sobel 算子](https://en.wikipedia.org/wiki/Sobel_operator)
