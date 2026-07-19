# Day 28: GPU 光线追踪（简易版）

> 🏗️ **综合大项目 · 建议 2-3 天完成**
> 本项目把「每个像素一个线程」的二维并行范式用到渲染上，实现一个最小可运行的光线追踪器。
> 建议按「动手挑战」的里程碑推进：**MVP（渲染球体）→ 优化（真实光线-球体求交 + 光照）→ 可视化（输出并显示图像）**。

## 学习目标

1. **实现**二维线程网格映射（每个线程渲染一个像素），理解 `dim3` block/grid 的配置
2. **实现**光线与球体求交，并对多个球体取**最近命中**，计算简单明暗着色
3. **解释**光线追踪为何天然适合 GPU 大规模并行（像素之间完全独立、无数据依赖）
4. **测量** 512×512 图像的渲染耗时，并**对比** GPU 与 CPU（Python 版小图）的性能差距
5. **对比** CUDA 版的正交近似着色与 Python/Numba 版的完整光线-球体二次方程求交 + 漫反射光照

---

## 前置依赖

- **Day 03 矩阵加法**：二维 grid/block 索引（row/col）
- **Day 07 合并访问**：像素缓冲区的写入模式
- **Day 15 图像滤波**：二维图像上的逐像素并行处理

---

## 核心概念

### 1. 二维线程映射（每像素一线程）

渲染核心是把图像的每个像素交给一个 GPU 线程处理（见 [main.cu Kernel 入口](file:///f:/code/cuda_learning/day28_raytracing/src/main.cu#L38-L45)）：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (col >= width || row >= height) return;
// 归一化到 [-1,1] 屏幕坐标
float x = (float)col / width * 2.0f - 1.0f;
float y = (float)row / height * 2.0f - 1.0f;
```

执行配置为 `dim3 block(16,16)` + 覆盖整幅图的二维 grid，`IMAGE_WIDTH=IMAGE_HEIGHT=512`，`BLOCK_SIZE=16`。

```
图像 512x512
 ┌───────────────────────┐
 │ ■ ■ ■ ... 每个■=16x16 block │
 │ 每个 block 内 16x16=256 线程 │
 │ 每个线程 = 1 个像素          │
 └───────────────────────┘
```

### 2. 光线-球体求交（CUDA 版：正交近似）

CUDA 版用一个轻量的正交投影相交函数 [sphere_intersect](file:///f:/code/cuda_learning/day28_raytracing/src/main.cu#L24-L36)：判断屏幕点 `(ox,oy)` 是否落在球体投影圆内，若在则由 `dz = sqrt(r² - dx² - dy²)` 反算深度与法线分量 `n = dz/r`（用于明暗）：

```cpp
if (dx*dx + dy*dy < r*r) {
    float dz = sqrtf(r*r - dx*dx - dy*dy);
    *n = dz / rad;          // 法线 z 分量，作为亮度
    return dz + s.z;        // 深度值
}
return -1.0f;               // 未命中
```

主 Kernel 遍历 3 个球体（见 [main.cu 球体数组](file:///f:/code/cuda_learning/day28_raytracing/src/main.cu#L47-L67)），保留**深度最小（最近）**的命中，命中则用 `shade = 255*n` 输出灰度，未命中则填天蓝背景 `(50,100,150)`。

### 3. Python/Numba 版：完整光线-球体求交

Python 版 [raytrace_kernel](file:///f:/code/cuda_learning/day28_raytracing/python/raytracing.py#L33-L118) 实现的是标准光线追踪：对每像素构造光线方向，解二次方程 `at²+bt+c=0` 求交，取最近正根，计算法线并做**环境光 + 漫反射**着色（8 个彩色球体）：

```
|O + t·D - C|² = r²  →  a t² + b t + c = 0
判别式 disc = b² - 4ac ≥ 0 才有交点
最近交点 t = (-b - √disc) / 2a
颜色 = 球色 × (ambient + (1-ambient)·max(0, N·L))
```

---

## 文件结构

```
day28_raytracing/
├── src/main.cu          # CUDA 版：正交近似求交 + 灰度着色（3 球体，512x512）
├── python/raytracing.py # Numba 版：完整光线-球体二次求交 + 漫反射（8 彩球）+ CPU 验证
├── CMakeLists.txt       # 构建配置
├── build.bat            # Windows 编译脚本（生成 raytracing.exe）
├── build.sh             # Linux/macOS 编译脚本
└── README.md            # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：build.bat

```powershell
cd day28_raytracing
.\build.bat
.\build\Release\raytracing.exe
```

方式二：手动 CMake

```powershell
cd day28_raytracing
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\raytracing.exe
```

### Linux / macOS

```bash
cd day28_raytracing
bash build.sh
./build/raytracing
```

### Python 版本

```bash
cd day28_raytracing/python
python raytracing.py   # 需要 numba；GPU 渲染 512x512，CPU 渲染 64x64 做验证
```

---

## 知识点总结

1. **每像素一线程**：光线追踪像素间无依赖，是 GPU 并行的理想负载。
2. **二维 grid/block**：用 `dim3(16,16)` 划分图像，`(width+15)/16` 计算 grid 尺寸并做边界检查。
3. **最近命中**：遍历所有物体、保留深度最小的交点，才能正确处理遮挡。
4. **光线-球体求交**：本质是解二次方程（Python 版），判别式 ≥0 表示相交。
5. **着色模型**：环境光 + 漫反射 `N·L`（Lambert）是最基础的光照；CUDA 版用法线 z 分量近似亮度。
6. **RGB 缓冲**：输出按 `(row*width+col)*3` 的交错布局写 3 通道。
7. **进阶方向**（程序末尾打印）：BVH 加速结构、RT Core 硬件光追、路径追踪全局光照、OptiX SDK。

---

## 验收标准

- 程序打印渲染尺寸 `512 x 512`。
- 打印 `光线追踪渲染` 耗时（毫秒级），程序无 CUDA 错误、正常退出。
- 末尾打印四条光线追踪优化方向。
- Python 版打印 GPU 渲染耗时、像素吞吐（像素/ms），并显示 `结果验证: 通过 / 差异可接受`（阈值 `1.0`）。

---

## 动手挑战（分阶段里程碑）

### 里程碑 1 — MVP：把结果存成图片

- CUDA 版目前只在显存/主机内存里得到 `h_output`，并未落盘。补一段把 `h_output` 写成 PPM 图像（`P6` 头 + 原始 RGB 字节）的代码，用看图软件打开确认能看到 3 个球体。

### 里程碑 2 — 优化：升级为真实光线 + 光照

- 参考 Python 版，把 CUDA 版的正交近似改为**真实光线-球体二次方程求交**，并加入环境光 + 漫反射着色（`N·L`）。对比渲染效果与耗时变化。
- 进阶：加入镜面高光（Phong）或简单阴影射线（从交点向光源发射一条遮挡测试光线）。

### 里程碑 3 — 可视化：交互 / 动画

- 让相机位置或球体位置随帧变化，连续渲染多帧输出为图片序列，合成动画（如用 Python 读取 PPM 序列生成 GIF）。
- 进阶：把球体数量提高到几十个并加一层地面平面，观察渲染耗时随物体数量的增长趋势。

---

## 常见问题

- **Q：CUDA 版和 Python 版结果看起来不一样？** A：两者渲染模型不同——CUDA 版是正交投影 + 灰度近似（3 球），Python 版是完整光线求交 + 漫反射彩色（8 球）。它们分别演示「最简可运行」与「标准做法」。
- **Q：为什么用 16×16 的 block？** A：256 线程/块是常用的二维图像分块尺寸，既能保证足够占用率，又让相邻线程访问相邻像素、利于合并写入。
- **Q：渲染出来一片背景色（没有球）？** A：检查球体的 z 坐标与屏幕坐标范围是否匹配，以及最近命中判断 `t < closest_t` 的初值是否够大。

---

## 延伸阅读

- [Ray Tracing in One Weekend（经典入门教程）](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
- [NVIDIA OptiX 光线追踪 SDK](https://developer.nvidia.com/rtx/ray-tracing/optix)
