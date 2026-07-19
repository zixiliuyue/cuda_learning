# Day 23: OpenCV CUDA 图像处理

> **重要说明**：本天的 `src/main.cu` 用**原生 CUDA Kernel** 手写 RGB→灰度、3×3 高斯模糊，
> 展示 OpenCV CUDA 模块（如 `cv::cuda::cvtColor`、`cv::cuda::GaussianBlur`）**底层等效的实现**——
> 编译它无需 OpenCV。而 [python/opencv_cuda.py](file:///f:/code/cuda_learning/day23_opencv_cuda/python/opencv_cuda.py)
> 则直接调用 `opencv-python`（`cv2`）做高斯模糊与 Sobel 边缘检测，未装 OpenCV 时回退到 NumPy 实现。

## 学习目标

1. **实现** 二维网格/线程块映射到图像像素（`dim3` block/grid），处理 `1920×1080` 图像
2. **实现** RGB→灰度转换 Kernel，理解 BT.601 亮度公式 `Y = 0.299R + 0.587G + 0.114B`
3. **实现** 3×3 高斯模糊 Kernel，理解卷积核加权与归一化（权重和 16）
4. **对应** 手写 Kernel 与 OpenCV CUDA API（`cvtColor`、`GaussianBlur`、`Sobel`）的功能关系
5. **对比** OpenCV 与 NumPy 实现同一图像操作的耗时，体会 GPU 加速的价值

---

## 前置依赖

- **Day 13/14（二维索引 / Stencil）**：二维线程映射与邻域（stencil）访问
- **Day 07/08（共享内存）**：卷积可用共享内存缓存邻域（本天挑战方向）
- **Day 05（逐元素 Kernel）**：RGB→灰度是逐像素独立操作

---

## 核心概念

### 1. 二维像素映射

图像天然是二维的，用 `dim3` 把线程组织成二维网格（[main.cu](file:///f:/code/cuda_learning/day23_opencv_cuda/src/main.cu#L94-L95)）：

```cpp
dim3 block_dim(16, 16);   // BLOCK_SIZE=16，每块 16×16=256 线程
dim3 grid_dim((width + 15) / 16, (height + 15) / 16);
// Kernel 内：
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int idx = row * width + col;   // 行主序线性下标
```

### 2. RGB→灰度（对应 cv::cuda::cvtColor）

[rgb_to_gray_kernel](file:///f:/code/cuda_learning/day23_opencv_cuda/src/main.cu#L23-L34) 每线程处理一个 `uchar3` 像素，按 BT.601 标准加权：

```cpp
uchar3 pixel = d_rgb[idx];
d_gray[idx] = (unsigned char)(0.299f*pixel.x + 0.587f*pixel.y + 0.114f*pixel.z);
```

人眼对绿色最敏感，故绿色权重（0.587）最大——这正是标准灰度转换的由来。

### 3. 3×3 高斯模糊（对应 cv::cuda::GaussianBlur）

[gaussian_blur_3x3_kernel](file:///f:/code/cuda_learning/day23_opencv_cuda/src/main.cu#L39-L58) 用固定的 3×3 高斯核做加权平均：

```
高斯核 (权重和 = 16):        每个输出像素 = Σ(邻域像素 × 权重) / 16
    1  2  1
    2  4  1        中心权重最大，越远越小 → 平滑降噪
    1  2  1
```

```cpp
float kernel[9] = {1,2,1, 2,4,2, 1,2,1};
for (dy=-1..1) for (dx=-1..1)
    sum += d_in[(row+dy)*width + (col+dx)] * kernel[(dy+1)*3 + (dx+1)];
d_out[row*width+col] = (unsigned char)(sum / 16.0f);
```

注意边界处理：Kernel 只对 `1 <= col < width-1 && 1 <= row < height-1` 的内部像素计算，跳过最外圈以免越界。

### 4. Python 层：直接用 OpenCV / NumPy

[opencv_cuda.py](file:///f:/code/cuda_learning/day23_opencv_cuda/python/opencv_cuda.py) 生成 `512×512` 测试图（棋盘格+渐变+噪声），若装了 `cv2` 则用 `cv2.GaussianBlur`、`cv2.Sobel` + `cv2.magnitude`，并与纯 NumPy 卷积实现对比耗时；未装则仅跑 NumPy Sobel。

---

## 文件结构

```
day23_opencv_cuda/
├── src/main.cu             # 原生 CUDA：RGB→灰度 + 3×3 高斯模糊 Kernel
├── python/opencv_cuda.py   # opencv-python 高斯模糊/Sobel，NumPy fallback 对比
├── CMakeLists.txt          # 构建配置
├── build.bat               # Windows 构建脚本（CMake + VS 2022）
├── build.sh                # Linux/macOS 构建脚本
└── README.md               # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）—— 原生 CUDA 版

方式一：使用 build.bat（内部调用 CMake，产物在 `build\Release\`，可执行文件名 `opencv_cuda_demo.exe`）

```powershell
cd day23_opencv_cuda
.\build.bat
.\build\Release\opencv_cuda_demo.exe
```

方式二：手动 CMake

```powershell
cd day23_opencv_cuda
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\opencv_cuda_demo.exe
```

### Linux / macOS

```bash
cd day23_opencv_cuda
bash build.sh
./build/opencv_cuda_demo
```

### Python 版本（OpenCV）

```bash
cd day23_opencv_cuda/python
python opencv_cuda.py
```

> Python 版需要 **OpenCV**（`pip install opencv-python`，GPU 版为 `opencv-contrib-python` 并需编译时启用 CUDA）。
> 未安装时自动 fallback 到 NumPy，仅运行 Sobel 边缘检测演示。

---

## 知识点总结

1. **二维线程映射**：`dim3` block/grid 把线程对应到 `(row, col)` 像素，`idx = row*width + col`。
2. **BT.601 灰度**：`Y = 0.299R + 0.587G + 0.114B`，绿色权重最高符合人眼感知。
3. **卷积/Stencil**：模糊、边缘检测都是邻域加权，需处理边界（本例跳过最外圈）。
4. **高斯核归一化**：权重和为 16，除以 16 保持亮度不变。
5. **OpenCV CUDA 对应**：`cvtColor`↔灰度 Kernel，`GaussianBlur`↔模糊 Kernel，`Sobel`↔边缘检测。
6. **GpuMat 与内存**：OpenCV CUDA 用 `cv::cuda::GpuMat` 管理显存图像，避免频繁主机-设备拷贝。

---

## 验收标准

- 运行 `opencv_cuda_demo.exe`，看到 `图像尺寸: 1920 x 1080`、`RGB 转灰度` 与 `高斯模糊` 两个计时输出，程序正常退出。
- 能说出高斯核权重 `{1,2,1;2,4,2;1,2,1}` 除以 16 的含义（归一化保亮度）。
- Python 版（装了 OpenCV）输出 `OpenCV 高斯模糊`、`NumPy 高斯模糊` 耗时对比及 `高斯模糊加速比`——OpenCV 通常远快于逐像素 NumPy 循环。
- Python 版无 OpenCV 时能 fallback 到 NumPy Sobel 并打印输出图像形状。

---

## 动手挑战

1. **加 Sobel 边缘检测 Kernel**：仿照高斯模糊，在 main.cu 里写 `sobel_kernel`，对灰度图计算 `sqrt(Gx²+Gy²)`（对应 Python 版的 `cv2.Sobel`）。
2. **共享内存优化模糊**：把 3×3 模糊改成先把 block 邻域（含 halo）加载进共享内存，减少全局内存重复读取，对比耗时。
3. **验证灰度正确性**：把灰度结果拷回主机，与 CPU 逐像素 BT.601 计算对比，确认误差在 ±1（取整）以内。

---

## 常见问题

- **Q：为什么高斯模糊 Kernel 跳过图像最外一圈像素？**
  A：3×3 核需要访问 `col±1, row±1`，对最边缘像素会越界；教学版直接跳过边界。生产实现会用边界填充（clamp / reflect）。

- **Q：`pip install opencv-python` 装的版本有 CUDA 吗？**
  A：一般没有。PyPI 的预编译包默认不含 CUDA 模块（`cv2.cuda`）；要用 GPU 版需自行用 `WITH_CUDA` 编译 OpenCV，或用带 CUDA 的发行版。

- **Q：本天 main.cu 的输入是随机像素，看不到"图像效果"怎么办？**
  A：它侧重演示 Kernel 与计时。想看真实效果可改成读入真实图片（用 stb_image 或经 Python 侧的 OpenCV）再跑 Kernel。

---

## 延伸阅读

- [OpenCV CUDA Module 官方文档](https://docs.opencv.org/4.x/d1/d1e/group__cuda.html) — `cv::cuda` GPU 加速模块参考
- [OpenCV — Getting Started with CUDA](https://docs.opencv.org/4.x/d2/d1f/tutorial_cuda_intro.html) — GpuMat 与 CUDA 流入门
