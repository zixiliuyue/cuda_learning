# 🚀 Speed-Demon-Cuda 快速入门指南

## 📋 目录

1. [环境准备](#环境准备)
2. [项目构建](#项目构建)
3. [第一个CUDA程序](#第一个cuda程序)
4. [运行基准测试](#运行基准测试)

---

## 🔧 环境准备

### Windows 环境

#### 1. 安装 CUDA Toolkit
```
1. 访问 https://developer.nvidia.com/cuda-downloads
2. 选择 Windows -> x86_64 -> 11.0+ 版本
3. 下载并安装（默认安装路径即可）
```

#### 2. 验证安装
```cmd
nvcc --version
nvidia-smi
```

#### 3. 安装 CMake
- 下载 CMake 3.18+
- 添加到系统 PATH

#### 4. 安装 Visual Studio 2022
- 选择 "使用C++的桌面开发" 工作负载

### Linux 环境

#### 1. 安装 CUDA Toolkit
```bash
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

#### 2. 配置环境变量
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 🛠️ 项目构建

### Windows

```cmd
cd Speed-Demon-Cuda
build.bat
```

### Linux/Mac

```bash
cd Speed-Demon-Cuda
chmod +x build.sh
./build.sh
```

### 手动构建（可选）

```bash
mkdir build
cd build
cmake ..
make -j8  # Linux
# cmake --build . --config Release  # Windows
```

---

## 💻 第一个CUDA程序

### 1. 创建 hello_cuda.cu

```cpp
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_utils.h"

__global__ void hello_kernel() {
    printf("Hello from GPU! Thread %d\n", threadIdx.x);
}

int main() {
    std::cout << "Hello from CPU!" << std::endl;
    
    hello_kernel<<<1, 4>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return 0;
}
```

### 2. 编译运行

```bash
nvcc hello_cuda.cu -o hello_cuda
./hello_cuda
```

### 3. 预期输出

```
Hello from CPU!
Hello from GPU! Thread 0
Hello from GPU! Thread 1
Hello from GPU! Thread 2
Hello from GPU! Thread 3
```

---

## 📊 运行基准测试

### Windows

```cmd
run_all_benchmarks.bat
```

### Linux/Mac

```bash
chmod +x run_all_benchmarks.sh
./run_all_benchmarks.sh
```

### 查看结果

```bash
cd benchmark_results
# 查看最新的结果文件
```

---

## 🎯 下一步

1. 浏览 [01_theory/](01_theory/) 学习理论基础
2. 查看 [02_practice/](02_practice/) 实践示例
3. 阅读 [CHEATSHEET.md](CHEATSHEET.md) 快速查阅常用命令

---

## ❓ 常见问题

### Q: CUDA 报错 "no kernel image is available"
A: 检查 CMakeLists.txt 中的 CUDA_ARCHITECTURES 设置，确保与你的 GPU 匹配。

### Q: 编译找不到 cuda_runtime.h
A: 确保 CUDA Toolkit 已正确安装，并且 CMake 能找到 CUDA 路径。

---

**🚀 开始你的 CUDA 学习之旅吧！**
