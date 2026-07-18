# Day 5: GPU 设备查询与性能基准

## 学习目标

1. 使用 `cudaGetDeviceProperties` 获取设备详细信息
2. 理解 GPU 硬件参数的含义（SM数、Warp大小、寄存器等）
3. 计算 GPU 的理论峰值性能
4. 运行内存带宽基准测试
5. 了解 Compute Capability 与功能特性的关系

## 核心概念

### 关键硬件参数

| 参数 | 含义 |
|------|------|
| Compute Capability | 计算能力版本，决定支持的 CUDA 特性 |
| SM | Streaming Multiprocessor，GPU 的核心计算单元 |
| Warp | 32 个线程组成的执行单元 |
| 共享内存 | 每个 SM 上的高速缓存 |
| 寄存器 | 每个线程的私有高速存储 |

### Compute Capability 演进

| CC | 架构 | 关键特性 |
|----|------|----------|
| 3.x | Kepler | 统一内存、动态并行 |
| 5.x | Maxwell | FP16 |
| 6.x | Pascal | Pascal 架构优化 |
| 7.x | Turing/Volta | Tensor Core、独立线程调度 |
| 8.x | Ampere | 第三代 Tensor Core |
| 9.x | Ada Lovelace | Ada 架构优化 |
| 10.x+ | Blackwell | FP8/FP4、第五代 Tensor Core |

## 编译与运行

```powershell
cd day05_device_query
.\build.bat
.\build\Release\device_query.exe
```
