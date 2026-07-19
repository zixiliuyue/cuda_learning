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

---

## 验收标准

达到以下几点，说明你已掌握本天内容：

1. 程序打印出「检测到 N 个 CUDA 设备」，并对每个设备输出完整的【基本信息】【计算资源】【内存资源】【维度限制】【特性支持】五大分区，能读懂 SM 数量、Warp 大小、每 Block 最大线程数等字段。
2. 能从输出中读出本机 GPU 的计算能力（CC major.minor），并对照【Compute Capability x.x 特性】列表判断它是否支持 FP16、Tensor Core、FP8/FP4 等特性。
3. 「显存带宽基准测试」表格针对 16/64/256/1024 MB 四档数据，打印出 H2D、D2H、设备内拷贝三列带宽（GB/s），且随数据增大带宽趋于稳定并接近理论显存带宽量级。
4. 理解程序中理论显存带宽的计算式 `2 * 显存频率 * 总线宽度 / 8`，以及为何设备内拷贝带宽按 `bytes * 2`（一读一写）计算。

## 动手挑战

基于 [main.cu](src/main.cu) 中的 `print_device_details` 与 `run_bandwidth_benchmark` 循序渐进地练习：

1. **扩展基准数据档位**：修改 `sizes_mb[]` 数组（如新增 4、2048 MB），观察小数据下 H2D/D2H 带宽偏低（固定启动开销占比大）、大数据下带宽趋于饱和的现象。
2. **计算带宽利用率**：在带宽测试中把实测的「设备内拷贝带宽」与 `print_device_details` 算出的「理论显存带宽」相除，打印出带宽利用率百分比，评估访存效率。
3. **改进带宽测试 Kernel**：让 `bandwidth_test_kernel` 采用 grid-stride loop（每个线程处理多个元素），或改用 `float4` 向量化访问，对比设备内拷贝带宽是否提升。
