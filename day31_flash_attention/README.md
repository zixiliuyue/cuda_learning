# Day 31: Flash Attention 原理与实现（教学简化版）

## 学习目标

1. 理解 Transformer 注意力机制：`Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V`
2. 理解标准注意力为何存在 **O(N²) 显存瓶颈**（需要保存完整 N×N 分数矩阵）
3. 掌握 **Flash Attention** 的两大核心思想：
   - **Tiling（分块）**：把 K/V 分块加载到共享内存，从不写出完整分数矩阵
   - **Online Softmax（在线 softmax）**：一边遍历 tile 一边增量更新归一化因子
4. 理解 online softmax 的数值稳定技巧：减去 running max，`alpha = exp(m_old - m_new)` 回溯修正
5. 使用 `float4` 向量化加载（一次 128 bit）提升显存带宽利用率

> 注意：本项目是**教学简化版**，用于讲清核心原理，**不包含**反向传播（backward）、
> causal mask、block-sparse、多头/批处理等工业级 Flash Attention 的高级特性。

---

## 原理图解（文字描述）

### 1. 标准注意力（O(N²) 显存）

```
Q [N,d]                          需要一次性构造并存储
   \                             完整的 N x N 分数矩阵 S
    x K^T  --->  S [N,N]  --->  softmax(S)  --->  x V  --->  O [N,d]
   /             ^^^^^^^^
K [N,d]          显存瓶颈：N=1024 时 S 就是 1024x1024 个 float（4 MB），
                 序列越长，S 呈平方增长，很快耗尽显存与带宽。
```

### 2. Flash Attention（O(N) 显存，tiled + online softmax）

对每个 Q 行维护三个"运行状态"：`m`（running max）、`l`（running sum）、`O`（累加器）。

```
对于每个 Q 行:
    m = -inf, l = 0, O = [0...0]
    遍历 K/V 的每个 tile（分块进共享内存）:
        加载 K_tile, V_tile 到 shared memory
        S      = Q_row · K_tile^T / sqrt(d)      # 只算一个小块，不存整行
        m_new  = max(m, max(S))                  # 更新 running max
        alpha  = exp(m - m_new)                  # 旧累加量的回溯缩放因子
        P      = exp(S - m_new)                  # 当前块的指数权重
        l      = l * alpha + sum(P)              # 增量更新 softmax 分母
        O      = O * alpha + P · V_tile          # 增量更新输出累加器
        m      = m_new
    O = O / l                                    # 最终归一化
```

关键点：**每来一个新 tile，若出现更大的 max，就用 `alpha` 把之前累加的
`l` 和 `O` 统一"降权"**，这样最终结果与一次性对完整行做 softmax **完全等价**，
但全程只保存 O(d) 的运行状态，显存从 O(N²) 降到 O(N)。

---

## 文件结构

```
day31_flash_attention/
├── src/main.cu               # CUDA 实现（标准 / Flash / float4 三个 Kernel）
├── python/flash_attention.py # Python 版（torch 优先，否则 numpy）
├── CMakeLists.txt            # 版本感知架构选择 + /utf-8
├── build.bat                 # Windows 编译脚本（sm_120）
├── build.sh                  # Linux/macOS 编译脚本
└── README.md                 # 本文件
```

### 三个 Kernel 说明

| Kernel | 作用 | 特点 |
|--------|------|------|
| `standard_attention_kernel` | 标准 O(N²) 注意力 | 对比基准，每线程算一行 |
| `flash_attention_kernel` | 简化版 Flash | tiled + online softmax，共享内存缓存 K/V |
| `flash_attention_float4_kernel` | Flash + 向量化 | 使用 `float4` 加载，128 bit/指令 |

参数：`N=1024`（序列长度），`d=64`（头维度），`TILE_SIZE=32`。

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：直接用 nvcc（build.bat）

```powershell
cd day31_flash_attention
.\build.bat
.\build\flash_attention.exe
```

方式二：使用 CMake

```powershell
cd day31_flash_attention
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\flash_attention.exe
```

### Linux / macOS

```bash
cd day31_flash_attention
bash build.sh
./build/flash_attention
```

### Python 版本

```bash
cd day31_flash_attention/python
python flash_attention.py
```

> 若安装了 PyTorch，会用 torch 张量演示并与官方
> `F.scaled_dot_product_attention` 交叉验证；否则自动 fallback 到 NumPy。

---

## 知识点总结

1. **注意力公式**：`Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V`，`1/sqrt(d)` 是缩放因子，稳定分数量级。
2. **标准实现瓶颈**：需要完整的 N×N 分数矩阵，显存与带宽随序列长度平方增长。
3. **Flash 的 Tiling**：K/V 分块加载进共享内存，输出增量累加，显存降到 O(N)。
4. **Online Softmax**：维护 running max `m` 与 running sum `l`，用 `alpha = exp(m_old - m_new)` 修正历史累加，保证与全量 softmax 等价。
5. **数值稳定性**：softmax 前减去当前最大值，避免 `exp` 溢出。
6. **float4 向量化**：一次搬运 128 bit，减少访存指令、提升带宽利用率（要求地址 16 字节对齐、`d` 为 4 的倍数）。
7. **CUDA 13 兼容**：`print_device_info()` 用 `cudaDeviceGetAttribute` 查询显存带宽，避免使用已移除的 `cudaDeviceProp` 字段。

---

## 常见问题

- **结果验证为何用较大容差（1e-2）？** 因为 GPU 单精度浮点累加顺序与 CPU 不同，
  且 online softmax 涉及多次 exp/缩放，累积误差略大，属正常现象。
- **为什么本例 Flash 不一定比标准 Kernel 快很多？** 本教学版每线程处理一整行、
  规模较小（N=1024），主要目的是讲清原理；Flash 的性能优势在**长序列、
  多头、需要节省显存**时才充分体现。
