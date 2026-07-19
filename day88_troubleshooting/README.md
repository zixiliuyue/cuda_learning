# Day 88: Linux 工程化 — GPU 异常排查 / coredump / compute-sanitizer

> 🛠️ **第十三阶段（通信·存储·工程化）工程素养专段**。写得出 kernel 只是入门，线上能**快速定位问题**才是 AI Infra 的核心竞争力。本天系统梳理 OOM / 非法访存 / hang / NaN 四大高频问题的排查方法，并用可运行的诊断工具演示 NaN 钩子与显存监控。

## 学习目标

1. **建立** GPU 问题的排查地图：症状 → 定位工具 → 修复思路
2. **实现** NaN/Inf 检测钩子，自动定位「哪一层最先产生异常值」
3. **掌握**显存估算公式与 OOM 的正确处理流程（捕获 → 诊断 → 优化）
4. **会用** compute-sanitizer / cuda-gdb / GPU coredump / py-spy 等工程化排查工具

---

## 前置依赖

- **Day 4 错误处理**：CUDA 错误码与检查宏，本天在此之上讲运行期排查
- **Day 35 Nsight 深度分析**：性能类问题（hang/骤降）常配合 nsys/ncu
- **Day 26 AMP / Day 53 量化 / Day 75 梯度检查点**：OOM 的常见修复手段

---

## 核心概念

### 四大高频问题速查

| 问题 | 典型症状 | 定位工具 | 常见修复 |
|------|----------|----------|----------|
| OOM | `CUDA out of memory` | `nvidia-smi` / `memory_summary()` | 减 batch / 梯度检查点 / AMP / 量化 |
| 非法访存 | `illegal memory access`（常延迟报错） | `compute-sanitizer` / `cuda-gdb` | 修越界索引 / 边界检查 / 同步 |
| hang | 进程不退、利用率异常 | `py-spy` / `nsys` / `gdb` | 死锁 / 通信不对齐 |
| NaN/Inf | `loss=nan`、梯度爆炸 | NaN 钩子 / `detect_anomaly` | 降 lr / grad clip / 查 log(0)、除0、溢出 |

### NaN 钩子：定位「首个出问题的层」

`loss=nan` 时，根因往往在某一层。用 `register_forward_hook` 逐层检查输出的 `isnan/isinf`，**第一个报警的层就是最可疑的源头**。反向 NaN 则用 `torch.autograd.set_detect_anomaly(True)`。

### 非法访存为什么难查

CUDA kernel 的错误常**延迟报告**（异步执行 + 错误在下一次同步才抛出），报错行号未必是真凶。`compute-sanitizer --tool memcheck` 能精确定位越界读写、未初始化访问、竞争等。

### OOM 显存构成

```
总显存 ≈ 参数 + 梯度 + 优化器状态(如 Adam 2x) + 激活 + KV Cache + 碎片余量
```

估算单个张量：`元素数 × 每元素字节`（fp32=4, fp16/bf16=2, int8=1）。

---

## 文件结构

```
day88_troubleshooting/
├── python/troubleshooting.py   # 速查表 + NaN 钩子 + 显存估算/OOM + 工具用法
├── build.sh                    # Linux/WSL2 运行脚本
├── build.bat                   # Windows 运行脚本
└── README.md                   # 本文件
```

---

## 编译与运行

### Linux / WSL2

```bash
cd day88_troubleshooting
python python/troubleshooting.py
# 进阶：用 sanitizer 包一层运行
compute-sanitizer --tool memcheck python python/troubleshooting.py
```

### Windows

```powershell
cd day88_troubleshooting
.\build.bat
```

> 代码部分（NaN 钩子 / 显存估算）在任意环境可跑；`compute-sanitizer`/`cuda-gdb`/coredump 需 Linux + CUDA 工具链。

---

## 关键命令速查

```bash
nvidia-smi                                  # 显存/进程/利用率/温度
compute-sanitizer --tool memcheck ./app     # 非法访存/未初始化
compute-sanitizer --tool racecheck ./app    # 共享内存竞争
cuda-gdb --args python train.py             # GPU 断点调试, bt 看栈
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1  # 开启 GPU coredump
ulimit -c unlimited                         # 允许生成 coredump
py-spy dump --pid <PID>                      # 无侵入抓 Python 栈(查 hang)
```

---

## 知识点总结

1. **排查地图**：先分类（OOM/访存/hang/NaN），再选对应工具，避免盲目试错。
2. **NaN 钩子**：forward hook 逐层查 `isnan/isinf`，第一个报警层即嫌疑源。
3. **延迟报错**：CUDA 错误异步抛出，报错位置不可全信，用 sanitizer 定位真凶。
4. **compute-sanitizer**：memcheck/racecheck/synccheck 三工具覆盖访存与并发问题。
5. **coredump**：`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` + `ulimit -c unlimited` 保存崩溃现场。
6. **OOM 优化阶梯**：减 batch → 混合精度 → 梯度检查点 → 量化 → 分布式切分。

---

## 验收标准

- 程序打印四大问题速查表，你能对每类说出「用什么工具、怎么修」。
- NaN 钩子能准确报出「fc2 / 制造 Inf 的那一层」是首个异常层。
- 你能用显存估算公式预判一个 batch 是否会 OOM。
- 你能说出 `compute-sanitizer` 三个子工具的用途，以及 hang 时该抓什么栈。

---

## 动手挑战

1. **反向 NaN**：给演示网络加一个会产生反向 NaN 的算子（如 `sqrt(负数)`），用 `set_detect_anomaly(True)` 定位。
2. **写越界 kernel**：写一个故意越界的 CUDA/Triton kernel，用 `compute-sanitizer` 抓出来。
3. **显存预算器**：把显存估算扩展成一个函数，输入模型规模/batch/序列长度，输出预计显存与是否 OOM。

---

## 常见问题

- **Q：报错行号指向的代码看起来没问题？**
  A：CUDA 异步执行导致错误延迟报告。用 `CUDA_LAUNCH_BLOCKING=1` 让 kernel 同步执行，或用 `compute-sanitizer` 精确定位。

- **Q：多卡训练卡死怎么查？**
  A：先 `py-spy dump` 抓各 rank 的 Python 栈，看是否卡在某次 collective；再核对各 rank 的通信调用形状/次数是否一致。

- **Q：Windows 能用 compute-sanitizer 吗？**
  A：CUDA 工具链在 Windows 上也有，但工程化排查以 Linux 为主，建议 WSL2。本天代码部分 Windows 可直接跑。

---

## 延伸阅读

- [compute-sanitizer 官方文档](https://docs.nvidia.com/compute-sanitizer/)
- [cuda-gdb 官方文档](https://docs.nvidia.com/cuda/cuda-gdb/)
- [PyTorch 显存管理与 OOM 调试](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [py-spy（无侵入 Python 性能/栈抓取）](https://github.com/benfred/py-spy)
