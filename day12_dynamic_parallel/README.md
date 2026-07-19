# Day 12: 动态并行与 CUDA Graphs（Dynamic Parallelism & Graphs）

## 学习目标

1. **解释**动态并行（Dynamic Parallelism）：在设备端的父 Kernel 中直接启动子 Kernel
2. **实现**父 Kernel `parent_kernel` 调用子 Kernel `child_kernel`，并与普通 Kernel 对照
3. **构建**一个 CUDA Graph：创建图、添加 Kernel 节点、实例化、重复启动
4. **对比**重复启动 100 次时 "CUDA Graph" 与 "直接逐次启动" 的耗时，理解启动开销
5. **描述**动态并行与 CUDA Graphs 各自的适用场景与硬件/编译要求

---

## 前置依赖

- **Day 3 线程层级 / Day 4 Kernel 启动**：理解 `<<<grid, block>>>` 启动语法
- **Day 10 流与异步**：Graph 通过流捕获/启动，与流机制相关
- **计算能力要求**：动态并行需要 Compute Capability 3.5+ 与可分离编译（`-rdc=true`）

---

## 核心概念

### 动态并行：Kernel 里启动 Kernel

传统模式下所有 Kernel 由 CPU（host）启动。动态并行允许**设备端**的线程根据运行时数据自行启动新的子 Kernel，适合递归、树遍历、自适应网格等"工作量运行时才确定"的场景，减少 CPU-GPU 往返。

```
普通模式:                        动态并行:
  Host ──启动──► Kernel            Host ──启动──► parent_kernel
                                                     │ 每个线程按需
                                                     └──启动──► child_kernel<<<1,1>>>
```

本天 `src/main.cu`（`ARRAY_SIZE = 1024`，`BLOCK_SIZE = 256`）中：

```c
__global__ void child_kernel(float* data, int idx) {
    data[idx] = data[idx] * 2.0f + 1.0f;
}
__global__ void parent_kernel(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        child_kernel<<<1, 1>>>(data, tid);   // 设备端启动子 Kernel
}
__global__ void normal_kernel(float* data, int n) {   // 对照：无动态并行
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] = data[tid] * 2.0f + 1.0f;
}
```

程序启动时会检查 `prop.major/minor`，若设备不支持动态并行（低于 CC 3.5）会打印警告并跳过该部分，仅演示 CUDA Graphs。

### CUDA Graphs：预定义执行流程降低启动开销

当同一组操作要**重复执行很多次**时，逐次 `<<<>>>` 启动会累积可观的 CPU 提交开销。CUDA Graph 把工作流"录制"成一张图（节点 = 操作，边 = 依赖），实例化后可以极低开销地重复 `cudaGraphLaunch`。

```
直接启动 100 次:   [启动开销][Kernel] × 100    ← 每次都有提交开销
CUDA Graph:        构建 + 实例化(一次)  →  [Launch][Kernel] × 100  ← 提交开销更低
```

本天构建 Graph 的核心步骤：

```c
cudaGraphCreate(&graph, 0);                       // 1. 创建空图
// 2. 配置一个 Kernel 节点（注意宏 ARRAY_SIZE 是右值，需用局部变量承载才能取地址）
int array_size_arg = ARRAY_SIZE;
void* kernel_args[2] = { (void*)&d_data, (void*)&array_size_arg };
cudaKernelNodeParams kernel_params = {0};
kernel_params.func = (void*)normal_kernel;
kernel_params.gridDim = dim3(grid_size, 1, 1);
kernel_params.blockDim = dim3(BLOCK_SIZE, 1, 1);
kernel_params.kernelParams = kernel_args;
cudaGraphAddKernelNode(&kernel_node, graph, NULL, 0, &kernel_params);  // 3. 添加节点
cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);               // 4. 实例化
for (int i = 0; i < 100; i++) cudaGraphLaunch(graph_exec, 0);          // 5. 重复启动
```

程序会对比 "CUDA Graph 100 次" 与 "直接启动 100 次" 的总耗时。

### Python 版

`python/dynamic_parallel.py` 说明 Numba 不完全支持 CUDA 动态并行（CDP），改用多流 + 分段/递归式 Kernel 启动来**模拟**动态并行的概念（`process_segment_kernel` 分段处理，`sum_kernel` 归约汇总，`DEPTH = 4` 模拟递归深度）。

---

## 文件结构

```
day12_dynamic_parallel/
├── src/main.cu                 # child/parent/normal Kernel（动态并行）+ CUDA Graph 构建与对比
├── python/dynamic_parallel.py  # Python/Numba 版（多流模拟动态并行）
├── CMakeLists.txt              # CMake 构建配置（动态并行需 -rdc=true / separable compilation）
├── build.bat                   # Windows 编译脚本（调用 CMake + VS2022）
├── build.sh                    # Linux/macOS 编译脚本
└── README.md                   # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat

```powershell
cd day12_dynamic_parallel
.\build.bat
.\build\Release\dynamic_parallel.exe
```

方式二：手动使用 CMake

```powershell
cd day12_dynamic_parallel
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\dynamic_parallel.exe
```

### Linux / macOS

```bash
cd day12_dynamic_parallel
bash build.sh
./build/dynamic_parallel
```

### Python 版本

```bash
cd day12_dynamic_parallel/python
python dynamic_parallel.py
```

> 需要安装 Numba：`pip install numba`。

---

## 知识点总结

1. **动态并行**：设备端 Kernel 可用 `<<<>>>` 启动子 Kernel，适合递归/自适应/运行时决定工作量的算法。
2. **编译要求**：动态并行需要 Compute Capability 3.5+ 并开启可分离编译（`-rdc=true`）。
3. **CUDA Graph 组成**：节点表示操作（Kernel/拷贝/内存操作），边表示依赖关系。
4. **构建流程**：`cudaGraphCreate` → 添加节点（如 `cudaGraphAddKernelNode`）→ `cudaGraphInstantiate` → 反复 `cudaGraphLaunch`。
5. **降低启动开销**：Graph 把提交开销前置到实例化阶段，重复执行时启动延迟更低。
6. **只读的可执行图**：`graph_exec` 实例化后是只读的，若要改结构须修改 `graph` 再重新实例化。
7. **参数取地址陷阱**：宏（右值）不能取地址，需先用局部变量承载（如 `int array_size_arg = ARRAY_SIZE;`）再传入 `kernelParams`。

---

## 验收标准

- 程序打印普通 Kernel 耗时，以及 "CUDA Graph (100 次)" 与 "直接启动 (100 次)" 的耗时对比。
- 对于重复启动很多次的场景，**CUDA Graph 的总耗时通常不高于（往往低于）逐次直接启动**，体现启动开销的节省。
- 若设备支持动态并行，父/子 Kernel 正常运行；不支持时程序打印警告并跳过、继续 Graph 演示，且无崩溃。
- 你能解释：为什么 Graph 在"重复执行相同工作流"时收益最明显。

---

## 动手挑战

1. **加大重复次数**：把 Graph 与直接启动的循环次数从 100 提高到 1000 / 10000，观察 Graph 的相对优势是否随次数增大而更明显。
2. **多节点图**：向 Graph 再添加一个 Kernel 节点或一个 `cudaGraphAddMemcpyNode`（H2D/D2H），用依赖关系连接两个节点，构建一条真正的执行链。
3. **验证动态并行正确性**：把 `parent_kernel` 的结果拷回主机，与 `normal_kernel` 结果逐元素比对，确认动态并行版本计算正确（`data[i]*2+1`）。

---

## 常见问题

- **Q：编译动态并行报链接错误怎么办？**
  A：动态并行需要可分离编译与设备端运行时库。请确认 CMake/nvcc 开启了 `-rdc=true`（separable compilation）并链接 `cudadevrt`。

- **Q：动态并行一定更快吗？**
  A：不一定。设备端启动子 Kernel 也有开销，`child_kernel<<<1,1>>>` 这种极小规模启动往往比普通 Kernel 慢；它的价值在于**表达递归/自适应逻辑**而非单纯提速。

- **Q：CUDA Graph 和 CUDA Stream 是什么关系？**
  A：Graph 可从流"捕获"生成，也可手工添加节点构建；`cudaGraphLaunch` 需要在一个流上执行。Graph 侧重"固定工作流的低开销重复"，流侧重"异步与并发调度"。

---

## 延伸阅读

- [NVIDIA Developer Blog — Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
- [CUDA C++ Programming Guide — CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism)
