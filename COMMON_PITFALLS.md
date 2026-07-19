# CUDA 新手常见踩坑点（Common Pitfalls）

> 本文档汇总 CUDA / cuda_learning 项目学习过程中最容易踩到的坑，覆盖**编译、运行时、性能、调试、PyTorch 扩展**五大类。
> 每个条目遵循统一格式：**问题** → **原因** → **解决方案** → **参考代码/文件**。
>
> 适用环境：Windows + MSVC + CUDA Toolkit（本项目主力平台），大部分内容同样适用于 Linux。

---

## 目录

- [一、编译问题](#一编译问题)
- [二、运行时问题](#二运行时问题)
- [三、性能问题](#三性能问题)
- [四、调试技巧](#四调试技巧)
- [五、PyTorch CUDA 扩展](#五pytorch-cuda-扩展)

---

## 一、编译问题

### 1.1 nvcc 找不到 cl.exe（MSVC 自动检测失败）

- **问题**：在 Windows 上执行 `nvcc main.cu` 报错：
  ```
  nvcc fatal : Cannot find compiler 'cl.exe' in PATH
  ```
- **原因**：`nvcc` 在 Windows 上只负责编译设备代码（`.cu` 中的 GPU 部分），主机代码（C++）需要交给 MSVC 的 `cl.exe`。若没有在“x64 Native Tools Command Prompt”中运行，或 `cl.exe` 不在 `PATH` 里，nvcc 就找不到它。
- **解决方案**：
  1. 优先使用 **“x64 Native Tools Command Prompt for VS 2022”** 启动终端，它会自动配置好 `cl.exe` 的环境变量。
  2. 或手动向 nvcc 指定编译器路径：
     ```bat
     nvcc -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<版本>\bin\Hostx64\x64" main.cu -o main.exe
     ```
  3. 推荐直接用 **CMake** 构建（本项目做法），CMake 的 `LANGUAGES CUDA` 会自动定位 MSVC。
- **参考代码/文件**：`f:\code\cuda_learning\day01_hello_cuda\build.bat`、`f:\code\cuda_learning\day01_hello_cuda\CMakeLists.txt`（`project(... LANGUAGES CXX CUDA)`）。

### 1.2 中文注释导致 C4819 警告 / 编译失败

- **问题**：源码含中文注释时，MSVC 报大量 `warning C4819: 该文件包含不能在当前代码页(936)中表示的字符`，严重时中文乱码甚至报错。
- **原因**：`.cu` / `.cpp` 文件是 UTF-8 编码，但 MSVC 默认按系统本地代码页（中文 Windows 为 GBK/936）解释源文件，导致 UTF-8 多字节字符无法识别。
- **解决方案**：
  1. 给 MSVC 传 `/utf-8`，让它按 UTF-8 读取源码。对 CUDA 需要通过 `-Xcompiler` 转发：
     ```cmake
     if(MSVC)
         add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
         add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
     endif()
     ```
  2. 命令行手动编译时：`nvcc -Xcompiler "/utf-8" main.cu`。
  3. 确保源文件保存为 **UTF-8（推荐带 BOM）**，本项目 `.cu` 文件即以 BOM 开头。
- **参考代码/文件**：`f:\code\cuda_learning\day01_hello_cuda\CMakeLists.txt`（第 48-52 行 `/utf-8` 配置）。

### 1.3 CUDA 版本不兼容（sm_50 在 CUDA 13 被移除）

- **问题**：使用 CUDA 13 编译时指定 `-arch=sm_50` 或 `CMAKE_CUDA_ARCHITECTURES 50` 报错：
  ```
  nvcc fatal : Unsupported gpu architecture 'compute_50'
  ```
- **原因**：CUDA Toolkit 会随版本淘汰旧计算能力。**CUDA 13+ 移除了 sm_50 ~ sm_70**（Maxwell/Pascal/Volta），仅支持 sm_75 及以上；CUDA 12.x 才支持 sm_50 ~ sm_90。
- **解决方案**：根据 CUDA 版本动态选择架构列表，不要写死：
  ```cmake
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 100 120)
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
      set(CMAKE_CUDA_ARCHITECTURES 50 61 70 75 80 86 89 90)
  else()
      set(CMAKE_CUDA_ARCHITECTURES 50 52 61 70 75)
  endif()
  ```
  另外，`cudaDeviceProp` 的 `memoryClockRate` / `memoryBusWidth` 字段在 CUDA 13 中被移除，需改用 `cudaDeviceGetAttribute()` 查询。
- **参考代码/文件**：`f:\code\cuda_learning\day01_hello_cuda\CMakeLists.txt`（第 40-46 行）、`f:\code\cuda_learning\common\include\cuda_utils.h`（第 160-184 行显存带宽兼容写法）。

### 1.4 链接错误：找不到 cublas / cufft 等库

- **问题**：使用 cuBLAS / cuFFT 时，编译通过但链接报错：
  ```
  undefined reference to `cublasSgemm'   （Linux）
  LNK2019: 无法解析的外部符号 cublasCreate_v2   （Windows）
  ```
- **原因**：`cublas`、`cufft`、`cusparse` 等是**独立的库**，不随 `cudart` 自动链接，必须显式指定。
- **解决方案**：
  1. 命令行：`nvcc main.cu -lcublas -lcufft -o app`
  2. CMake 使用官方 imported target（推荐，自动处理头文件和库路径）：
     ```cmake
     find_package(CUDAToolkit REQUIRED)
     target_link_libraries(app PRIVATE CUDA::cublas CUDA::cufft CUDA::cusparse)
     ```
  3. 确认库版本与运行时 DLL/so 一致（避免 `cublas64_12.dll` 找不到）。
- **参考代码/文件**：`f:\code\cuda_learning\day13_cublas\CMakeLists.txt`、`f:\code\cuda_learning\day14_cufft\CMakeLists.txt`、`f:\code\cuda_learning\day32_cusparse\CMakeLists.txt`。

### 1.5 CMake 找不到 CUDA

- **问题**：`cmake ..` 报错：
  ```
  No CUDA toolset found.
  Failed to find nvcc. / Could NOT find CUDAToolkit
  ```
- **原因**：CUDA 未安装、`nvcc` 不在 `PATH`，或 Visual Studio 缺少 CUDA 集成的 `.props` 文件；也可能是 CMake 版本过低。
- **解决方案**：
  1. 确认 `nvcc --version` 可执行，`CUDA_PATH` 环境变量已设置。
  2. CMake ≥ 3.18，使用 `project(xxx LANGUAGES CXX CUDA)` 而不是老的 `find_package(CUDA)`。
  3. Windows 上若报 “No CUDA toolset found”，需把 CUDA 安装包里的 VS 集成文件拷到对应 VS 的 `BuildCustomizations` 目录，或用 `-T cuda=<版本>` 指定 toolset。
  4. 指定生成器与架构：`cmake -G "Visual Studio 17 2022" -A x64 ..`。
- **参考代码/文件**：`f:\code\cuda_learning\CMakeLists.txt`（顶层）、各 day 的 `build.bat` / `build.sh`。

---

## 二、运行时问题

### 2.1 忘记 cudaDeviceSynchronize() 导致结果不正确

- **问题**：Kernel 启动后立刻 `cudaMemcpy` 拷回结果或直接打印，得到的却是旧值 / 空值 / 部分结果。
- **原因**：Kernel 启动是**异步**的，主机代码不会等待 GPU 执行完成就继续往下走。若在 Kernel 尚未跑完时读取结果，就会拿到错误数据。（注意：普通阻塞式 `cudaMemcpy` 本身会隐式同步，但异步流、`cudaMemcpyAsync`、纯计时场景不会。）
- **解决方案**：
  1. 需要读结果或计时前，调用 `cudaDeviceSynchronize()` 等待 GPU 完成。
  2. Numba 中对应 `cuda.synchronize()`；使用流时用 `stream.synchronize()`。
  3. 计时务必包住同步，否则测到的只是“启动耗时”而非“执行耗时”。
- **参考代码/文件**：`f:\code\cuda_learning\day10_streams\python\streams.py`（`cuda.synchronize()` / `s.synchronize()`）、`f:\code\cuda_learning\tests\test_day10_streams.py`。

### 2.2 cudaMalloc 失败（显存不足）

- **问题**：分配大数组时 `cudaMalloc` 返回 `cudaErrorMemoryAllocation`，或程序崩溃 / 结果全 0。
- **原因**：请求的显存超过剩余可用量（显存被其他进程占用、之前分配未释放、数据规模过大）。忽略返回值时错误会被“吞掉”。
- **解决方案**：
  1. 始终检查返回值，用本项目的 `CUDA_SAFE_CALL` 宏包裹每次分配。
  2. 用 `cudaMemGetInfo(&free, &total)` 查询可用显存，`nvidia-smi` 查看占用。
  3. 分块（tiling / streaming）处理超大数据；及时 `cudaFree` 或使用内存池复用。
- **参考代码/文件**：`f:\code\cuda_learning\common\include\cuda_utils.h`（`CUDA_SAFE_CALL`）、`f:\code\cuda_learning\day34_memory_pool\`（内存池复用示例）。

### 2.3 Kernel 启动配置错误（Grid/Block 太大）

- **问题**：Kernel 启动即报错 `invalid configuration argument`，或线程越界导致结果错误。
- **原因**：Block 线程数超过硬件上限（**每块最多 1024 线程**，`maxThreadsPerBlock`），或 Grid 维度超上限；也可能是 Grid 数量算错导致漏算/越界。
- **解决方案**：
  1. Block 线程数不超过 1024，通常取 128 / 256；且最好是 32（warp）的整数倍。
  2. Grid 数量用向上取整公式覆盖所有元素：
     ```cpp
     int grid = (n + block - 1) / block;   // 向上取整
     ```
  3. Kernel 内务必做边界判断 `if (idx < n) { ... }`。
  4. 用 `print_device_info()` 查询本机 `maxThreadsPerBlock` / `maxGridSize`。
- **参考代码/文件**：`f:\code\cuda_learning\day05_device_query\src\main.cu`、`f:\code\cuda_learning\common\include\cuda_utils.h`（`print_device_info`）、`f:\code\cuda_learning\day02_vector_add\python\vector_add.py`。

### 2.4 未检查 cudaError 返回值

- **问题**：程序“正常”结束但结果错误，或在无关的后续调用处才崩溃，难以定位真正出错点。
- **原因**：CUDA 的 API 错误和 Kernel 错误是**粘性 / 异步**的——不检查返回值就无法及时发现，且 Kernel 内部错误只能在下一次同步或 `cudaGetLastError()` 时暴露。
- **解决方案**：
  1. 每个 CUDA API 调用都检查返回值（`CUDA_SAFE_CALL`）。
  2. 每次 Kernel 启动后检查：
     ```cpp
     kernel<<<grid, block>>>(...);
     CHECK_KERNEL_ERROR();   // 内部: cudaGetLastError + cudaDeviceSynchronize
     ```
  3. Python/Numba 下 Kernel 异常会直接抛出，注意不要 `try/except` 静默吞掉。
- **参考代码/文件**：`f:\code\cuda_learning\common\include\cuda_utils.h`（`check_cuda_error` / `check_kernel_error` / `CHECK_KERNEL_ERROR`）。

### 2.5 页锁定内存（cudaMallocHost）使用不当

- **问题**：`cudaMemcpyAsync` 没有真正异步、流水线没有重叠效果；或大量使用页锁定内存后系统变卡、分配失败。
- **原因**：`cudaMemcpyAsync` 只有在**页锁定（pinned）内存**上才是真正异步的；普通可分页内存会退化为同步拷贝。而 pinned 内存不可被换出，占用物理内存，过量分配会拖慢整个系统。
- **解决方案**：
  1. 需要传输/计算重叠时，主机缓冲用 `cudaMallocHost` / `cudaHostAlloc` 分配（Numba 用 `cuda.pinned_array`）。
  2. 用完及时 `cudaFreeHost` 释放，控制 pinned 内存总量。
  3. 结合多流 + 异步拷贝实现流水线并行。
- **参考代码/文件**：`f:\code\cuda_learning\day10_streams\python\streams.py`、`f:\code\cuda_learning\day10_streams\src\main.cu`。

---

## 三、性能问题

### 3.1 非合并访问导致带宽利用率低

- **问题**：Kernel 逻辑正确但比预期慢很多，`nsight` 显示全局内存带宽利用率很低。
- **原因**：同一个 warp（32 线程）访问全局内存时，若地址不连续（跨步、列优先），无法合并为少数几次事务，导致大量带宽被浪费。行优先 vs 列优先访问的性能可相差数倍到数十倍。
- **解决方案**：
  1. 让相邻线程访问相邻地址（`data[row * cols + col]`，相邻线程处理相邻 `col`）。
  2. 二维数据尽量按行优先布局并让线程沿行方向展开。
  3. 需要转置类访问时用共享内存中转（tiled transpose）。
- **参考代码/文件**：`f:\code\cuda_learning\day07_coalesced_access\python\coalesced_access.py`（`sum_rows_kernel` 合并 vs `sum_cols_kernel` 非合并）、`f:\code\cuda_learning\tests\test_day07_coalesced.py`、`f:\code\cuda_learning\day04_matrix_transpose\`。

### 3.2 过多的 __syncthreads() 导致性能下降

- **问题**：共享内存归约 / tiled matmul 中同步过多，Kernel 变慢；甚至在分支内调用 `__syncthreads()` 导致**死锁**。
- **原因**：`__syncthreads()` 是块内屏障，所有线程都要到达才放行。过度使用增加等待开销；而放在 `if` 分支内（部分线程不执行）会让屏障永远等不齐，直接挂死。
- **解决方案**：
  1. 只在真正需要保证共享内存可见性的地方同步，去掉冗余屏障。
  2. **绝不**在会导致线程发散的分支内调用 `__syncthreads()`；需要时把同步放到分支外。
  3. Warp 内归约（32 线程以内）可用 `__shfl_down_sync` 免去同步。
- **参考代码/文件**：`f:\code\cuda_learning\day09_reduction\src\main.cu`（`reduce_shared_kernel` 与 `reduce_warp_shuffle_kernel` 对比）、`f:\code\cuda_learning\day06_tiled_matmul\`。

### 3.3 动态并行性能代价

- **问题**：在 Kernel 内部用 `<<<>>>` 再启动子 Kernel（动态并行），性能远低于预期。
- **原因**：设备端启动 Kernel 有不小的固定开销（调度、内存、隐式同步）；如果递归层级深、每次启动的工作量又小，启动开销会淹没计算收益。
- **解决方案**：
  1. 只在工作量足够大、且难以在主机侧预先划分时才用动态并行。
  2. 尽量增大每次子 Kernel 的粒度，减少启动次数；控制递归深度。
  3. 很多场景可用一次性扁平化的 Grid 或 grid-stride loop 替代动态并行。
- **参考代码/文件**：`f:\code\cuda_learning\day12_dynamic_parallel\src\main.cu`、`f:\code\cuda_learning\day12_dynamic_parallel\python\dynamic_parallel.py`。

### 3.4 频繁的 cudaMalloc / cudaFree

- **问题**：在循环 / 每帧 / 每个 batch 里反复 `cudaMalloc` + `cudaFree`，整体吞吐上不去。
- **原因**：`cudaMalloc` / `cudaFree` 是**重量级、会隐式同步**的操作，频繁调用会打断流水线、造成显著停顿。
- **解决方案**：
  1. 预先一次性分配好缓冲区，在循环中复用，循环外再释放。
  2. 使用内存池（自定义或 `cudaMallocAsync` + memory pool）管理临时显存。
  3. Numba 中避免在热路径反复 `cuda.device_array`，改为预分配。
- **参考代码/文件**：`f:\code\cuda_learning\day34_memory_pool\python\memory_pool.py`、`f:\code\cuda_learning\day34_memory_pool\src\main.cu`。

### 3.5 过多的主机-设备数据传输

- **问题**：GPU 计算本身很快，但整体耗时高，`nsight` 显示大量时间花在 `Memcpy HtoD` / `DtoH` 上。
- **原因**：PCIe 传输带宽远低于显存带宽，频繁在 CPU/GPU 之间来回搬数据（尤其是小块、每步都拷回）会成为瓶颈。
- **解决方案**：
  1. 数据尽量“留在 GPU 上”，多个 Kernel 串起来处理，减少中间结果回传。
  2. 合并小传输为大传输；用页锁定内存 + 异步拷贝 + 多流实现传输与计算重叠。
  3. 只在最终需要时把结果拷回主机。
- **参考代码/文件**：`f:\code\cuda_learning\day10_streams\python\streams.py`（多流重叠）、`f:\code\cuda_learning\day20_performance_tuning\`。

---

## 四、调试技巧

### 4.1 cuda-gdb / Nsight 基本用法

- **问题**：Kernel 结果不对，但不知道如何单步调试 GPU 代码、查看某个线程的变量。
- **原因**：GPU 是大规模并行执行，普通 CPU 调试器无法直接观察设备端线程状态，需要专用工具。
- **解决方案**：
  - **Linux（cuda-gdb）**：编译加 `-g -G` 生成设备调试信息：
    ```bash
    nvcc -g -G main.cu -o app
    cuda-gdb ./app
    (cuda-gdb) break my_kernel
    (cuda-gdb) run
    (cuda-gdb) cuda thread (0,0,0) block (0,0,0)   # 切换到指定线程
    (cuda-gdb) print my_var
    ```
  - **Windows（Nsight）**：用 **Nsight Visual Studio Edition** 做源码级调试；用 **Nsight Systems**（时间线/流并发）和 **Nsight Compute**（单 Kernel 指标：带宽、占用率、访存效率）做性能分析。
  - 注意 `-G` 会关闭优化，仅用于调试，性能测量时务必去掉。
- **参考代码/文件**：`f:\code\cuda_learning\day35_nsight_profiling\`、`f:\code\cuda_learning\.vscode\launch.json`。

### 4.2 使用 cuda-memcheck / compute-sanitizer 检测越界访问

- **问题**：程序偶发崩溃、结果随机错误，报 `unspecified launch failure` 或 `illegal memory access`。
- **原因**：Kernel 内数组越界、未初始化指针、共享内存越界、竞态条件等。这类问题往往不稳定复现。
- **解决方案**：
  1. 用越界检测工具直接定位到出错的 Kernel 与代码行：
     ```bash
     compute-sanitizer --tool memcheck ./app     # 新版（CUDA 11.6+）
     cuda-memcheck ./app                          # 旧版
     ```
  2. `--tool racecheck` 检测共享内存竞态，`--tool initcheck` 检测未初始化访问，`--tool synccheck` 检测同步误用。
  3. 配合 `-lineinfo` 编译可让报告显示源码行号。
- **参考代码/文件**：`f:\code\cuda_learning\day35_nsight_profiling\README.md`。

### 4.3 printf 在 Kernel 中的使用限制

- **问题**：在 Kernel 里 `printf` 没有输出，或输出顺序混乱、内容丢失。
- **原因**：
  - 设备端 `printf` 通过一个**固定大小的环形缓冲区**输出，超出会被丢弃；
  - 输出**必须在同步后**（`cudaDeviceSynchronize`）才会刷新到主机；
  - 成千上万个线程同时打印会刷屏且乱序。
- **解决方案**：
  1. 只让少数线程打印，如 `if (threadIdx.x == 0 && blockIdx.x == 0) printf(...)`。
  2. `printf` 后调用 `cudaDeviceSynchronize()` 确保缓冲区刷新。
  3. 需要更多输出时增大缓冲：`cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size)`。
  4. Numba Kernel 内可用 `from numba import cuda` 的 `cuda.printf` 风格（`print()`）但同样受限，调试大数据优先用工具而非 printf。
- **参考代码/文件**：`f:\code\cuda_learning\day01_hello_cuda\src\main.cu`（Kernel 内 printf 示例）。

### 4.4 如何检查 Kernel 错误

- **问题**：Kernel 启动那行没报错，问题却在后面别处冒出来。
- **原因**：Kernel 启动是异步的，启动配置错误由 `cudaGetLastError()` 捕获，执行期错误（越界等）要到同步时才由 `cudaDeviceSynchronize()` 返回。只看启动语句本身看不出问题。
- **解决方案**：Kernel 后紧跟两步检查——
  ```cpp
  my_kernel<<<grid, block>>>(...);
  cudaError_t err = cudaGetLastError();          // 捕获启动配置错误
  if (err != cudaSuccess) { /* 处理 */ }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());       // 捕获执行期错误
  ```
  本项目已封装为 `CHECK_KERNEL_ERROR()` 宏，一行搞定。
- **参考代码/文件**：`f:\code\cuda_learning\common\include\cuda_utils.h`（`check_kernel_error` / `CHECK_KERNEL_ERROR`）、`f:\code\cuda_learning\day09_reduction\src\main.cu`（每次 Kernel 后 `CHECK_KERNEL_ERROR()`）。

---

## 五、PyTorch CUDA 扩展

### 5.1 setup.py 编译常见问题

- **问题**：`python setup.py install` / `pip install -e .` 编译 CUDA 扩展失败，常见报错：
  ```
  RuntimeError: The detected CUDA version (13.x) mismatches the version that was used to compile PyTorch (12.x)
  error: Microsoft Visual C++ 14.0 or greater is required   （Windows）
  ```
- **原因**：
  - 系统 CUDA Toolkit 版本与安装的 PyTorch 编译时所用 CUDA 版本不一致；
  - Windows 缺少匹配的 MSVC 编译器或未在 VS 环境中执行；
  - `TORCH_CUDA_ARCH_LIST` 未设置导致为所有架构编译（慢）或架构不匹配。
- **解决方案**：
  1. 保证 `nvcc --version` 与 `torch.version.cuda` 主版本一致。
  2. 使用 `torch.utils.cpp_extension` 的 `CUDAExtension` + `BuildExtension`：
     ```python
     from torch.utils.cpp_extension import CUDAExtension, BuildExtension
     setup(
         ext_modules=[CUDAExtension("my_ext", ["main.cu"])],
         cmdclass={"build_ext": BuildExtension},
     )
     ```
  3. 设置目标架构：`set TORCH_CUDA_ARCH_LIST=8.6`（Windows）/ `export ...`（Linux）。
  4. Windows 在 “x64 Native Tools Command Prompt” 中构建。
- **参考代码/文件**：`f:\code\cuda_learning\day21_pytorch_extension\src\main.cu`、`f:\code\cuda_learning\day21_pytorch_extension\python\pytorch_extension.py`、`f:\code\cuda_learning\day36_pybind11_cuda\CMakeLists.txt`。

### 5.2 反向传播梯度形状不匹配

- **问题**：自定义 `autograd.Function` 训练时报错：
  ```
  RuntimeError: Function returned an incorrect number of gradients / grad shape mismatch
  ```
- **原因**：`backward()` 返回的梯度个数 / 形状必须与 `forward()` 的输入**一一对应**。非 Tensor 输入（如标量参数）也要返回对应位置的 `None`；且返回梯度形状要与对应输入完全一致。
- **解决方案**：
  1. `backward` 的返回值个数 = `forward` 的参数个数（`ctx` 不算）。
  2. 不需要梯度的输入返回 `None` 占位。
  3. 用 `ctx.save_for_backward()` 保存前向张量，反向据其原始形状构造梯度：
     ```python
     @staticmethod
     def backward(ctx, grad_output):
         (x,) = ctx.saved_tensors
         grad_x = grad_output * ...     # 形状须与 x 一致
         return grad_x, None            # 第二个输入无梯度
     ```
  4. 用 `torch.autograd.gradcheck` 做数值梯度校验。
- **参考代码/文件**：`f:\code\cuda_learning\day21_pytorch_extension\python\pytorch_extension.py`。

### 5.3 Tensor 指针获取

- **问题**：在 C++/CUDA 扩展里拿到的裸指针访问出错（越界、读到垃圾值、`illegal memory access`）。
- **原因**：
  - 直接用 `tensor.data_ptr()` 却忽略了张量**非连续**（transpose/slice 后 stride 不规则）；
  - 张量在 CPU 上却当作设备指针使用；
  - dtype 与模板类型不符。
- **解决方案**：
  1. 传入前做检查与规整：
     ```cpp
     TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
     TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
     ```
     Python 侧先 `x = x.contiguous()`。
  2. 用正确 dtype 获取指针：`x.data_ptr<float>()`。
  3. 优先用 `packed_accessor32<scalar_t, N>()` 或 `AT_DISPATCH_FLOATING_TYPES` 宏，让访问带上 shape/stride 更安全。
- **参考代码/文件**：`f:\code\cuda_learning\day21_pytorch_extension\src\main.cu`（`data_ptr` / 连续性检查）。

---

## 附：快速自查清单（Checklist）

- [ ] 在 x64 Native Tools 命令行或 CMake 环境中编译（MSVC / nvcc 可见）
- [ ] 源码 UTF-8 + `/utf-8`，无 C4819 警告
- [ ] `CMAKE_CUDA_ARCHITECTURES` 按 CUDA 版本动态设置
- [ ] cuBLAS/cuFFT/cuSPARSE 等库已显式链接
- [ ] 每个 CUDA API 用 `CUDA_SAFE_CALL` 包裹
- [ ] 每次 Kernel 启动后 `CHECK_KERNEL_ERROR()`
- [ ] 读结果 / 计时前已 `cudaDeviceSynchronize()`
- [ ] Block ≤ 1024 线程且为 32 的倍数，Grid 用向上取整，Kernel 内做边界判断
- [ ] 访存合并（相邻线程访问相邻地址）
- [ ] 热路径避免频繁 malloc/free 与不必要的 HtoD/DtoH 传输
- [ ] 出问题先跑 `compute-sanitizer` / `cuda-memcheck`
- [ ] PyTorch 扩展：CUDA 版本匹配、张量 `.contiguous()`、backward 梯度形状对齐

> 相关运行/测试参考：`f:\code\cuda_learning\tests\`（pytest 用例，`require_cuda()` 会在无 GPU 时自动跳过）。
