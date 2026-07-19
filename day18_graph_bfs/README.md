# Day 18: GPU 并行 BFS 图遍历

## 学习目标

1. **实现** 基于 CSR（压缩稀疏行）格式的图存储，理解 `offsets` / `edges` 两个数组如何表示邻接关系
2. **实现** GPU 的前沿扩展（Frontier Expansion）BFS：每一层并行展开当前前沿的所有节点
3. **解释** `atomicCAS` 如何保证每个节点只被访问（标记）一次，避免并发重复入队
4. **测量** GPU BFS 遍历的层数与耗时，理解图算法的"逐层同步"执行模型
5. **对比** 串行 BFS 与前沿并行 BFS 的差异，认识图算法中的负载不均衡问题

---

## 前置依赖

- **Day 03/04（线程与索引）**：`blockIdx.x * blockDim.x + threadIdx.x` 的一维索引
- **Day 09/10（原子操作）**：`atomicCAS`、`atomicAdd` 的语义与用途
- **Day 06（内存拷贝）**：主机与设备之间 `cudaMemcpy` 的往返

---

## 核心概念

### 1. CSR（压缩稀疏行）图表示

图用两个数组紧凑存储，避免稠密邻接矩阵的 O(V²) 空间：

```
节点 i 的邻居 = csr_edges[ csr_offsets[i] .. csr_offsets[i+1] )

csr_offsets: [0, 3, 5, 8, ...]   # 长度 num_nodes + 1，累积边数
csr_edges:   [1,4,7, 2,9, 0,3,5, ...]  # 长度 num_edges，展平的邻居列表
```

本例在主机端生成随机图（[main.cu](file:///f:/code/cuda_learning/day18_graph_bfs/src/main.cu#L69-L91)）：`NUM_NODES = 10000`、`NUM_EDGES = 50000`，每个节点固定 `num_edges / num_nodes = 5` 条边。

### 2. 前沿扩展（Frontier-based）BFS

BFS 天然是"逐层"的：第 L 层的所有节点可以**同时**展开它们的邻居。GPU 把"当前前沿"数组交给一批线程并行处理：

```
current_level = 0
frontier = [source]          # 初始前沿只含源点
while frontier 非空:
    并行(每个前沿节点分配一个线程):
        for 邻居 in 该节点的邻居:
            if 邻居未访问:
                distances[邻居] = current_level + 1
                把邻居加入 next_frontier
    swap(frontier, next_frontier)   # 指针交换，进入下一层
    current_level++
```

对应的 Kernel 见 [bfs_kernel](file:///f:/code/cuda_learning/day18_graph_bfs/src/main.cu#L30-L55)，每个线程处理前沿数组中的一个节点。

### 3. 用 atomicCAS 保证"只访问一次"

多个前沿节点可能同时指向同一个未访问邻居，若不加保护会重复入队。原子比较交换解决竞态：

```cpp
if (atomicCAS(&visited[neighbor], 0, 1) == 0)   // 谁把 0 换成 1 谁赢
{
    distances[neighbor] = current_level + 1;
    int pos = atomicAdd(next_size, 1);          // 原子分配下一前沿的写入位置
    next_frontier[pos] = neighbor;
}
```

- `atomicCAS(addr, 0, 1)`：只有第一个把 `visited` 从 0 改成 1 的线程返回旧值 0，从而"抢到"该节点。
- `atomicAdd(next_size, 1)`：为 `next_frontier` 原子地分配一个唯一的写入下标，实现无锁并发追加。

### 4. 逐层主机循环与前沿交换

主循环在主机端（[main.cu](file:///f:/code/cuda_learning/day18_graph_bfs/src/main.cu#L119-L139)）：每层把 `next_size` 清零、启动一次 Kernel、`cudaDeviceSynchronize`、交换 `d_frontier` 与 `d_next_frontier` 指针，再把 `next_size` 拷回主机作为下一轮 `frontier_size`，直到前沿为空。

---

## 文件结构

```
day18_graph_bfs/
├── src/main.cu          # CUDA 实现（CSR 图 + bfs_kernel 前沿扩展）
├── python/graph_bfs.py  # Python/NumPy 版：串行 BFS vs Frontier 模拟并交叉验证
├── CMakeLists.txt       # 构建配置
├── build.bat            # Windows 构建脚本（CMake + VS 2022）
├── build.sh             # Linux/macOS 构建脚本
└── README.md            # 本文件
```

---

## 编译与运行

### Windows（PowerShell / CMD）

方式一：使用 build.bat（内部调用 CMake，产物在 `build\Release\`）

```powershell
cd day18_graph_bfs
.\build.bat
.\build\Release\graph_bfs.exe
```

方式二：手动 CMake

```powershell
cd day18_graph_bfs
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\graph_bfs.exe
```

### Linux / macOS

```bash
cd day18_graph_bfs
bash build.sh
./build/graph_bfs
```

### Python 版本

```bash
cd day18_graph_bfs/python
python graph_bfs.py
```

> Python 版用 NumPy 构建 CSR 图，分别运行串行 BFS 与 Frontier 模拟 BFS，并校验两者的距离数组完全一致（`diff_count == 0`）。

---

## 知识点总结

1. **CSR 格式**：`offsets[i]..offsets[i+1]` 界定节点 i 的邻居范围，空间随边数线性增长。
2. **前沿扩展**：每层并行处理前沿节点，层与层之间需要全局同步（本例通过主机端 Kernel 串行调用实现）。
3. **atomicCAS**：以无锁方式确保每个节点只被首次访问的线程占有，避免重复入队。
4. **atomicAdd 分配位置**：对 `next_size` 原子自增，得到并发安全的写入下标。
5. **指针交换**：用 `d_frontier` 与 `d_next_frontier` 双缓冲，避免每层重新分配显存。
6. **负载不均衡**：度数大的节点其线程要处理更多邻居，是图算法性能的主要瓶颈。

---

## 验收标准

- 编译成功并运行 `graph_bfs.exe`，看到 `图规模: 10000 节点, 50000 边` 与 `GPU BFS` 计时输出。
- 程序打印 **BFS 层数**（随机图通常为个位数层），且能正常退出（`程序正常退出。`）。
- Python 版输出 `结果一致性 : 通过`，说明 Frontier BFS 与串行 BFS 的距离数组完全相同。
- 能解释为何随机稠密图的 BFS 层数很小（少数几跳即可覆盖大部分可达节点）。

---

## 动手挑战

1. **统计每层前沿大小**：在主机循环里打印每一层的 `frontier_size`，观察前沿"先膨胀后收缩"的形状。
2. **改为可复现图**：在生成随机边前调用 `srand(42)`，并验证两次运行得到相同的层数（当前代码未固定随机种子）。
3. **优化负载均衡**：把"每个线程处理一个节点"改为"每个线程处理一条边"（edge-centric），比较高度数节点场景下的耗时差异。

---

## 常见问题

- **Q：为什么 `next_frontier` 用 `atomicAdd` 分配位置而不是按线程 id 写入？**
  A：因为不是每个前沿节点都会产生新节点（邻居可能已访问），产出是稀疏且数量未知的，必须用原子计数器动态压紧到连续区间。

- **Q：源点 `distances[0]=0`、`visited[0]=1` 为何在初始化时手动设置？**
  A：源点不经过 Kernel 就已"访问"，作为第 0 层起点；否则第一次展开时它会被邻居再次错误标记。

- **Q：这个实现是最优的 GPU BFS 吗？**
  A：不是，它是教学版。工业级实现（如 Gunrock）会做负载均衡、方向优化（direction-optimizing BFS）等，参见延伸阅读。

---

## 延伸阅读

- [Gunrock: A High-Performance Graph Processing Library on the GPU](https://gunrock.github.io/docs/) — UC Davis 的 GPU 图计算库
- [Merrill, Garland, Grimshaw — Scalable GPU Graph Traversal (PPoPP 2012)](https://research.nvidia.com/publication/scalable-gpu-graph-traversal) — 经典的 GPU BFS 负载均衡论文
