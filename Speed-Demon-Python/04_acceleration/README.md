# Numba JIT 加速教程

## 目录说明

| 文件 | 说明 |
|------|------|
| `01_numba_jit.md` | 完整的Numba JIT教程文档，包含原理、使用方法和最佳实践 |
| `numba_benchmark.py` | 性能基准测试脚本，包含4个测试场景 |

## 环境准备

### 安装依赖

```bash
pip install numba numpy
```

或者使用项目根目录的requirements.txt：

```bash
cd f:\code\cuda_learning
pip install -r requirements.txt
```

## 运行基准测试

```bash
cd f:\code\cuda_learning\04_acceleration
python numba_benchmark.py
```

## 测试场景说明

### 1. 蒙特卡洛 π 计算
- 迭代次数: 1,000,000 次
- 对比: Python纯循环 vs @jit vs @njit vs @njit+parallel vs NumPy

### 2. 矩阵乘法
- 矩阵大小: 100x100
- 对比: Python三重循环 vs @njit vs @njit+parallel vs NumPy (BLAS)

### 3. 数组求和
- 数组大小: 10,000,000 元素
- 对比: Python循环 vs @njit vs @njit+parallel vs np.sum

### 4. @vectorize 向量化
- 数组大小: 1,000,000 元素
- 对比: cpu目标 vs parallel目标 vs NumPy原生

## 预期性能对比

| 实现方式 | 加速比 (相对于Python) |
|---------|----------------------|
| @jit | ~10x |
| @njit | ~50-100x |
| @njit + parallel | ~100-500x |
| NumPy向量化 | ~20-50x |

**注意**: 实际加速比取决于硬件、问题大小和算法特性。

## 关键要点

1. **始终使用 @njit 而非 @jit**: 确保 nopython 模式，获得最佳性能
2. **预热编译**: JIT首次调用开销大，生产环境请预热
3. **并行只适合大计算量**: 小循环使用并行反而会更慢
4. **避免全局变量**: 全局变量会被编译时常量化
5. **所有内部函数也需要 Numba 化**: 否则会有调用开销

## 常见问题

### Q: 为什么我的 Numba 代码没有变快?
A: 检查是否降级到 object 模式，使用 `njit` 而非 `jit`，确保没有 Python 对象操作。

### Q: 为什么并行版本反而更慢?
A: 并行有线程启动开销，只适合计算量较大的循环。

### Q: 如何查看 Numba 编译的中间结果?
A: 使用 `func.inspect_types()` 查看类型推断，`func.inspect_llvm()` 查看 LLVM IR。
