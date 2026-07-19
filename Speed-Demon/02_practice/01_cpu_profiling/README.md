# CPU Profiling 实战

## 实验目标

1. 学习如何使用 pprof 进行 CPU 性能分析
2. 理解 CPU 采样的原理
3. 掌握火焰图的阅读方法
4. 学习优化 CPU 密集型代码

## 运行方式

### 1. Server 模式（推荐）

```bash
go run main.go
```

然后在另一个终端运行：

```bash
# 采集 10 秒 CPU profile
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=10
```

### 2. CLI 模式

```bash
# 直接运行并生成 profile 文件
go run main.go -mode cli -duration 30 -cpuprofile cpu.pprof

# 分析 profile
go tool pprof cpu.pprof
```

### 3. Benchmark 模式

```bash
go run main.go -mode bench
```

## pprof 常用命令

```bash
# 进入交互模式后
top          # 查看热点函数
top -cum     # 按累计时间排序
list fibonacci  # 查看函数源码级耗时
web          # 生成火焰图（需要 graphviz）
svg          # 生成 SVG 图
pdf          # 生成 PDF 报告
```

## 实验步骤

### 步骤 1: 发现热点

1. 启动服务: `go run main.go`
2. 用 curl 制造负载:
```bash
# 在另一个终端
while true; do curl http://localhost:6060/heavy; done
```
3. 采集 profile:
```bash
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=10
```
4. 在 pprof 中输入 `top`，观察哪个函数占 CPU 最多

**预期结果**: `heavyComputation` 应该是最大的热点

### 步骤 2: 源码级分析

```bash
(pprof) list heavyComputation
```

观察哪一行代码耗时最多。

**思考**:
- 为什么 `math.Sin` 和 `math.Cos` 调用开销大？
- 能否预先计算某些值？

### 步骤 3: 火焰图可视化

```bash
(pprof) web
```

或生成 SVG:
```bash
(pprof) svg > cpu.svg
```

**火焰图阅读指南**:
- X 轴: CPU 时间占比
- Y 轴: 调用栈深度
- 每个矩形: 一个函数
- 宽度越宽，占用 CPU 越多

### 步骤 4: 不同函数对比

分别访问不同的 endpoint，观察 profile 变化：

```bash
curl http://localhost:6060/fib
curl http://localhost:6060/prime
curl http://localhost:6060/matrix
```

**思考**:
- Fibonacci 的调用图是什么形状？
- 矩阵乘法的热点在哪里？

## 代码优化练习

### 练习 1: 优化 Fibonacci

**当前实现**:
```go
func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}
```

**问题**: 指数级时间复杂度 O(2^n)

**任务**: 实现一个 O(n) 或 O(log n) 的版本，并对比性能。

**优化方向**:
- 迭代法
- 矩阵快速幂
- 动态规划

### 练习 2: 优化素数查找

**当前实现**:
```go
func isPrime(n int) bool {
    for i := 2; i*i <= n; i++ {
        if n%i == 0 {
            return false
        }
    }
    return true
}
```

**优化方向**:
- 跳过偶数
- 使用埃拉托斯特尼筛法 (Sieve of Eratosthenes)
- 6k ± 1 优化

### 练习 3: 优化矩阵乘法

**当前实现**: O(n³) 三重循环

**优化方向**:
- 循环顺序优化（缓存局部性）
- 分块算法 (Blocking)
- Strassen 算法（O(n^2.8)）

## 进阶实验

### 1. 采样频率对结果的影响

修改采样频率（需要修改源码或使用不同采样时长）:

```bash
# 采样 1 秒
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=1

# 采样 30 秒
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30
```

**思考**:
- 采样时间短会有什么偏差？
- 多长时间的采样比较可靠？

### 2. 并发对 Profile 的影响

同时运行多个请求:

```bash
# 使用 ab 或 hey 进行并发测试
hey -n 1000 -c 10 http://localhost:6060/heavy
```

**思考**:
- 并发情况下，Profile 会有什么变化？
- runtime.schedule 会出现在热点中吗？

## 常见问题

### Q: 为什么 web 命令打不开？

A: 需要安装 graphviz:
- macOS: `brew install graphviz`
- Ubuntu: `sudo apt install graphviz`
- Windows: 下载并安装 [Graphviz](https://graphviz.org/download/)

### Q: profile 中看不到我的函数？

A: 可能原因:
1. 采样时间太短，函数没被采样到
2. 函数被内联了，看不到独立的条目
3. 函数执行时间确实很短

### Q: 优化后怎么验证效果？

A: 使用基准测试:
```bash
# 编写 _test.go 文件
go test -bench=. -cpuprofile=cpu.pprof

# 对比优化前后
benchstat old.pprof new.pprof
```

## 延伸阅读

- [Profiling Go Programs](https://blog.golang.org/profiling-go-programs)
- [pprof 官方文档](https://github.com/google/pprof/blob/master/doc/README.md)
- [火焰图指南](http://www.brendangregg.com/flamegraphs.html)
