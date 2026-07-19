# 基准测试与 benchstat 对比

## 概述

基准测试（Benchmark）是性能优化的基础。本章节展示如何：

1. 编写有效的 Go benchmark
2. 生成和比较 profile
3. 使用 benchstat 做统计对比
4. 理解 benchmark 的各种陷阱

## 快速开始

### 运行所有 benchmark

```bash
go test -bench=. -benchmem
```

### 运行指定 benchmark

```bash
# 只运行字符串相关的
go test -bench="Concat" -benchmem

# 运行并发相关的
go test -bench="Mutex|Atomic" -benchmem
```

### 带 profile 的 benchmark

```bash
# CPU profile
go test -bench=. -cpuprofile=cpu.pprof

# Memory profile
go test -bench=. -memprofile=mem.pprof

# All together
go test -bench=. -benchmem -cpuprofile=cpu.pprof -memprofile=mem.pprof -trace=trace.out
```

## 使用 benchstat 对比

### 安装 benchstat

```bash
go install golang.org/x/perf/cmd/benchstat@latest
```

### 对比两次运行

```bash
# 第一次运行（优化前）
go test -bench=. -benchmem -count=5 > old.txt

# 第二次运行（优化后）
go test -bench=. -benchmem -count=5 > new.txt

# 对比
benchstat old.txt new.txt
```

### 示例输出

```
name                    old time/op    new time/op    delta
NaiveConcat-8            95.4µs ± 2%     2.1µs ± 1%   -97.80%  (p=0.008 n=5+5)
BuilderConcat-8          2.05µs ± 1%    2.01µs ± 1%    -1.95%  (p=0.008 n=5+5)

name                    old alloc/op   new alloc/op   delta
NaiveConcat-8            3.22MB ± 0%    0.01MB ± 0%   -99.69%  (p=0.008 n=5+5)
BuilderConcat-8          6.14kB ± 0%    6.14kB ± 0%      ~     (all equal)

name                    old allocs/op  new allocs/op  delta
NaiveConcat-8             1.00k ± 0%     0.00k ± 0%  -100.00%  (p=0.008 n=5+5)
BuilderConcat-8           1.00 ± 0%      1.00 ± 0%      ~     (all equal)
```

## Benchmark 最佳实践

### 1. 使用 b.ResetTimer()

如果 benchmark 有昂贵的初始化步骤：

```go
func BenchmarkSomething(b *testing.B) {
    // 昂贵的初始化
    data := setupExpensiveData()
    
    b.ResetTimer()  // 重置计时器，不计入初始化时间
    
    for i := 0; i < b.N; i++ {
        process(data)
    }
}
```

### 2. 使用 b.StopTimer() 和 b.StartTimer()

如果每次迭代都需要清理：

```go
func BenchmarkSomething(b *testing.B) {
    for i := 0; i < b.N; i++ {
        b.StopTimer()
        cleanup()  // 不计入 benchmark 时间
        b.StartTimer()
        
        process()  // 只有这部分被计时
    }
}
```

### 3. 并行 benchmark

使用 `RunParallel` 测试并发性能：

```go
func BenchmarkConcurrent(b *testing.B) {
    var mu sync.Mutex
    var counter int64
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            mu.Lock()
            counter++
            mu.Unlock()
        }
    })
}
```

**注意**: `RunParallel` 默认使用 `GOMAXPROCS` 个 goroutine

### 4. 子 benchmark

使用 `b.Run` 组织相关的 benchmark：

```go
func BenchmarkMap(b *testing.B) {
    b.Run("Small", func(b *testing.B) {
        m := make(map[int]int, 100)
        benchmarkMap(b, m)
    })
    
    b.Run("Large", func(b *testing.B) {
        m := make(map[int]int, 10000)
        benchmarkMap(b, m)
    })
}
```

运行指定的子 benchmark：
```bash
go test -bench="Map/Small"
```

### 5. 控制编译器优化

防止编译器把 benchmark 代码优化掉：

```go
var result int

func BenchmarkFib(b *testing.B) {
    var r int
    for i := 0; i < b.N; i++ {
        r = fib(40)
    }
    result = r  // 赋值给包级变量，防止被优化
}
```

更好的方式：使用 `testing.B.ReportAllocs()`

## 常见性能对比案例

### 案例 1: 字符串拼接

```bash
go test -bench="Concat" -benchmem
```

**预期结果**:
- `BuilderConcat` 比 `NaiveConcat` 快 ~50 倍
- 内存分配少 1000 倍

### 案例 2: Mutex vs Atomic

```bash
go test -bench="Mutex|Atomic" -benchmem
```

**预期结果**:
- Atomic 比 Mutex 快 ~5-10 倍
- 两者都没有内存分配

### 案例 3: Channel 有无缓冲

```bash
go test -bench="Channel" -benchmem
```

**预期结果**:
- 有缓冲的 channel 吞吐量高 ~2-5 倍
- 延迟也更低

### 案例 4: 预分配切片

```bash
go test -bench="SliceAppend" -benchmem
```

**预期结果**:
- 预分配版本快 ~3 倍
- 内存分配少很多

### 案例 5: sync.Pool

```bash
go test -bench="Pool" -benchmem
```

**预期结果**:
- 使用 Pool 可以显著减少内存分配
- 但获取/放回有一定开销

## 深入分析 benchmark

### 结合 pprof

```bash
# 生成 CPU profile
go test -bench="NaiveConcat" -benchmem -cpuprofile=cpu.pprof

# 分析热点
go tool pprof cpu.pprof
(pprof) top
(pprof) list BenchmarkNaiveConcat
```

### 结合 trace

```bash
go test -bench="Mutex|Atomic" -trace=trace.out
go tool trace trace.out
```

在 trace 中可以看到：
- 锁竞争的具体情况
- Goroutine 调度情况
- 上下文切换频率

## Benchmark 陷阱与注意事项

### ❌ 陷阱 1: 编译器优化掉代码

```go
// BAD: 结果未被使用，可能被优化
func BenchmarkBad(b *testing.B) {
    for i := 0; i < b.N; i++ {
        compute()  // 返回值未被使用
    }
}

// GOOD: 使用一个全局变量保存结果
var sink int

func BenchmarkGood(b *testing.B) {
    var total int
    for i := 0; i < b.N; i++ {
        total += compute()
    }
    sink = total
}
```

### ❌ 陷阱 2: 每次迭代的工作量不一致

```go
// BAD: N 在循环中会变化！
func BenchmarkBad(b *testing.B) {
    for i := 0; i < b.N; i++ {
        for j := 0; j < b.N; j++ {  // ❌ 不要这样！
            compute()
        }
    }
}
```

### ❌ 陷阱 3: 内存分配被编译器优化

```go
// BAD: make 可能被优化掉，因为没有逃逸
func BenchmarkBad(b *testing.B) {
    for i := 0; i < b.N; i++ {
        s := make([]int, 100)  // 只在栈上分配
        _ = s
    }
}
```

### ❌ 陷阱 4: 基准测试本身的开销

```go
// BAD: benchmark 中的操作影响结果
func BenchmarkBad(b *testing.B) {
    for i := 0; i < b.N; i++ {
        b.StopTimer()  // 频繁的 Stop/Start 有开销
        // ...
        b.StartTimer()
    }
}
```

### ❌ 陷阱 5: 运行次数太少

```bash
# GOOD: 多次运行取平均
go test -bench=. -count=10

# 或者运行更长时间
go test -bench=. -benchtime=10s
```

## 高级 benchstat 用法

### 只看特定指标

```bash
# 只看时间
benchstat -col=/time old.txt new.txt

# 只看内存分配
benchstat -col=/alloc new.txt old.txt
```

### 生成 CSV 格式

```bash
benchstat -format csv old.txt new.txt > result.csv
```

### 对比 delta

```bash
# 显示 delta 百分比
benchstat -delta old.txt new.txt
```

## 性能优化工作流

### 1. 建立基准线

```bash
go test -bench=. -benchmem -count=5 > baseline.txt
```

### 2. 实施优化

修改代码，应用你的优化技巧

### 3. 再次测量

```bash
go test -bench=. -benchmem -count=5 > optimized.txt
```

### 4. 对比结果

```bash
benchstat baseline.txt optimized.txt
```

### 5. 分析差异

- 性能提升是否显著？（看 p-value）
- 哪些 benchmark 提升最大？
- 有没有意外的性能下降？
- 内存分配变化如何？

## 延伸阅读

- [Go 官方 Benchmark 指南](https://golang.org/pkg/testing/#hdr-Benchmarks)
- [benchstat 文档](https://godoc.org/golang.org/x/perf/cmd/benchstat)
- [High Performance Go Workshop](https://dave.cheney.net/high-performance-go-workshop/dotgo-paris.html)
- [Debugging performance issues in Go programs](https://software.intel.com/en-us/blogs/2014/05/10/debugging-performance-issues-in-go-programs)
