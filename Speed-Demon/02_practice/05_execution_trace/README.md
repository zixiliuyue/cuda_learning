# Execution Trace 深度实战

## 概述

`go tool trace` 是 Go 提供的最强大的性能诊断工具之一，它可以记录程序执行的完整时间线，帮助你发现：

- Goroutine 调度延迟
- GC STW 停顿
- 网络阻塞
- 同步原语等待
- syscall 开销

## 运行示例

```bash
go run main.go
```

## 快速开始

### 方法 1: 通过 HTTP 下载

```bash
# 采集 5 秒 trace 并下载
curl http://localhost:6060/trace > trace.out

# 或指定时长
curl "http://localhost:6060/trace?duration=10s" > trace.out
```

### 方法 2: 在代码中保存

```bash
curl http://localhost:6060/save
```

### 方法 3: 用测试生成

```bash
# 运行测试并生成 trace
go test -trace=trace.out
```

## 分析 Trace

```bash
go tool trace trace.out
```

这会启动一个 Web 服务器，通常在 `http://127.0.0.1:6060`

## Trace 视图详解

### 1. View trace（时间轴视图）

这是最核心的视图，按时间轴展示：

```
时间轴 ────────────────────────────────────────────
         │    │    │    │    │    │
Goroutine G0   G1   G2   G3   G4   ...
         │    │╲╱│    │    │    │
         │    │╱╲│    │    │    │
Proc      P0        P1        P2        ...
         │         │         │
Heap      ────────────────────────────────────
GC Mark   ████     ████████     ████
```

**操作技巧**:
- `W/S` 或鼠标滚轮：缩放
- `A/D`：左右移动
- 点击事件：查看详情

**关注指标**:
- **调度延迟**: Goroutine 从 Runnable 到 Running 的时间
- **STW 时间**: GC 的 mark termination 阶段
- **上下文切换频率**: Proc 上的 Goroutine 切换

### 2. Goroutine analysis（Goroutine 分析）

展示每个 Goroutine 的：

- 执行时间
- 调度等待时间
- 阻塞时间（I/O、同步等）
- 系统调用时间

**典型问题识别**:

| 现象 | 可能原因 |
|-----|----------|
| 大量 Goroutine 长时间阻塞在 channel | channel 无缓冲或容量不足 |
| Goroutine 长时间在可运行状态 | 调度器过载，GOMAXPROCS 不足 |
| 大量 Goroutine 同时启动 | thundering herd 问题 |

### 3. Network/Sync blocking profile

展示 Goroutine 阻塞等待的位置：

- 网络 I/O 等待
- Channel 操作
- Mutex 等待
- Cond 等待

**对比 pprof**:
- trace 展示**每次阻塞的具体时间点**
- pprof 展示**统计汇总**

### 4. Minimum mutator utilization（MMU）

展示 GC 对应用的影响：

- Y 轴：应用运行时间占比
- X 轴：时间窗口
- 曲线越低 = GC 影响越大

## 实战案例分析

### 案例 1: 过多 Goroutine

**运行 Bad Demo**:
```bash
hey -n 1000 -c 10 http://localhost:6060/demo/bad
```

**采集 Trace**:
```bash
curl http://localhost:6060/trace > bad_trace.out
go tool trace bad_trace.out
```

**你会看到**:
- Proc 上频繁的上下文切换
- 大量的调度延迟
- Goroutine 在可运行队列等待

**运行 Good Demo**:
```bash
hey -n 1000 -c 10 http://localhost:6060/demo/good
```

**对比差异**:
- 更少的 Goroutine
- 更低的调度开销
- 更好的 CPU 利用率

### 案例 2: GC 停顿分析

```go
// 制造大量 GC 压力
func pressureGC() {
    for i := 0; i < 100000; i++ {
        data := make([]byte, 1024)
        _ = data
    }
}
```

**在 Trace 中观察**:
1. GC 标记阶段（Mark Assist）
2. STW 停顿的位置和时长
3. 是否有 Goroutine 在 GC 时被阻塞

## 自定义 Trace 注解

### 1. Task 注解

```go
ctx, task := trace.NewTask(context.Background(), "ProcessRequest")
defer task.End()

// Task 内的所有代码会被归类
process(ctx, req)
```

### 2. Region 注解

```go
// 方式 1: 显式开始结束
region := trace.StartRegion(ctx, "DatabaseQuery")
result := db.Query(...)
region.End()

// 方式 2: 使用 WithRegion
trace.WithRegion(ctx, "Compute", func() {
    // ... 计算逻辑
})
```

### 3. Log 注解

```go
trace.Log(ctx, "event", "key", "value")
trace.Logf(ctx, "info", "processed %d items", count)
```

## Trace 分析技巧

### 技巧 1: 找调度瓶颈

在 View trace 中:
1. 放大到一个 GC 周期
2. 看 Proc 列，是否有空闲的 P
3. 看 Goroutine 列，有多少在 Runnable 状态

**如果 Runnable 队列很长**:
- 可能是 GOMAXPROCS 太小
- 或者 Goroutine 数量过多

### 技巧 2: 分析 STW

1. 找 GC 标记开始的位置
2. 看 STW 阶段多长
3. 展开看 STW 在做什么

**STW 时间长可能原因**:
- 堆内存太大，扫描时间长
- 有大量的 finalizer
- GC 频率太高

### 技巧 3: 找 syscall 瓶颈

1. 看 Syscall 列
2. 找长时间在 syscall 状态的 P
3. 看是哪个 Goroutine 导致的

**常见原因**:
- 慢的文件 I/O
- 网络调用超时
- 阻塞的系统调用

## Benchmark + Trace 组合使用

### 运行带 trace 的 benchmark

```bash
go test -bench=. -benchtime=10s -trace=trace.out
```

### 分析热点函数

```bash
# 同时生成 pprof profile
go test -bench=. -cpuprofile=cpu.pprof -trace=trace.out

# 先看 cpu profile 找热点
go tool pprof cpu.pprof
(pprof) top

# 再用 trace 看该热点的执行情况
go tool trace trace.out
```

## 常见性能问题模式

### 模式 1: 惊群效应 (Thundering Herd)

**Trace 表现**:
- 大量 Goroutine 同时被唤醒
- Proc 上出现密集的调度活动
- 很多 Goroutine 很快又被抢占

**解决**:
- 使用带缓冲的 channel
- 使用 worker pool
- 避免同时唤醒太多 Goroutine

### 模式 2: 锁竞争严重

**Trace 表现**:
- Goroutine 频繁进入阻塞状态
- 很多时间花在 Sync block
- Proc 利用率不高

**解决**:
- 见锁竞争分析章节

### 模式 3: GC 辅助标记过多

**Trace 表现**:
- GC 期间大量 Mark Assist
- 用户代码执行被打断

**解决**:
- 减少内存分配
- 调整 GOGC
- 考虑对象池

## Trace 性能开销

| 指标 | 典型值 |
|-----|--------|
| CPU 开销 | 10-20% |
| 内存开销 | 几十 MB |
| 存储开销 | ~1MB/秒 |

**建议**:
- 生产环境可以短暂采集
- 采集时间建议 5-30 秒
- 不要连续长时间采集

## 高级用法

### 1. 自定义采集时长

```bash
# 采集 30 秒
curl "http://localhost:6060/trace?duration=30s" > trace.out

# 采集 1 分钟
curl "http://localhost:6060/trace?duration=1m" > trace.out
```

### 2. 对比优化前后

```bash
# 优化前
curl http://localhost:6060/demo/bad
curl http://localhost:6060/trace > bad.out

# 优化后
curl http://localhost:6060/demo/good
curl http://localhost:6060/trace > good.out

# 分别分析对比
go tool trace bad.out
go tool trace good.out
```

### 3. 与 pprof 联动

```bash
# 1. 先用 trace 定位问题时间段
# 2. 用 pprof 在该时间段采集更详细的 profile

# CPU profile
curl http://localhost:6060/debug/pprof/profile?seconds=10

# Heap profile
curl http://localhost:6060/debug/pprof/heap
```

## 延伸阅读

- [Go Execution Tracer 设计文档](https://golang.org/s/go15trace)
- [Diagnostics 官方指南](https://golang.org/doc/diagnostics)
- [使用 go tool trace 诊断延迟问题](https://about.sourcegraph.com/go/an-introduction-to-go-tool-trace)
- [Go 语言高性能编程 - Trace 篇](https://geektutu.com/post/high-performance-go.html)
