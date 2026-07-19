# GC 工作原理与调优策略

## Go GC 算法演进

| 版本 | 算法 | 特点 |
|-----|------|------|
| Go 1.0 | 标记-清除 (Mark-Sweep) | 全 Stop-The-World |
| Go 1.3 | 标记-清除 | 并行清除，减少 STW |
| Go 1.5 | 三色标记 + 写屏障 | 并发标记，STW < 10ms |
| Go 1.8 | 混合写屏障 | 消除 STW 阶段 re-scan |
| Go 1.14 | 页分配器重写 | 更好的内存复用 |

## 三色标记抽象

### 三色对象

```
  黑色 (Black)
  ────────────
  已扫描完成
  所有引用的对象也已扫描
      │
      │ 引用
      ▼
  灰色 (Gray)
  ────────────
  已标记但未扫描
  等待被扫描
      │
      │ 引用
      ▼
  白色 (White)
  ────────────
  未访问
  可能是垃圾
```

### 标记过程

```
初始状态:
  [根对象] ───▶ 黑色
  其他对象 ───▶ 白色

1. 从黑色对象出发，标记可达对象为灰色
2. 逐个扫描灰色对象
3. 扫描完成，将灰色变为黑色
4. 重复直到无灰色对象

最终:
  黑色 = 存活
  白色 = 垃圾
```

## 并发标记的难点

### 三色不变性 (Tri-color Invariant)

**黑对象不能直接引用白对象**

如果破坏这个不变性，可能导致：
- 白对象被黑对象引用
- 但白对象不在灰色集合中
- 最终被当作垃圾回收！

### 解决方案：写屏障 (Write Barrier)

```go
// 简化的写屏障伪代码
func writePointer(slot *Object, newObj *Object) {
    // 1. 如果 slot 指向的对象是白色，标记为灰色
    if color(*slot) == white {
        markGray(*slot)
    }
    // 2. 如果 newObj 是白色，标记为灰色
    if color(newObj) == white {
        markGray(newObj)
    }
    // 3. 执行实际的写操作
    *slot = newObj
}
```

**写屏障的作用**：
- 确保所有可能被引用的对象都被标记
- 允许 GC 与应用并发运行
- 有一定的性能开销 (~5-10%)

## GC 完整周期

```
┌─────────────────────────────────────────────────────┐
│                  GC 周期                           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐                                       │
│  │  Mark 1  │ ◀── STW，设置写屏障                  │
│  └────┬─────┘                                       │
│       │                                              │
│       ▼                                              │
│  ┌─────────────────────────────────────────────┐   │
│  │           Mark 2 (并发标记)                 │   │
│  │  - 从根出发扫描所有可达对象                  │   │
│  │  - 写屏障记录新的引用                       │   │
│  └──────────────────┬──────────────────────────┘   │
│                     │                               │
│                     ▼                               │
│  ┌─────────────────────────────────────────────┐   │
│  │           Mark Termination (STW)            │   │
│  │  - 重新扫描工作队列                         │   │
│  │  - 关闭写屏障                               │   │
│  └──────────────────┬──────────────────────────┘   │
│                     │                               │
│                     ▼                               │
│  ┌─────────────────────────────────────────────┐   │
│  │           Sweep (并发清除)                  │   │
│  │  - 清扫白色对象                             │   │
│  │  - 将内存归还给 heap                        │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 各阶段时间分布

```
阶段              典型时间    是否 STW
───────────────────────────────────────────
Mark 1             ~1ms       ✓
Mark 2            几十 ms
Mark Termination   1-3ms      ✓
Sweep             并发进行

总 STW 时间: ~5ms (Go 1.14+)
```

## GC 触发条件

### 1. 堆内存增长

```
GC 触发阈值 = 上次 GC 后堆大小 × GOGC
```

**GOGC 环境变量**：
- 默认值 = 100 (即堆增长 100% 时触发)
- GOGC=off 关闭 GC（不推荐，除非知道自己在做什么）
- GOGC=50 更频繁的 GC（更低的内存占用）

### 2. 时间触发

如果超过 2 分钟没有 GC，强制执行一次。

### 3. 手动触发

```go
runtime.GC()  // 阻塞直到 GC 完成
```

## GC 调优策略

### 1. 调整 GOGC

**低延迟优先**：
```bash
# 更频繁的 GC，减少每次 GC 的工作量
GOGC=50 ./myapp
```

**高吞吐优先**：
```bash
# 减少 GC 频率，增加吞吐
GOGC=200 ./myapp
```

### 2. 减少分配

**这是最重要的 GC 优化**

| 技术 | 效果 |
|-----|------|
| sync.Pool | 减少临时对象分配 |
| 预分配 slice | 避免多次扩容 |
| 值传递 vs 指针 | 减少逃逸 |
| 结构体重排 | 减少内存占用 |

### 3. 对象生命周期管理

**短生命周期对象**：
- 通常在年轻代分配（如果 Go 有分代的话...）
- 实际：Go 没有分代 GC，但小对象分配快

**长生命周期对象**：
- 放在全局或包级变量
- 考虑对象池
- 避免频繁创建销毁

## GC 观测工具

### 1. GODEBUG=gctrace=1

```bash
GODEBUG=gctrace=1 ./myapp
```

**输出示例**：
```
gc 1 @0.020s 2%: 0.015+0.23+0.004 ms clock, 0.12+0.11/0.060/0.13+0.032 ms cpu, 4->4->0 MB, 5 MB goal, 8 P
```

**解读**：
```
gc 1                  - 第 1 次 GC
@0.020s               - 程序启动 0.020 秒
2%                    - GC 占 CPU 2%
0.015+0.23+0.004 ms  - STW 阶段时间 (mark1 + mark termination)
4->4->0 MB           - GC前堆 -> GC后堆 -> 存活大小
5 MB goal            - 本次 GC 目标堆大小
8 P                   - GOMAXPROCS=8
```

### 2. GC Trace

使用 execution trace 查看 GC 详细信息：

```go
import "runtime/trace"

f, _ := os.Create("trace.out")
defer f.Close()
trace.Start(f)
defer trace.Stop()
```

```bash
go tool trace trace.out
```

**在 trace 中可以看到**：
- GC 各阶段的精确时间
- GC 期间哪些 goroutine 被阻塞
- 辅助 GC 的 goroutine 分布
- 协助 GC 的标记工作占比

### 3. pprof - gc_latency

```bash
# 获取 GC 延迟 profile
curl http://localhost:6060/debug/pprof/heap?gc=1

# 分析
go tool pprof -alloc_space http://localhost:6060/debug/pprof/heap
```

## 常见 GC 问题与诊断

### 1. GC 频率过高

**现象**：
- gctrace 中 GC 间隔 < 100ms
- GC CPU 占用 > 20%

**诊断步骤**：
```bash
# 1. 看 GC 频率
GODEBUG=gctrace=1 ./myapp

# 2. 看分配热点
go tool pprof -alloc_space http://localhost:6060/debug/pprof/heap

# 3. 看逃逸分析
go build -gcflags="-m" ./...
```

**解决方案**：
- 减少小对象分配
- 使用 sync.Pool
- 增大 GOGC

### 2. GC 延迟过高

**现象**：
- STW 时间 > 10ms
- 尾延迟尖刺

**诊断步骤**：
```bash
# 1. 看详细 GC 阶段时间
GODEBUG=gctrace=1,gcpacertrace=1 ./myapp

# 2. 用 trace 看具体哪个阶段慢
go tool trace trace.out

# 3. 看是否有大对象扫描
```

**解决方案**：
- 减少超大对象（> 1MB）
- 避免在 GC 期间分配大量内存
- 考虑手动内存管理（如 arena）

### 3. 内存持续增长

**是否是内存泄漏？**

```go
// 监控内存趋势
func monitorMem() {
    ticker := time.NewTicker(5 * time.Second)
    for range ticker.C {
        var stats runtime.MemStats
        runtime.ReadMemStats(&stats)
        fmt.Printf("HeapAlloc: %d MB\n", stats.HeapAlloc/1024/1024)
    }
}
```

**诊断内存泄漏**：
```bash
# 1. 多次 heap profile
curl http://localhost:6060/debug/pprof/heap > heap1.pprof
sleep 60
curl http://localhost:6060/debug/pprof/heap > heap2.pprof

# 2. 对比差异
go tool pprof -base heap1.pprof heap2.pprof
(pprof) top
```

## 高级 GC 主题

### 1. GC 辅助 (GC Assist)

**原理**：
- 应用 goroutine 分配内存时，如果发现 GC 正在进行
- 会被要求"帮忙"做一些标记工作
- 这称为 GC assist

**权衡**：
- (+) 防止堆在 GC 期间无限膨胀
- (-) 分配延迟增加

**观察**：
```
# 在 gctrace 中看到 assist 时间
gc assist: 0.2+0.5 ms
```

### 2. 清扫 pacing

Go 有一个复杂的 pacing 算法，用来：
- 预测下一次 GC 触发时机
- 调整 GC 工作速率
- 平衡 CPU 使用与内存占用

### 3. 大对象处理

**大对象 (> 32KB)**：
- 不从 mcache 分配，直接从 mheap 分配
- 标记扫描成本高
- 建议复用或拆分

## 不同场景的 GC 配置

### Web 服务（低延迟优先）

```bash
# 降低单次 GC 工作量
GOGC=50

# 关注：
# - 99 分位延迟
# - GC 频率是否合理
# - 最大 GC STW 时间
```

### 离线处理（高吞吐优先）

```bash
# 减少 GC 频率，提升吞吐
GOGC=200

# 关注：
# - 总处理时间
# - 内存峰值是否可接受
# - GC 总 CPU 占比
```

### 软实时系统

```bash
# 最激进的 GC 配置
GOGC=30

# 可能还需要：
# - 控制并发度
# - 避免大对象
# - 预留足够的 CPU 给 GC
```

## GC 未来展望

### Go 1.x 可能的改进

1. **分代 GC** - 年轻代/老年代
   - 降低扫描成本
   - 但 Go 的 escape analysis 已经把很多对象留在栈上

2. **更细粒度的并发**
   - 进一步减少 STW 时间
   - 目标: STW < 1ms

3. **用户态控制**
   - 更灵活的 GC 触发 API
   - 按优先级 GC

## 深入阅读

- [Go GC: Latency, Throughput, and the 10ms Pause](https://blog.golang.org/ismmkeynote)
- [Go 1.5 GC 文档](https://docs.google.com/document/d/167Yn9cYQ-3kfGmZeaZ013p9XUj9hYVjM9VZ-JdX0M7Q)
- [Getting to Go: The Journey of Go's Garbage Collector](https://blog.golang.org/ismmkeynote)
- [Go Runtime Source - mgc.go](https://github.com/golang/go/blob/master/src/runtime/mgc.go)
