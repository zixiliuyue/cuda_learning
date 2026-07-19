# Go 性能反模式与真实故障案例

> 面试加分题："你遇到过哪些印象深刻的性能问题？怎么解决的？"

## 💥 经典性能反模式 Top 10

这些是生产环境踩过的坑，也是面试官最爱问的实战题。

---

### ❌ 反模式 1：Goroutine 泄漏之王 - for + select 没退出条件

**问题代码：**
```go
// BAD: goroutine 永远不会退出！
func process(ch chan Data) {
    go func() {
        for {
            select {
            case data := <-ch:
                handle(data)
            // 没有 default 或超时！
            }
        }
    }()
}
```

**现象：**
- Goroutine 数量缓慢上涨，从几百到几万
- 内存缓慢上涨，GC 压力变大
- 最后 OOM 或者调度器炸了

**怎么发现：**
```bash
# goroutine profile 看大量 goroutine 卡在 chan receive
curl http://localhost:6060/debug/pprof/goroutine?debug=2

# 观察 stack trace：
goroutine 1234 [chan receive]:
main.process.func1()
    /path/to/code.go:42 +0x3c
```

**修复：**
```go
// GOOD: 用 context 控制生命周期
func process(ctx context.Context, ch chan Data) {
    go func() {
        for {
            select {
            case data := <-ch:
                handle(data)
            case <-ctx.Done():  // 加退出条件
                return
            }
        }
    }()
}
```

**面试讲述要点：**
- 讲清楚现象：内存缓慢上涨，OOM
- 讲排查过程：先看 goroutine profile 发现大量卡在 chan receive
- 讲根因：goroutine 没有退出机制
- 讲修复：用 context 控制生命周期
- 讲预防：code review 检查 goroutine 是否有退出方式

---

### ❌ 反模式 2：Time After 泄漏

**问题代码：**
```go
// BAD: 每次循环创建新的 timer，直到超时才会被清理！
for {
    select {
    case data := <-ch:
        handle(data)
    case <-time.After(30 * time.Second):  // ❌ 每次循环都创建新 timer！
        return
    }
}
```

**现象：**
- 内存持续上涨，heap profile 看到大量 time Timer 对象
- GC 次数变多，但都能回收（但有延迟）
- 高并发场景下 1 分钟能泄漏几万个 timer

**怎么发现：**
```bash
# heap profile 看 runtime.timerproc 占比高
go tool pprof http://localhost:6060/debug/pprof/heap
(pprof) top -> runtime.newTimer 占比很高
```

**修复：**
```go
// GOOD: 复用 timer
timer := time.NewTimer(30 * time.Second)
defer timer.Stop()  // 一定要 stop！

for {
    timer.Reset(30 * time.Second)  // 重置而不是新建
    select {
    case data := <-ch:
        handle(data)
    case <-timer.C:
        return
    }
}
```

**面试讲述要点：**
- 这是 Go 官方文档提到的经典坑
- 重点讲：time.After 创建的 timer 在超时前不会被 GC 回收
- 教训：循环里永远不要用 time.After

---

### ❌ 反模式 3：无界 Channel + 慢消费者

**问题代码：**
```go
// BAD: channel 无缓冲，消费者太慢
var ch = make(chan Data)  // 无缓冲！

// 生产者：每秒来 1000 个
func producer() {
    for data := range source {
        ch <- data  // 阻塞在这里！
    }
}

// 消费者：每秒只能处理 100 个
func consumer() {
    for data := range ch {
        slowProcess(data)  // 太慢了！
    }
}
```

**现象：**
- 生产者 goroutine 越来越多，都阻塞在 chan send
- 内存上涨，因为大量 goroutine 栈 + channel 里的数据
- 服务响应越来越慢，最后超时雪崩

**怎么发现：**
```bash
# trace 看大量 goroutine 在 runnable 状态
# goroutine profile 看大量 goroutine 卡在 chan send
go tool pprof http://localhost:6060/debug/pprof/goroutine
```

**修复方案：**
1. 加缓冲 + 限流
2. 增加消费者数量
3. 丢弃或降级（背压）

```go
// GOOD: 有缓冲 + 丢弃策略
var ch = make(chan Data, 10000)  // 加缓冲

func producer() {
    for data := range source {
        select {
        case ch <- data:
            // 正常发送
        default:
            metrics.DropCounter.Inc()  // 统计丢弃
            // 队列满了，直接丢弃或降级
        }
    }
}
```

**面试讲述要点：**
- 提到"背压"概念（加分！）
- 讲系统怎么优雅降级，而不是雪崩
- 提到 channel 一定要考虑缓冲和溢出策略

---

### ❌ 反模式 4：全局锁 + 临界区太大

**问题代码：**
```go
// BAD: 一个全局锁保护所有，临界区超大！
var mu sync.Mutex
var cache = make(map[string]*Data)

func Get(key string) *Data {
    mu.Lock()         // 开始锁
    defer mu.Unlock()
    
    // 从 map 读
    data, ok := cache[key]
    if ok {
        return data
    }
    
    // 去 DB 查！这可是几百 ms！
    data = queryFromDB(key)  // ❌ 慢操作在锁里面！
    cache[key] = data
    
    return data
}
```

**现象：**
- QPS 上不去，压测发现 CPU 不高但延迟很高
- mutex profile 显示锁等待时间超长
- trace 看大量 goroutine 阻塞在 mutex.Lock

**怎么发现：**
```bash
# 启用 mutex profile
runtime.SetMutexProfileFraction(1)

# 查看
go tool pprof http://localhost:6060/debug/pprof/mutex
(pprof) top -> 看哪个锁等的时间最长
```

**修复：**
1. 把慢操作移出临界区
2. 用分片锁减少竞争
3. 用 singleflight 合并重复查询

```go
// GOOD: 先查，没查到去 DB 查完再写回
func Get(key string) *Data {
    // 先读，可以用 RLock
    mu.RLock()
    data, ok := cache[key]
    mu.RUnlock()
    if ok {
        return data
    }
    
    // 没命中，去 DB 查（锁外面！）
    data = queryFromDB(key)
    
    // 查完写回缓存
    mu.Lock()
    cache[key] = data
    mu.Unlock()
    
    return data
}
```

**面试讲述要点：**
- 区分"读多写少"和"读写均衡"场景
- 提到锁粒度：能小则小，能短则短
- 提分片锁、singleflight 等进阶方案

---

### ❌ 反模式 5：疯狂反射 + JSON 序列化

**问题代码：**
```go
// BAD: 每个请求都反射 + 序列化大对象
func HandleRequest(req *Request) {
    // 把整个请求 dump 成 json 打日志
    log.Printf("request: %v", toJSON(req))
    
    // 再把整个响应 dump 成 json
    resp := process(req)
    log.Printf("response: %v", toJSON(resp))
}

func toJSON(v interface{}) string {
    data, _ := json.Marshal(v)  // 每次都反射！
    return string(data)
}
```

**现象：**
- CPU 很高，pprof 看 runtime.reflect 占了 40%+
- GC 频繁，因为产生大量临时 byte slice
- QPS 怎么都上不去

**怎么发现：**
```bash
go tool pprof http://localhost:6060/debug/pprof/profile
(pprof) top -> encoding/json.Marshal 占比超高
```

**修复：**
1. 生产环境不要打全量日志！
2. 需要打日志用更高效的 JSON 库
3. 结构化日志而不是序列化整个对象

```go
// GOOD: 只打必要字段，用高效日志库
import "go.uber.org/zap"

func HandleRequest(req *Request) {
    logger.Info("request", 
        zap.String("id", req.ID),  // 只打关键字段
        zap.Int("size", req.Size),
    )
    // ...
}
```

**面试讲述要点：**
- 提到"可观测性成本"——日志也是有性能成本的
- 区分开发环境和生产环境的日志策略
- 提到 json-iterator、easyjson 等高性能库（加分）

---

### ❌ 反模式 6：defer 在循环里

**问题代码：**
```go
// BAD: defer 在循环里，累积了 10000 个 defer！
func processMany(items []Item) {
    for _, item := range items {  // 10000 次循环
        mu.Lock()
        defer mu.Unlock()  // ❌ 函数返回才执行！
        
        // ... 处理 item
    }
}
```

**现象：**
- 内存暴涨，defer 结构体占了大量内存
- 栈开销大，因为 defer 是栈上分配的
- 最后锁永远不释放？不，是函数返回时才统一释放！

**怎么发现：**
- pprof heap 看 runtime.deferproc 占比
- 或者代码审查发现这个模式

**修复：**
```go
// GOOD: 包装成匿名函数，每轮执行 defer
func processMany(items []Item) {
    for _, item := range items {
        func() {  // 匿名函数
            mu.Lock()
            defer mu.Unlock()  // 匿名函数返回时就执行
            // ... 处理 item
        }()
    }
}

// 或者：不用 defer，直接 Unlock
func processMany(items []Item) {
    for _, item := range items {
        mu.Lock()
        process(item)
        mu.Unlock()
    }
}
```

**面试讲述要点：**
- 解释 defer 的执行时机：函数返回时，不是代码块结束时
- 这是一个经典的"看起来对，实际错"的 Go 陷阱

---

### ❌ 反模式 7：空接口导致的逃逸

**问题代码：**
```go
// BAD: 所有参数都变成 interface{}，全部逃逸到堆！
func Log(format string, args ...interface{}) {
    // ...
}

// 调用
func handle() {
    id := 12345                    // 本来在栈上
    name := "hello"                // 本来在栈上
    Log("id=%d name=%s", id, name) // 全部逃逸到堆！
}
```

**现象：**
- GC 频繁，但 heap inuse 不高
- alloc profile 看大量小对象分配
- 逃逸分析确认大量 escape to heap

**怎么发现：**
```bash
go build -gcflags="-m" ./... 2>&1 | grep escape
```

**修复：**
1. 热路径用类型安全的日志库
2. 避免在高频调用里用 interface{}
3. 用 zap 之类的零分配日志库

```go
// GOOD: 强类型，零分配
import "go.uber.org/zap"

log.Info("processed", zap.Int("id", id), zap.String("name", name))
```

**面试讲述要点：**
- 解释 Go 的逃逸分析机制
- 提到 interface{} 调用会导致参数逃逸
- 这也是为什么很多高性能库避免用 interface{}

---

### ❌ 反模式 8：sync.Pool 放进去又拿出来，等于没池化

**问题代码：**
```go
// BAD: Pool 里放大对象，然后又放新的进去
var bufPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 1024*1024)  // 1MB buffer
    },
}

func handle() {
    buf := bufPool.Get().([]byte)
    // ... 使用 buf ...
    
    // ❌ 因为某些原因，创建新的放回去
    newBuf := make([]byte, 1024*1024)
    bufPool.Put(newBuf)  // 老的 buf 直接扔了，白池化了！
}
```

**现象：**
- GC 频繁，大量内存分配
- Pool 命中率很低
- 内存占用比不用 Pool 还高

**怎么发现：**
```bash
# alloc profile 看大量 make([]byte)
go tool pprof -alloc_space heap.pprof
```

**修复：**
1. 放回去原来的对象
2. Pool 里放小对象，复用率高
3. 加监控看 Pool 命中率

**面试讲述要点：**
- sync.Pool 不是银弹，要看场景
- 适合生命周期短、大小适中的对象
- 放错东西反而起反作用

---

### ❌ 反模式 9：Context 传了，但是没用到

**问题代码：**
```go
// BAD: 接收了 context，但超时根本没传下去
func HandleRequest(ctx context.Context, req *Request) {
    // 调用下游，但没传 ctx！
    resp, err := http.Post("http://downstream/api", "json", body)
    // 下游超时了，这里还在死等！
}
```

**现象：**
- 下游出问题时，本服务 goroutine 暴增
- 大量 goroutine 阻塞在网络 I/O
- 最后服务雪崩

**怎么发现：**
```bash
# trace 看大量 goroutine 在 IO wait
go tool trace trace.out
```

**修复：**
```go
// GOOD: 把 ctx 传下去，超时会取消
func HandleRequest(ctx context.Context, req *Request) {
    req, _ := http.NewRequestWithContext(ctx, "POST", url, body)
    resp, err := http.DefaultClient.Do(req)
    // 下游超时了，这里会立即返回错误
}
```

**面试讲述要点：**
- 提到"故障传播"和"级联失败"
- context 除了传值，更重要的是传递取消信号
- 这是微服务稳定性的关键点（加分！）

---

### ❌ 反模式 10：循环里 append，从不预分配

**问题代码：**
```go
// BAD: QPS 10000，每个请求 append 1000 次
func buildResponse(items []Item) []Data {
    var result []Data  // 没有预分配！
    
    for _, item := range items {  // 1000 次循环
        result = append(result, convert(item))  // 每次可能扩容
    }
    
    return result
}
```

**现象：**
- alloc profile 看 runtime.makeslice 占比很高
- GC 压力大
- 内存带宽成为瓶颈

**怎么发现：**
```bash
go tool pprof -alloc_objects heap.pprof
(pprof) list buildResponse -> 看哪行分配多
```

**修复：**
```go
// GOOD: 预分配，一次到位
func buildResponse(items []Item) []Data {
    result := make([]Data, 0, len(items))  // 预分配容量
    
    for _, item := range items {
        result = append(result, convert(item))
    }
    
    return result
}
```

**面试讲述要点：**
- 这是最简单但收益最大的优化之一
- append 的扩容策略（2 倍增长）的代价
- 强调"知道但不做"是很多团队的问题

---

## 🎯 面试怎么讲性能故事（STAR 法则）

### 模板 1：Goroutine 泄漏案例

> **S (Situation)**：
> "我之前在 XX 公司做订单服务，有一次发现上线一周后，服务内存从 2G 慢慢涨到了 8G，最后 OOM 重启了。"
>
> **T (Task)**：
> "我的任务是找出内存泄漏的原因，并且修复它，同时避免影响线上业务。"
>
> **A (Action)**：
> "首先我看了监控，CPU 不高但内存一直在涨，GC 次数也越来越多。然后我抓了 goroutine profile，发现有 5 万多个 goroutine，而且都卡在同一个地方——channel receive。
> 原来是一个后台 goroutine 处理订单变更，用了 for + select 但是没有退出条件，channel 没数据就永远卡着。
> 修复也简单，加了个 ctx.Done 的 case，服务退出时取消。"
>
> **R (Result)**：
> "上线后 goroutine 数量稳定在 200 左右，内存稳定在 2G。后来我们把'goroutine 必须有退出条件'加入了 code review checklist。"

### 模板 2：锁竞争案例

> **S**：
> "有一次我们做秒杀活动，压测时 QPS 怎么都上不去，只有 200，但 CPU 只用了 20%。"
>
> **T**：
> "我负责排查瓶颈，优化到预期的 2000 QPS。"
>
> **A**：
> "首先看 CPU profile，没发现明显热点。然后抓 mutex profile，发现一个库存检查的锁等待时间占了 80%。
> 原来是一个全局锁保护库存，锁里面还包含了一次 Redis 调用。我把锁拆成了按商品 ID 分片的 256 个锁，
> 同时把 Redis 调用移到锁外面。"
>
> **R**：
> "优化后 QPS 直接到了 2500，锁等待时间从 80% 降到了 5%。那次活动很顺利，没有出问题。
> 后来我们还做了锁的监控，发现锁等待超过一定阈值就告警。"

---

## 💡 性能优化的关键启示

1. **90% 的性能问题都是低级错误** - goroutine 泄漏、锁太大、没预分配...
2. **没有监控 = 瞎子摸象** - 出问题了连从哪下手都不知道
3. **性能问题是最好的老师** - 踩过一次坑，你就记住了
4. **优化的本质是权衡** - 没有免费的午餐，每个优化都有代价

---

## 📋 面试自检清单

讲性能故事时，确保覆盖这些点：

- [ ] 问题背景是什么，业务影响多大？
- [ ] 你是怎么一步步排查的，用了什么工具？
- [ ] 根因是什么，为什么会发生？
- [ ] 你是怎么修复的，有几种方案，为什么选这个？
- [ ] 优化前后的数据对比（量化！）
- [ ] 后续怎么预防，有没有加监控、改流程？
- [ ] 从这个问题里你学到了什么？

**记住：面试官能从你讲问题的方式，看出你是不是真的做过。**
