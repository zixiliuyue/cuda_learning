# 🎯 端到端性能故障复现案例

> 场景：某电商商品详情接口，大促前压测发现性能问题
>
> **面试讲故事神器！按这个流程走一遍，面试官直接跪了

---

## 📋 故事背景

**时间**：大促前 2 周压测
**接口**：/api/product/detail?id=123
**现象**：
- 压测 100 并发时：QPS 只有 15，P99 延迟 6 秒
- CPU 使用率 30%，内存 800MB
- 数据库 CPU 不高，但连接池满了
- Goroutine 数量持续上涨不回落

**你作为 Senior SRE / Go 开发工程师，负责排查这个问题

---

## 🎬 故障复现步骤

### Step 1: 启动故障版本
```bash
cd v1_bad
go run main.go
```

服务启动在 `http://localhost:8080`

### Step 2: 用 hey 压测
```bash
# 100 并发，压 30 秒
hey -n 3000 -c 100 http://localhost:8080/api/product?id=123
```

### Step 3: 一边压测同时观察指标
开另一个终端看：

```bash
# 1. 看 goroutine 数量
curl http://localhost:8080/debug/vars | grep goroutine

# 2. heap profile
go tool pprof http://localhost:8080/debug/pprof/heap

# 3. goroutine profile
curl http://localhost:8080/debug/pprof/goroutine?debug=2 > gor.txt
cat gor.txt | grep chan | wc -l

# 4. mutex profile
go tool pprof http://localhost:8080/debug/pprof/mutex
```

---

## 🔍 排查过程（面试就按这个讲）

### 📍 问题 1: 数据库连接池满了

**现象**：
```
# 压测时看 trace 里大量 goroutine 卡在：
database/sql.(*DB).conn(0x0...
  -> database/sql.(*DB).queryConn...
    -> select...
```

**根因**：
```go
// v1_bad/main.go 第 42 行
db, err := sql.Open("mysql", dsn)
// ❌ 没设置最大连接数！默认 0 = 无限！
```

**修复**：
```go
db.SetMaxOpenConns(50)   // 最大打开连接
db.SetMaxIdleConns(25)    // 最大空闲连接
db.SetConnMaxLifetime(time.Hour)
```

**效果**：QPS 从 15 → 40，P99 从 6s → 2s

---

### 📍 问题 2: 全局大锁

**现象**：
```
mutex profile 显示：
18.87s 70% 的锁等待时间
lockCounter lock
```

**根因**：
```go
// v1_bad/main.go 第 15 行
var cacheLock sync.Mutex

// 锁里包含了 DB 查询！！
func GetProduct(id int) *Product {
    cacheLock.Lock()         // ❌ 进来就上锁
    defer cacheLock.Unlock()

    p, ok := localCache[id]
    if ok {
        return &p
    }
    
    p = queryFromDB(id)  // ❌ 慢操作在锁里！
    localCache[id] = p
    return &p
}
```

**修复**：双重检查锁 + 锁只锁 map 操作
```go
func GetProduct(id int) *Product {
    // 1. 先读（可以加读锁，或不加锁）
    cacheLock.RLock()
    p, ok := localCache[id]
    cacheLock.RUnlock()
    if ok {
        return &p
    }
    
    // 2. 没命中才去 DB（锁外面！）
    p = queryFromDB(id)
    
    // 3. 写回缓存（只锁写操作
    cacheLock.Lock()
    localCache[id] = p
    cacheLock.Unlock()
    
    return &p
}
```

**效果**：QPS 从 40 → 120，P99 从 2s → 800ms

---

### 📍 问题 3: for+select goroutine 泄漏

**现象**：
```
压测前 goroutine: 20
压测后 goroutine: 1500+
而且还在持续上涨！

goroutine profile 看：
1200+ goroutine 卡在：
  select {
  case <-ctx.Done():
  case item <- ch:
  }

  但是 ch 没人写！！
  }
```

**根因**：
```go
// v1_bad/stock.go 第 18 行
func CheckStock(ctx context.Context, id int) int {
    ch := make(chan int, 1)
    go func() {
        // 查库存 RPC，超时 3 秒
        ch <- queryStockRPC(ctx, id)
    }()
    
    select {
    case stock := <-ch:
        return stock
    case <-ctx.Done():  // ❌ 超时了，但 goroutine 还在查 RPC，没人接！
        return 0
    }
    // ch 没缓冲，goroutine 写不进去，永远卡住！
}
```

**修复**：channel 加缓冲
```go
ch := make(chan int, 1)  // ✅ 加缓冲，goroutine 写完就可以退出了
```

**效果**：goroutine 稳定在 50 左右，不再泄漏

---

### 📍 问题 4: 没有预分配 + 疯狂 append

**现象**：
```
alloc profile 看：
json.Marshal 每次分配占 60%
每次接口分配 3MB/s，每秒 120 次接口 = 360MB/s 分配
GC 每秒 2 次
```

**根因**：
```go
// v1_bad/main.go 第 68 行
func (p *Product) ToJSON() []byte {
    data, _ := json.Marshal(p)
    return data
}
```

**修复**：
1. 用 jsoniter 替代标准库 json
2. 用 sync.Pool 复用 buffer

**效果**：GC 停顿减少 70%，P99 从 800ms → 400ms

---

### 📍 问题 5: 没有用 singleflight，重复查 DB

**现象**：
```
同一个 product id 100 并发请求，DB 查了 100 次！
相同的 SQL 执行了 100 次
```

**根因**：缓存击穿
**修复**：加 singleflight 合并相同请求
```go
var sf singleflight.Group

func GetProduct(id int) *Product {
    // 相同 id 的并发只查一次 DB
    result, err, _ := sf.Do(strconv.Itoa(id), func() (interface{}, error) {
        // 查缓存 + DB
    })
    return result.(*Product)
}
```

**效果**：DB QPS 降为原来的 1/50，P99 从 400ms → 150ms

---

## 📊 优化前后对比

| 指标 | 优化前 | 优化后 | 提升 |
|-----|-------|--------|------|
| **QPS** | 15 | 350 | **23 倍** |
| **P99 延迟 | 6000ms | 120ms | **50 倍** |
| **Goroutine 数** | 1500+ | 50 | **97% 减少 |
| **GC 频率** | 2次/秒 | 1次/5秒 | **10 倍减少 |
| **DB QPS** | 150/秒 | 3/秒 | **50 倍减少** |
| **CPU 使用率** | 30% | 45% | 资源合理利用 |

---

## 💯 面试怎么讲这个故事（STAR 法则）

> **S (Situation)**
> "大促前压测商品详情接口，100 并发时 QPS 只有 15，P99 延迟 6 秒，完全达不到大促 300 QPS 的目标。
>
> **T (Task)**
> 我作为主要负责人，负责定位和优化。
>
> **A (Action)**
> 我按这个流程排查的：
> 1. 先看监控四个黄金指标，发现：CPU不高但延迟大，goroutine 数量在涨，说明不是 CPU 瓶颈，怀疑是锁或者 IO 阻塞。
>
> 2. 抓 goroutine profile，发现 1000多个 goroutine 卡在获取数据库连接，连接池满了，一看代码没设 MaxOpenConns，默认无限创建连接。先加了连接池设置。
>
> 3. 再压测，QPS 到了 40，但还是不够。抓 mutex profile，发现锁等待占了 70% 时间，
一看缓存的锁里包含了 DB 查询。把 DB 操作移到锁外面，QPS 到了 120。
>
> 4. 但 goroutine 还在涨，抓 goroutine profile 看到大量卡在 channel 接收，
发现查库存的超时了但 channel 没缓冲，goroutine 泄漏了，加了缓冲就好了。
>
> 5. 最后看 alloc profile，json 序列化分配太多，用了对象池和 faster json 库，
还加了 singleflight 合并相同查询，减少 DB 压力。
>
> **R (Result)**
> 最后 QPS 从 15 到 350，P99 从 6 秒降到 120 毫秒，DB 压力降了 50 倍。
> 后来我们把这次的经验整理成了 checklist，code review 时照着勾，
避免类似问题再上线。"

---

## 🚀 验证优化效果

```bash
# 启动优化版本
cd ../v2_good
go run main.go

# 再压测
hey -n 10000 -c 100 http://localhost:8080/api/product?id=123
```

对比前后结果，感受 20 倍性能提升的快感！💪
