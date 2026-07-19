# HTTP/RPC 性能优化最佳实践

> **面试必问**：接口超时怎么处理？熔断降级怎么做？

---

## ⏱️ 超时控制：最基础也最重要

### 为什么必须设置超时？

```go
// ❌ 没有超时：下游挂了，你的 goroutine 也永远卡住！
func badCallRPC(ctx context.Context, req Request) (*Response, error) {
    return client.Call(ctx, req)  // 没设置超时！永远等下去！
}
```

**正确的超时设置原则：**

```
整体超时 > 单次调用超时 × 重试次数
```

---

### Go 标准 Timeout 模式

```go
// ✅ 统一超时控制
func goodCallRPC(ctx context.Context, req Request) (*Response, error) {
    // 给每个 RPC 调用设置超时
    ctx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
    defer cancel()
    
    select {
    case resp := <-client.CallAsync(ctx, req):
        return resp, nil
    case <-ctx.Done():
        return nil, ctx.Err()  // 超时返回！
    }
}
```

---

### 超时设置经验值

| 调用类型 | 推荐超时 | 说明 |
|---------|---------|------|
| Redis 缓存 | 50-100ms | 内网很快 |
| 数据库查询 | 100-500ms | 复杂查询可以更长 |
| HTTP 接口调用 | 1-3s | 依赖网络质量 |
| RPC 微服务调用 | 500ms-2s | 根据下游 SLA |
| 消息队列消费 | 30s | 可以长一点 |

---

## 🔄 重试机制

### 什么时候该重试？

✅ **可以重试的错误：**
- 网络超时、临时 503
- 限流返回的 429
- 服务刚启动不可用

❌ **绝对不能重试的错误：**
- 400 Bad Request（参数错了，重试也没用）
- 401 Unauthorized（没权限）
- 403 Forbidden（被拒绝）
- 404 Not Found（资源不存在）
- 500 Internal Server Error（服务端报错，可能是脏数据）

---

### 指数退避 + 抖动

```go
// ❌ 立即重试：1ms 内重试 10 次，把下游打死！
func badRetry() {
    for i := 0; i < 3; i++ {
        err := call()
        if err == nil {
            return
        }
        // 立即重试！
    }
}

// ✅ 指数退避 + 随机抖动
func goodRetry(ctx context.Context, maxRetries int) error {
    var err error
    for i := 0; i < maxRetries; i++ {
        if err = call(); err == nil {
            return nil
        }
        
        // 指数退避：100ms, 200ms, 400ms...
        backoff := time.Duration(100*(1<<i)) * time.Millisecond
        
        // 加随机抖动防止惊群效应
        jitter := time.Duration(rand.Intn(50)) * time.Millisecond
        backoff += jitter
        
        select {
        case <-time.After(backoff):
            continue
        case <-ctx.Done():
            return ctx.Err()
        }
    }
    return err
}
```

---

## 🛡️ 限流：保护自己不被打死

### 常用限流算法对比

| 算法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| **计数器** | 简单，内存少 | 边界问题，突刺 | 要求不高的场景 |
| **滑动窗口** | 精确 | 实现复杂 | 精度要求高 |
| **令牌桶** | 平滑，允许一定突刺 | 内存稍大 | 推荐！大部分场景 |
| **漏桶** | 绝对平滑 | 不允许突刺 | 流量整形 |

---

### Go 实现令牌桶限流

```go
import "golang.org/x/time/rate"

// 限流器：每秒 1000 个请求，桶容量 200
var limiter = rate.NewLimiter(1000, 200)

func HandleRequest(w http.ResponseWriter, r *http.Request) {
    // ✅ 先过限流
    if !limiter.Allow() {
        w.WriteHeader(429)  // Too Many Requests
        w.Write([]byte("rate limited"))
        return
    }
    
    // 正常处理请求
    process(r)
}

// ✅ 每个用户单独限流（更公平）
var userLimiters = make(map[int]*rate.Limiter)
var mu sync.RWMutex

func getUserLimiter(userID int) *rate.Limiter {
    mu.RLock()
    l, ok := userLimiters[userID]
    mu.RUnlock()
    
    if ok {
        return l
    }
    
    mu.Lock()
    defer mu.Unlock()
    // 每个用户每秒 10 次请求
    userLimiters[userID] = rate.NewLimiter(10, 20)
    return userLimiters[userID]
}
```

---

## 🔥 熔断：保护下游不被打死

### 熔断器状态机

```
                   失败率超过阈值
Closed ──────────────────────────────→ Open
  ↑                                         │
  │                                         │  熔断一段时间
  │  半开探测成功                            ↓
Half-Open ←───────────────────────────── Open
     │
     └────→ 半开探测失败 → Open
```

---

### Go 实现简单熔断器

```go
type CircuitBreaker struct {
    state          int32  // 0=closed, 1=open, 2=half-open
    failureCount   int64
    lastFailureTime int64
    
    // 配置
    failureThreshold  int64         // 连续失败多少次熔断
    timeout           time.Duration // 熔断多久后半开
    halfOpenMaxCalls  int           // 半开状态允许多少请求探测
}

func (cb *CircuitBreaker) Call(f func() error) error {
    // 1. 检查状态
    state := atomic.LoadInt32(&cb.state)
    
    // 熔断状态，直接拒绝
    if state == 1 {
        if time.Now().UnixNano()-atomic.LoadInt64(&cb.lastFailureTime) < int64(cb.timeout) {
            return errors.New("circuit breaker open")
        }
        // 超时了，切到半开状态
        atomic.CompareAndSwapInt32(&cb.state, 1, 2)
        atomic.StoreInt64(&cb.failureCount, 0)
    }
    
    // 2. 调用函数
    err := f()
    
    // 3. 统计成功/失败
    if err != nil {
        fc := atomic.AddInt64(&cb.failureCount, 1)
        atomic.StoreInt64(&cb.lastFailureTime, time.Now().UnixNano())
        
        // 半开状态失败，或者失败超过阈值 → 熔断
        if (state == 2) || (state == 0 && fc >= cb.failureThreshold) {
            atomic.CompareAndSwapInt32(&cb.state, state, 1)
        }
        return err
    }
    
    // 成功
    if state == 2 {
        // 半开状态成功 → 关闭
        atomic.CompareAndSwapInt32(&cb.state, 2, 0)
    }
    atomic.StoreInt64(&cb.failureCount, 0)
    return nil
}
```

---

## 📊 降级：宁可部分可用，不可完全不可用

### 降级策略

| 降级策略 | 场景 | 示例 |
|---------|------|------|
| **返回默认值** | 非核心功能 | 推荐接口挂了，返回空列表/热门商品 |
| **返回缓存值** | 读多写少 | 商品详情挂了，返回 1 小时前的缓存 |
| **简化流程** | 非必要步骤 | 跳过埋点、日志、风控等 |
| **关闭非核心** | 大促高峰 | 关闭评论/收藏功能，保证下单可用 |

---

### 降级示例代码

```go
func GetProductDetail(ctx context.Context, id int) (*Product, error) {
    // 1. 先查缓存
    product, err := redisGetProduct(id)
    if err == nil {
        return product, nil
    }
    
    // 2. 缓存没了，查下游服务
    product, err = productService.Get(ctx, id)
    if err == nil {
        // 写回缓存
        redisSetProduct(id, product)
        return product, nil
    }
    
    // ❌ 下游挂了！降级！
    // ✅ 返回缓存里的旧值（可能过期了，但总比报错好）
    staleProduct, err := redis.Get(fmt.Sprintf("product_stale:%d", id))
    if err == nil {
        return staleProduct, nil  // 返回旧数据
    }
    
    // 连旧数据都没有，返回兜底默认值
    return getDefaultProduct(id), nil
}
```

---

## ⚠️ 常见坑

### 坑 1: 重试风暴

```go
// ❌ A 调用 B，B 调用 C，每个都重试 3 次
// C 挂了：A 发 3 次，每个 B 发 3 次 → 9 倍流量！
// C 本来就挂了，现在更挂了！

// ✅ 只在最上层重试，下层不要重试
func A_Call() {
    // A 层重试 2 次
    retry(func() error {
        return B_Call()  // B 层不重试！
    }, 2)
}

func B_Call() error {
    // 不重试，直接透传给上层
    return C_Call()
}
```

---

### 坑 2: 超时设置太短

```go
// ❌ 超时设 10ms，但接口 P99 就是 100ms
// 结果：所有请求都超时，等于接口不可用！

// ✅ 怎么设置合理的超时？
// 1. 看下游的 P99 延迟
// 2. 超时 = P99 × 1.5 ~ 2 倍
// 3. 压测验证！
```

---

### 坑 3: 熔断阈值太敏感

```go
// ❌ 连续失败 2 次就熔断
// 网络偶尔抖一下就熔断了，用户体验差

// ✅ 结合失败率和最小请求数
// 比如：1 分钟内请求 >= 100，失败率 >= 50% 才熔断
```

---

## 💯 面试回答模板

**Q: 一个高并发接口，你会做哪些稳定性保障？**

> "我会从这几个层面做稳定性保障：
>
> **第一层：超时控制**
> 每个 RPC/DB/Redis 调用都必须设置超时，绝对不能无限等下去。
> 超时时间参考下游 P99 延迟，乘以 1.5~2 倍。
>
> **第二层：重试机制**
> 只有临时错误才重试，参数错、权限错这种绝对不能重试。
> 重试必须用指数退避加随机抖动，防止重试风暴。
> 而且只在最上层重试，下层不要重试，防止放大。
>
> **第三层：限流**
> 保护自己不被打死，用令牌桶算法，每秒多少请求，超过就直接返回。
> 最好是按用户维度限流，更公平，防止单个用户打满整个服务。
>
> **第四层：熔断**
> 保护下游不被打死，连续失败超过阈值就熔断，
> 一段时间后放少量请求探测，恢复了就关闭熔断。
>
> **第五层：降级**
> 非核心功能挂了，返回旧数据或者默认值，
> 绝对不能因为非核心功能挂了影响核心链路。
>
> **最后：监控告警**
> P99 延迟、错误率、QPS 这几个黄金指标必须有监控，
> 超过阈值立刻告警，别等用户投诉了才知道。
>
> **记住原则：** 宁可部分可用，不可完全不可用！"

---

## ✅ HTTP/RPC 稳定性 Checklist

- [ ] 每个调用都设置超时了吗？
- [ ] 超时时间合理吗？参考下游 P99 了吗？
- [ ] 重试只重试临时错误吗？
- [ ] 重试有指数退避和抖动吗？
- [ ] 只在最上层重试，下层不重试吗？
- [ ] 加限流了吗？令牌桶还是滑动窗口？
- [ ] 加熔断了吗？熔断阈值合理吗？
- [ ] 有降级策略吗？下游挂了能返回旧数据吗？
- [ ] 黄金指标都有监控告警吗？

---

## 🚀 动手练习

1. 实现一个带超时、重试、指数退避的 HTTP 客户端
2. 实现一个简单的熔断器，模拟熔断状态切换
3. 写一个令牌桶限流的中间件，压测看效果

**思考题：** 为什么重试要加随机抖动？不加会有什么问题？
