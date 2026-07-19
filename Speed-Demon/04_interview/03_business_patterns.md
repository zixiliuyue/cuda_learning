# 业务场景性能优化 - 数据库、缓存、gRPC

> 面试必问："你们线上服务的性能瓶颈一般在哪？怎么优化的？"

## 🗄️ 数据库性能优化

### 反模式 1：N+1 查询

**问题代码：**
```go
// BAD: 查订单列表，然后循环每个订单去查用户信息
func GetOrders(ctx context.Context, userIDs []int64) ([]*Order, error) {
    orders, _ := db.QueryContext(ctx, "SELECT * FROM orders WHERE ...")
    
    for _, order := range orders {
        // ❌ 每个订单都查一次 DB！
        user, _ := db.QueryContext(ctx, 
            "SELECT name FROM users WHERE id = ?", 
            order.UserID,
        )
        order.UserName = user.Name
    }
    
    return orders, nil
}
```

**现象：**
- 100 个订单就会产生 101 次 DB 查询
- DB QPS 很高，但业务 QPS 上不去
- 接口延迟 = 1 次订单查询 + 100 次用户查询

**怎么发现：**
1. 链路追踪看一个请求有多少 DB 调用
2. 看 DB 监控，QPS 异常高
3. 慢日志里有大量相同的单条查询

**修复方案 1：IN 查询（推荐）**
```go
// GOOD: 先拿所有 ID，批量查询
func GetOrders(ctx context.Context, userIDs []int64) ([]*Order, error) {
    orders, _ := db.QueryContext(ctx, "SELECT * FROM orders WHERE ...")
    
    // 收集所有 userID
    ids := collectUserIDs(orders)
    
    // 批量查用户信息 - 1 次查询
    users, _ := db.QueryContext(ctx, 
        "SELECT id, name FROM users WHERE id IN (?)", 
        ids,
    )
    
    // 组装
    userMap := make(map[int64]string)
    for _, u := range users {
        userMap[u.ID] = u.Name
    }
    
    for _, order := range orders {
        order.UserName = userMap[order.UserID]
    }
    
    return orders, nil
}
```

**修复方案 2：SQL JOIN（简单场景）**
```sql
SELECT o.*, u.name FROM orders o
JOIN users u ON o.user_id = u.id
WHERE ...
```

**面试讲述要点：**
- 明确 N+1 的定义：1 次主查询 + N 次子查询
- 影响：DB 压力 ×10，延迟 ×10
- 排查：链路追踪看 span 数量，或者看日志里的 SQL 数量
- 两种方案的权衡：IN 查询 vs JOIN

---

### 反模式 2：事务里做慢操作

**问题代码：**
```go
// BAD: 事务里做 RPC 调用！
func CreateOrder(ctx context.Context, req *CreateOrderReq) error {
    tx, _ := db.BeginTx(ctx, nil)
    
    // 1. 写订单表 - 快速
    tx.ExecContext(ctx, "INSERT INTO orders ...")
    
    // 2. 调用库存服务扣库存 - ❌ 慢！可能几百 ms！
    // 事务持有期间做 RPC！锁不释放！
    inventoryClient.Deduct(ctx, req.SKU, req.Qty)
    
    // 3. 写订单流水
    tx.ExecContext(ctx, "INSERT INTO order_logs ...")
    
    return tx.Commit()
}
```

**现象：**
- 事务持有时间超长
- 数据库连接池很快耗尽
- 行锁冲突严重，大量锁等待超时

**怎么发现：**
- 数据库监控看事务平均执行时间
- Innodb 状态看 row lock wait time
- trace 看 DB span 之间夹着 RPC span

**修复方案：**
```go
// GOOD: 事务里只做 DB 操作，不做外部调用
func CreateOrder(ctx context.Context, req *CreateOrderReq) error {
    // 1. 先调库存服务（事务外！）
    if err := inventoryClient.Deduct(ctx, req.SKU, req.Qty); err != nil {
        return err
    }
    
    // 2. DB 操作放在一起，快速完成
    tx, _ := db.BeginTx(ctx, nil)
    tx.ExecContext(ctx, "INSERT INTO orders ...")
    tx.ExecContext(ctx, "INSERT INTO order_logs ...")
    return tx.Commit()
}
```

**极端情况方案：SAGA 模式**
- 先写订单（PENDING 状态）
- 发消息到 MQ 异步扣库存
- 库存成功回调更新订单状态

**面试讲述要点：**
- 事务原则：越小越快越好
- 区分"必须事务保证"和"可以最终一致"的操作
- 提到分布式事务的几种方案：2PC、TCC、SAGA

---

### 反模式 3：全表扫描

**问题代码：**
```go
// BAD: 没索引！每次全表扫描！
func GetUserByEmail(ctx context.Context, email string) (*User, error) {
    // users 表 email 字段没索引！
    row := db.QueryRowContext(ctx, 
        "SELECT * FROM users WHERE email = ?", 
        email,
    )
    // ...
}
```

**现象：**
- CPU 被 DB 打满
- 这个 SQL 执行时间从几 ms 变成几秒
- 数据量越大越慢（线性增长）

**怎么发现：**
- 数据库慢查询日志
- Explain 分析 SQL：type 列是 ALL 就是全表扫描

```sql
EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';
-- 看 key 列是不是 NULL，rows 是不是很大
```

**修复：**
```sql
-- 加索引
CREATE INDEX idx_users_email ON users(email);
```

**进阶：覆盖索引**
```sql
-- 如果只需要 name 字段
CREATE INDEX idx_users_email_name ON users(email, name);
-- 这样就不用回表查主键索引了
```

**面试讲述要点：**
- Explain 结果各列的含义（type、key、rows、extra）
- 覆盖索引是什么（Using index）
- 最左前缀原则

---

## 📦 缓存性能优化

### 反模式 1：缓存击穿 - 热点 Key 失效

**问题场景：**
- 商品详情缓存，过期时间 1 小时
- 爆款商品，每秒访问 10000 次
- 缓存过期瞬间，10000 个请求同时打穿到 DB！

**现象：**
- DB 流量突增 100 倍
- DB CPU 打满，超时增多
- 服务雪崩

**怎么发现：**
- 缓存命中率突降
- DB QPS 突增，出现大量相同 SQL
- 时间点和缓存过期时间吻合

**解决方案 1：热点 Key 永不过期 + 异步更新**
```go
// GOOD: 热点 key 不设置过期时间，后台异步刷新
func GetProduct(ctx context.Context, id int64) (*Product, error) {
    // 先查缓存 - 没有过期时间
    data, err := redis.Get(ctx, productKey(id)).Result()
    if err == nil {
        return decode(data), nil
    }
    
    // 缓存 miss，去 DB 查 + 写回
    product := loadFromDB(id)
    cacheProduct(id, product)
    
    // 同时加一个异步更新任务
    scheduleRefresh(id)
    
    return product, nil
}
```

**解决方案 2：singleflight - 合并重复请求**
```go
import "golang.org/x/sync/singleflight"

var sf singleflight.Group

func GetProduct(ctx context.Context, id int64) (*Product, error) {
    // 相同 id 的并发请求只会实际执行一次 DB 查询
    result, err, _ := sf.Do(strconv.Itoa(id), func() (interface{}, error) {
        // 只有一个 goroutine 会执行到这里
        product := loadFromDB(id)
        cacheProduct(id, product)
        return product, nil
    })
    
    return result.(*Product), err
}
```

**面试讲述要点：**
- 区分缓存穿透、缓存击穿、缓存雪崩三个概念
- singleflight 的原理和适用场景
- 热点 key 的特殊处理策略

---

### 反模式 2：缓存雪崩 - 大量 Key 同时过期

**问题场景：**
- 凌晨 2 点批量刷新缓存
- 所有 Key 都设了 24 小时过期
- 第二天凌晨 2 点，所有 Key 同时过期！

**现象：**
- DB 流量瞬间爆炸
- 服务大面积超时

**解决方案：**
```go
// GOOD: 过期时间加随机偏移
func cacheProduct(id int64, data *Product) {
    // 24 小时 + 0~1 小时随机
    expire := 24*time.Hour + time.Duration(rand.Intn(3600))*time.Second
    redis.SetEX(ctx, productKey(id), encode(data), expire)
}
```

---

### 反模式 3：大 Key - 单个 Key 几十 MB

**问题代码：**
```go
// BAD: 把全部商品列表都存在一个 key 里！
func CacheAllProducts(products []*Product) {
    // products 有 10 万个，JSON 序列化后 50MB！
    data, _ := json.Marshal(products)
    redis.Set(ctx, "all_products", data, time.Hour)
}

func GetAllProducts() []*Product {
    // 每次读都要拉 50MB！网络带宽扛不住！
    data, _ := redis.Get(ctx, "all_products").Result()
    // ...
}
```

**现象：**
- Redis 网卡跑满
- Get 操作超时
- 序列化/反序列化占大量 CPU

**怎么发现：**
- Redis bigkey 扫描
- network in/out 监控异常
- pprof 看 json.Marshal 占 CPU 高

**解决方案：**
1. 分片存储
```go
// 按 ID 分片，每个 key 存 100 个
for i := 0; i < 1000; i++ {
    chunk := products[i*100 : (i+1)*100]
    redis.Set(ctx, fmt.Sprintf("products:%d", i), encode(chunk), exp)
}
```

2. 用 Hash 结构，分字段存
3. 本地缓存 + Redis 二级缓存

**面试讲述要点：**
- Redis 大 Key 的危害：网络、CPU、内存碎片
- 怎么发现大 key：redis-cli --bigkeys
- 大 Key 拆分策略

---

## ⚡ gRPC / RPC 性能优化

### 反模式 1：同步调用链太长

**问题代码：**
```go
// BAD: 串行调用 N 个下游
func GetUserDetail(ctx context.Context, userID int64) (*UserDetail, error) {
    // 1. 查用户信息 - 50ms
    user, _ := userClient.Get(ctx, userID)
    
    // 2. 查订单信息 - 50ms
    orders, _ := orderClient.List(ctx, userID)
    
    // 3. 查优惠券 - 50ms
    coupons, _ := couponClient.List(ctx, userID)
    
    // 4. 查积分 - 50ms
    points, _ := pointClient.Get(ctx, userID)
    
    // 总共 200ms+！
    return &UserDetail{user, orders, coupons, points}, nil
}
```

**现象：**
- 接口延迟是所有下游延迟之和
- 只要一个下游慢，整个接口就慢

**修复方案：并发调用 + 超时控制**
```go
// GOOD: 并发调用所有下游
func GetUserDetail(ctx context.Context, userID int64) (*UserDetail, error) {
    var wg sync.WaitGroup
    var user *User
    var orders []*Order
    var coupons []*Coupon
    var points *Points
    
    // 设置超时 100ms，超过就返回
    ctx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
    defer cancel()
    
    // 用户
    wg.Add(1)
    go func() {
        defer wg.Done()
        user, _ = userClient.Get(ctx, userID)
    }()
    
    // 订单
    wg.Add(1)
    go func() {
        defer wg.Done()
        orders, _ = orderClient.List(ctx, userID)
    }()
    
    // ... 其他同理
    
    wg.Wait()
    
    // 部分失败降级处理（如果需要）
    return &UserDetail{user, orders, coupons, points}, nil
}
```

**优化后：延迟 = 最慢的那个下游（50ms），不是总和！**

---

### 反模式 2：连接池太小

**问题代码：**
```go
// BAD: 连接池默认配置太小
conn, _ := grpc.Dial(address, 
    grpc.WithInsecure(),
    // 没有配置连接池参数！默认值可能太小
)
```

**现象：**
- QPS 上不去，但下游很空闲
- trace 看大量时间花在等待连接上
- error log 里出现 "connection refused" 或 "context deadline exceeded"

**修复：**
```go
// GOOD: 合理配置连接池
conn, _ := grpc.Dial(address,
    grpc.WithInsecure(),
    
    // 每个连接最大并发流数
    grpc.WithMaxConcurrentStreams(100),
    
    // 连接超时
    grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
        return net.DialTimeout("tcp", addr, time.Second)
    }),
    
    // 保活配置
    grpc.WithKeepaliveParams(keepalive.ClientParameters{
        Time:                10 * time.Second,
        Timeout:             3 * time.Second,
        PermitWithoutStream: true,
    }),
)
```

---

### 反模式 3：每次都建立新连接

**问题代码：**
```go
// BAD: 每次调用都新建连接！
func CallDownstream(req *Request) (*Response, error) {
    // 每次都 dial！TCP 握手就要几 ms
    conn, _ := grpc.Dial(address, grpc.WithInsecure())
    defer conn.Close()
    
    client := NewServiceClient(conn)
    return client.Method(ctx, req)
}
```

**现象：**
- 大量 TIME_WAIT 连接
- 每次调用都有 TCP 握手开销
- 延迟高，QPS 上不去

**修复：**
```go
// GOOD: 全局单例连接，应用启动时初始化
var globalConn *grpc.ClientConn

func InitConn() {
    var err error
    globalConn, err = grpc.Dial(address, grpc.WithInsecure())
    // ...
}

func CallDownstream(req *Request) (*Response, error) {
    client := NewServiceClient(globalConn)  // 复用连接
    return client.Method(ctx, req)
}
```

**面试讲述要点：**
- gRPC 使用 HTTP/2，多路复用，一个连接可以跑多个请求
- 连接池的重要性，不要每次都建连
- 长连接的保活配置

---

## 📊 性能指标与监控 - 面试常考

### 四个黄金指标

面试官："上线一个新服务，你会重点监控哪些指标？"

**标准答案：**

1. **延迟 (Latency)**
   - P50 / P95 / P99（千万不要只看平均！）
   - 按接口维度细分

2. **流量 (Traffic)**
   - QPS / TPS
   - 按接口、按来源

3. **错误率 (Errors)**
   - 5xx / 4xx / 业务错误码
   - 错误率超过阈值告警

4. **饱和度 (Saturation)**
   - CPU / 内存 / 磁盘 / 网络
   - 连接池使用率
   - Goroutine 数量
   - GC 指标（频率、停顿时间）

**加分项：**
- 业务指标（订单创建成功率、支付成功率等）
- 下游依赖的四个黄金指标（DB、Redis、其他服务）

---

### Go 运行时关键指标

面试官："Go 服务你会重点监控哪些 runtime 指标？"

```go
// 可以用 prometheus 的 go collector 自动采集
import _ "github.com/prometheus/client_golang/prometheus/collectors/go"
```

**必看指标：**

| 指标 | 说明 | 异常阈值 |
|-----|------|---------|
| `go_goroutines` | goroutine 数量 | > 10000 或持续增长 |
| `go_info{version="xxx"}` | Go 版本 | - |
| `go_memstats_alloc_bytes` | 当前分配字节数 | 持续上涨 |
| `go_memstats_heap_objects` | 堆对象总数 | 持续上涨 |
| `go_gc_duration_seconds` | GC 停顿时间 | P99 > 100ms |
| `go_threads` | OS 线程数 | > 1000 |

---

## 💯 业务场景面试回答模板

### 题目："你们接口延迟一般多少？慢了怎么排查？"

**回答：**

> "我们核心接口 P99 一般控制在 100ms 以内，P999 在 200ms 左右。如果慢了我会按这个流程排查：
>
> **第一步：看监控，缩小范围**
> - 先看四个黄金指标：是所有接口都慢，还是某个接口慢？
> - CPU 高不高？内存高不高？有没有 GC 抖动？
> - 错误率有没有上升？
>
> **第二步：看链路追踪，定位具体慢在哪**
> - 找一个 trace example，看时间都花在哪一段
> - 是 DB 慢？Redis 慢？还是下游服务慢？
> - 还是业务逻辑本身慢？
>
> **第三步：针对性分析**
> - 如果 DB 慢：看慢日志，是不是全表扫描？是不是没走索引？是不是锁冲突？
> - 如果是 CPU 高：抓 pprof，看热点函数在哪里，是不是有循环、序列化、反射？
> - 如果是锁竞争：抓 mutex profile，看哪个锁等的时间最长，能不能缩小锁范围或分片？
> - 如果是 goroutine 泄漏：抓 goroutine profile，看大量 goroutine 卡在什么地方？
>
> **第四步：验证和复盘**
> - 优化后要有压测验证，对比优化前后的指标
> - 关键位置加上监控告警，下次提前发现
>
> 举个例子，上次我们商品详情接口 P99 从 80ms 涨到了 500ms，通过 trace 发现是调用库存服务慢了，
> 再看库存服务的监控，发现是加了个全表扫描的 SQL，加上索引就恢复了。"

---

### 题目："怎么保证高并发场景下服务不雪崩？"

**回答：**

> "我们主要从几个层面保证：
>
> **1. 超时 + 重试 + 熔断**
> - 所有 RPC 调用必须设超时，绝对不能无限等
> - 重试要有退避和次数限制，避免重试风暴
> - 下游出错率高了自动熔断，走降级逻辑
>
> **2. 限流 + 隔离**
> - 每个接口都有 QPS 限制，超过了直接拒绝，保护系统
> - 核心接口和非核心接口线程池/连接池隔离，避免互相影响
>
> **3. 多级缓存**
> - 热点数据本地缓存 + Redis 两级缓存
> - singleflight 合并相同请求，避免缓存击穿
>
> **4. 弹性伸缩**
> - CPU / 延迟超过阈值自动扩容
> - 提前压测，知道系统容量上限
>
> **5. 灰度发布 + 快速回滚**
> - 每次发布都灰度，有问题快速回滚
> - 发布前后对比核心指标
>
> 最重要的是：**要知道系统的极限在哪**，而不是出问题了才知道扛不住。
> 我们每次大促前都会做全链路压测，知道每个环节的瓶颈在哪，提前优化。"

---

## 📋 面试常见问题清单

- [ ] 说一个你排查过印象最深的性能问题？
- [ ] Go 的 GC 调优你有什么经验？
- [ ] sync.Pool 的原理是什么，适合什么场景，不适合什么场景？
- [ ] 怎么排查 goroutine 泄漏？
- [ ] Context 除了传值，还有什么用？
- [ ] 怎么设计接口的超时策略？
- [ ] 缓存穿透、击穿、雪崩的区别和解决方案？
- [ ] 怎么做服务限流？有哪些算法？
- [ ] 怎么定位锁竞争严重的地方？
- [ ] 怎么做全链路压测？

**记住：每个问题都最好能结合自己的实际经验回答，而不是背理论。**
