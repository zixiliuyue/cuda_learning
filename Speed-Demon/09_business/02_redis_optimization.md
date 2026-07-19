# Redis 性能优化最佳实践

> **面试必问**：缓存穿透/击穿/雪崩区别？Redis 性能问题遇到过吗？

---

## 🎯 Redis 性能问题三兄弟

| 问题 | 现象 | 根因 | 解决方案 |
|-----|------|------|---------|
| **缓存穿透** | 查一个一定不存在的数据 | 请求直接打到 DB | 布隆过滤器、空值缓存 |
| **缓存击穿** | 某一个热点 key 失效，大量请求同时打 DB | 热点 key 过期 | 互斥锁、永不过期、预热 |
| **缓存雪崩** | 大量 key 同时失效，DB 被打垮 | 大面积缓存同时过期 | 过期时间加随机值、多级缓存 |

---

## 🔫 缓存穿透

**问题：**

```go
// ❌ 黑客故意查一个一定不存在的 ID，缓存没有，每次都查 DB
func GetUser(id int) *User {
    // 1. 查缓存
    user := redis.Get(fmt.Sprintf("user:%d", id))
    if user != nil {
        return user
    }
    
    // 2. 缓存没有，查 DB
    user = db.QueryUser(id)
    
    // 3. 如果 DB 也没有，怎么办？❌ 下一次还查 DB！
    return user
}

// 黑客用不存在的 ID 狂打：id=-1, -2, -3...
// 每次都穿透到 DB，DB 被打死！
```

---

**解决方案 1: 空值缓存

```go
// ✅ DB 查不到也缓存个空值，下次就不用查 DB 了
func GetUser(id int) *User {
    key := fmt.Sprintf("user:%d", id)
    
    user := redis.Get(key)
    if user != nil {
        if user == "NULL" {  // 空值标记
            return nil
        }
        return user
    }
    
    user = db.QueryUser(id)
    if user == nil {
        // 缓存空值，短一点的时间短一点
        redis.SetEX(key, "NULL", 5*time.Minute)
        return nil
    }
    
    redis.SetEX(key, user, 1*time.Hour)
    return user
}
```

---

**解决方案 2: 布隆过滤器**

```go
import "github.com/bits-and-blooms/bloom/v3"

// 初始化布隆过滤器，把所有存在的 ID 放进去
var bf = bloom.NewWithEstimates(1000000, 0.01)  // 100 万元素，误判率 1%

// 启动时把所有用户 ID 全量加载进布隆过滤器
func initBloomFilter() {
    var allUserIDs []int
    db.Model(&User{}).Pluck("id", &allUserIDs)
    for _, id := range allUserIDs {
        bf.Add([]byte(strconv.Itoa(id)))
    }
}

// ✅ 查询前先过布隆过滤器
func GetUser(id int) *User {
    // 布隆过滤器说不存在，那一定不存在！直接返回
    if !bf.Test([]byte(strconv.Itoa(id)))) {
        return nil
    }
    
    // 布隆过滤器说可能存在，才去查缓存/DB
    // ... 后面逻辑同上
}
```

| 方案 | 优点 | 缺点 |
|-----|------|------|
| 空值缓存 | 简单 | 占点内存 | 能防不住大量不存在的 key | 内存爆炸 |
| 布隆过滤器 | 省内存 | 有 1% 误判率，需要维护 |

---

## 💥 缓存击穿

**问题：** 某一个超级热点 key 突然失效的瞬间，1000 个并发同时打 DB！

```go
// ❌ 热点商品详情，1 万 QPS，缓存 1 小时过期
func GetProduct(id int) *Product {
    key := fmt.Sprintf("product:%d", id)
    
    product := redis.Get(key)
    if product != nil {
        return product
    }
    
    // 缓存失效瞬间，1000 个 goroutine 同时到这里！
    // 同时查 DB，DB 直接被打挂！
    product = db.QueryProduct(id)
    redis.SetEX(key, product, 1*time.Hour)
    return product
}
```

---

**解决方案 1: 互斥锁（单飞）**

```go
// ✅ 只有一个 goroutine 去查 DB，其他人等结果
func GetProduct(id int) *Product {
    key := fmt.Sprintf("product:%d", id)
    
    product := redis.Get(key)
    if product != nil {
        return product
    }
    
    // 抢锁，只有抢到锁的人才能查 DB
    lockKey := fmt.Sprintf("lock:product:%d", id)
    if redis.SetNX(lockKey, 1, 10*time.Second) {  // 10 秒超时防止死锁
        defer redis.Del(lockKey)
        
        // 拿到锁，查 DB 写缓存
        product = db.QueryProduct(id)
        if product != nil {
            redis.SetEX(key, product, 1*time.Hour)
        }
        return product
    }
    
    // 没抢到锁，等 50ms 再查缓存
    time.Sleep(50 * time.Millisecond)
    return GetProduct(id)  // 递归重试
}
```

---

**解决方案 2: 永不过期 + 异步更新**

```go
// ✅ 物理不过期，逻辑过期
func GetProduct(id int) *Product {
    // value = 数据 + 逻辑过期时间
    // { data: xxx, expire_at: 1234567890 }
    
    key := fmt.Sprintf("product:%d", id)
    value := redis.Get(key)
    
    // 缓存永远有值，直接返回
    product := parseProduct(value)
    
    // 发现快过期了，后台异步去更新
    if time.Now().Unix() > value.ExpireAt {
        if redis.SetNX(fmt.Sprintf("lock:refresh:%d", id), 1, 10*time.Second) {
            go func() {  // 异步更新，不阻塞用户请求
                newProduct := db.QueryProduct(id)
                redis.Set(key, buildValue(newProduct, time.Now().Add(1*time.Hour).Unix()))
                redis.Del(fmt.Sprintf("lock:refresh:%d", id))
            }()
        }
    }
    
    return product  // 直接返回旧值，用户完全无感知！
}
```

---

## ❄️ 缓存雪崩

**问题：** 大量 key 同时过期，比如缓存时间都是 1 小时，整点同时失效，DB 被打垮！

```go
// ❌ 所有 key 都是 1 小时过期
for _, product := range products {
    redis.SetEX(fmt.Sprintf("product:%d", product.ID), product, 1*time.Hour)
    // 全都是整点过期！
}
```

**解决方案：过期时间加随机值**

```go
// ✅ 每个 key 的过期时间加个随机抖动
func getExpireTime() time.Duration {
    base := 1 * time.Hour
    // 加 ±10 分钟随机值
    jitter := time.Duration(rand.Intn(1200)-600) * time.Second
    return base + jitter
}

// 每个 key 的过期时间分散在 50~70 分钟之间，不会同时失效
```

---

## 🚀 Redis 性能优化技巧

### 技巧 1: Pipeline 批量操作

```go
// ❌ 循环 Get，每次网络往返
func badMultiGet(keys []string) []interface{} {
    results := make([]interface{}, len(keys))
    for i, key := range keys {
        results[i] = redis.Get(key)  // N 次网络往返！
    }
    return results
}

// ✅ MGet 一次搞定
func goodMultiGet(keys []string) []interface{} {
    return redis.MGet(keys...).Val()  // 1 次网络往返！
}

// ✅ Pipeline 批量混合操作
func pipelineExample() {
    pipe := redis.Pipeline()
    for i := 0; i < 100; i++ {
        pipe.Set(fmt.Sprintf("key:%d", i), i, 1*time.Hour)
        pipe.Get(fmt.Sprintf("key:%d", i))
    }
    // 一次发送 200 个命令，只需要 1 次网络往返！
    cmders, _ := pipe.Exec()
}
```

**性能提升：** 100 次操作，Pipeline 比循环快 **50-100 倍**！

---

### 技巧 2: 选择合适的数据结构

| 场景 | 选什么 | 为什么 |
|-----|---------|-------|
| 简单 KV | String | 最简单 |
| 粉丝/关注列表 | Set | 去重、交集并集差集 |
| 排行榜 | ZSet | 天然有序 |
| 购物车 | Hash | 方便增减字段 |
| 消息队列 | List | LPUSH + RPOP |
| 限流 | ZSet / String + Lua | 滑动窗口 |

**错误用法示例：**

```go
// ❌ 用 String 存 JSON 列表，每次都要反序列化
redis.Set("users", `[{"id":1}, {"id":2}]', 1*time.Hour)
// 每次读：json.Unmarshal 一下，加一个元素要全量读改了再写回去！

// ✅ 用 List / Hash
redis.LPush("users", `{"id":1}`)
redis.LPush("users", `{"id":2}`)
// 直接操作单个元素！
```

---

### 技巧 3: Big Key 拆分

**什么是 Big Key？**
- String 类型 value 超过 10KB
- List/Hash/Set/ZSet 元素超过 5000 个

**Big Key 的危害：**
- 网络 IO 大，读一个 Big Key 慢
- 内存碎片多
- 删除时阻塞整个 Redis

**拆分方法：**

```go
// ❌ 一个 Hash 存了 10 万个用户信息
redis.HSet("user_profile", "1", "...")  // 10 万个 field

// ✅ 按用户 ID 哈希拆分到多个 key
func getUserProfileKey(userID int) string {
    return fmt.Sprintf("user_profile:%d", userID%100)
}

func SetUserProfile(userID int, profile string) {
    key := getUserProfileKey(userID)
    redis.HSet(key, strconv.Itoa(userID), profile)
}
// 拆成 100 个 key，每个 key 1000 个 field
```

---

### 技巧 4: Lua 脚本保证原子性

```go
// ❌ 先 Get 再 Set，中间可能被其他请求修改
func badIncr(key string) int {
    val := redis.Get(key).Int()
    val++
    redis.Set(key, val)
    return val
    // 并发下会丢计数丢失！
}

// ✅ 用 Lua 脚本，原子执行
var incrScript = redis.NewScript(`
    local val = tonumber(redis.call('GET', KEYS[1]) or 0
    val = val + 1
    redis.call('SET', KEYS[1], val)
    return val
`)

func goodIncr(key string) int {
    return incrScript.Run(redis, []string{key}).Int()
}
```

---

## ⚠️ 常见 Redis 坑

### 坑 1: 热 Key 问题

**问题：** 某个热点 key 流量太大，单个 Redis 实例网卡打满！

```go
// 双 11 爆款商品，每秒 10 万次查询同一个 key
redis.Get("product:12345")
// 单 Redis 实例网卡 10Gbps 打满！
```

**解决方案：** 本地缓存 + 热点 Key 复制

```go
// ✅ 1. 本地缓存（进程内缓存）
var localCache = cache.New(5*time.Minute, 10*time.Minute)

func GetProduct(id int) *Product {
    // 先查本地缓存，命中直接返回，根本不打 Redis！
    if val, ok := localCache.Get(fmt.Sprintf("product:%d", id)); ok {
        return val.(*Product)
    }
    
    // 再查 Redis
    product := redis.Get(...)
    localCache.Set(fmt.Sprintf("product:%d", id), product, 1*time.Minute)
    return product
}

// ✅ 2. 热点 Key 复制多份
// product:12345_1, product:12345_2, ... product:12345_10
// 读的时候随机选一个，流量分散到 10 个 key
```

---

### 坑 2: 缓存和数据库一致性

**先更 DB 还是先删缓存？**

```go
// ❌ 先删缓存，再更 DB
// 线程 1：删缓存
// 线程 2：读，缓存没了，读 DB 旧值，写回缓存
// 线程 1：更新 DB
// 结果：缓存里是旧值！不一致！

// ✅ 先更 DB，再删缓存（推荐）
func UpdateUser(id int, newData User) {
    // 1. 先更数据库
    db.UpdateUser(id, newData)
    
    // 2. 再删缓存
    redis.Del(fmt.Sprintf("user:%d", id))
}

// 延时双删（极端场景）
func UpdateUser(id int, newData User) {
    redis.Del(fmt.Sprintf("user:%d", id))
    db.UpdateUser(id, newData)
    time.Sleep(1 * time.Second)  // 等读请求写完旧值再删一次
    redis.Del(fmt.Sprintf("user:%d", id))
}
```

---

### 坑 3: 内存淘汰策略

```go
// ❌ 默认 volatile-lru：只淘汰设置了过期时间的 key
// 没过期的 key 永远不会被淘汰，内存满了新的写不进去！

// ✅ 推荐 allkeys-lru：所有 key 都按 LRU 淘汰
// redis.conf
maxmemory-policy allkeys-lru
```

| 策略 | 适用场景 |
|-----|---------|
| allkeys-lru | 大部分是热点数据，推荐 |
| allkeys-random | 访问概率差不多 |
| volatile-lru | 有些数据需要永久保留 |
| noeviction | 写满了报错，不推荐 |

---

## 💯 面试回答模板

**Q: 缓存穿透、击穿、雪崩的区别和解决方案？**

> "这三个是 Redis 最经典的三个问题，区别是攻击范围不一样：
>
> **缓存穿透：** 查的是数据库里一定不存在的数据，
> 每次都穿透到数据库，比如用 -1 这种不存在的 ID 狂打。
> 解决方案：空值缓存，或者布隆过滤器先过滤掉肯定不存在的 ID。
>
> **缓存击穿：** 某一个超级热点 key 突然失效的瞬间，
> 大量并发同时打数据库，比如双 11 爆款商品缓存过期。
> 解决方案：互斥锁，只有一个 goroutine 去查 DB 写缓存，
> 或者永不过期，逻辑过期，后台异步更新。
>
> **缓存雪崩：** 大量 key 同时过期，数据库压力骤增，
> 比如所有 key 的过期时间都是 1 小时，整点同时失效。
> 解决方案：过期时间加随机抖动，分散过期时间。
>
> **一句话总结区别：**
> 穿透是查不存在的，击穿是单个热点 key 过期，
> 雪崩是大量 key 同时过期。
>
> **我的经验：** 90% 的场景，空值缓存防穿透，
> 互斥锁防击穿，过期加随机值防雪崩，就够了！"

---

## ✅ Redis 性能优化 Checklist

- [ ] 有没有缓存穿透问题？空值缓存/布隆过滤器加了吗？
- [ ] 有没有热点 key 击穿问题？互斥锁/永不过期加了吗？
- [ ] 大量 key 同时过期吗？过期时间加随机值了吗？
- [ ] 能批量操作吗？MGet / Pipeline 用了吗？
- [ ] 数据结构选对了吗？Big Key 拆分了吗？
- [ ] 热 Key 问题考虑了吗？本地缓存了吗？
- [ ] 缓存和数据库一致性策略对吗？
- [ ] 内存淘汰策略是 allkeys-lru 吗？

---

## 🚀 动手练习

1. 模拟缓存穿透场景，分别用空值缓存和布隆过滤器解决，对比效果
2. 模拟缓存击穿场景，用互斥锁和永不过期两种方案对比
3. 用 Pipeline 批量写入 1000 个 key，和循环写入对比性能差多少

**思考题：** 为什么推荐先更数据库再删缓存，而不是先删缓存再更数据库？
