# 数据库性能优化最佳实践

> **面试必问**：慢查询怎么优化？遇到过哪些数据库性能问题？

---

## 🎯 数据库优化金字塔

```
┌─────────────────────────────────────────────────┐
│  架构层优化（最大收益）                          │
│  • 读写分离、分库分表、热点数据隔离            │
├─────────────────────────────────────────────────┤
│  应用层优化（第二大收益）                        │
│  • 减少 N+1 查询、批量操作、连接池优化          │
├─────────────────────────────────────────────────┤
│  SQL 层优化（第三大收益）                        │
│  • 索引优化、避免全表扫描、优化执行计划          │
├─────────────────────────────────────────────────┤
│  配置层优化（锦上添花）                          │
│  • 缓存大小、连接数、超时配置                    │
└─────────────────────────────────────────────────┘
```

---

## 🔍 第一步：先找到慢查询

### 开启慢查询日志

```sql
-- MySQL 开启慢查询
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;  -- 超过 1 秒就算慢查询
SET GLOBAL log_queries_not_using_indexes = 'ON';  -- 记录没走索引的查询
```

### Go 应用层监控慢查询

```go
// 用 GORM 回调，自动记录慢查询
import (
    "gorm.io/gorm/logger"
    "time"
)

func initDB() *gorm.DB {
    db, _ := gorm.Open(mysql.Open(dsn), &gorm.Config{
        Logger: logger.New(
            log.New(os.Stdout, "\r\n", log.LstdFlags),
            logger.Config{
                SlowThreshold:             200 * time.Millisecond,  // 超过 200ms 就算慢查询
                LogLevel:                  logger.Warn,
                Colorful:                  false,
            },
        ),
    })
    return db
}
```

---

## 🚀 应用层优化

### 优化 1: 消灭 N+1 查询

**问题场景：**

```go
// ❌ N+1 查询：先查 1 次列表，再循环查每个详情
func badGetUsers() []User {
    var users []User
    db.Find(&users)  // 第 1 次查询
    
    for i := range users {
        // 循环 N 次查询！N=100 就是 101 次查询！
        db.First(&users[i].Profile, "user_id = ?", users[i].ID)
    }
    return users
}
```

**性能对比：**

| N | N+1 次数 | JOIN 次数 | 性能差 |
|---|---------|----------|-------|
| 10 | 11 次 | 1 次 | 10 倍 |
| 100 | 101 次 | 1 次 | 100 倍 |
| 1000 | 1001 次 | 1 次 | 1000 倍 |

**修复方案：**

```go
// ✅ 方案 1: JOIN 查询，一次搞定
func goodGetUsers1() []User {
    var users []User
    db.Joins("Profile").Find(&users)  // 1 次查询搞定！
    return users
}

// ✅ 方案 2: IN 查询，两次搞定（适合数据量大）
func goodGetUsers2() []User {
    var users []User
    db.Find(&users)
    
    // 收集所有 user_id，一次查所有 profile
    userIDs := make([]int, len(users))
    for i := range users {
        userIDs[i] = users[i].ID
    }
    
    var profiles []Profile
    db.Find(&profiles, "user_id IN ?", userIDs)  // IN 查询
    
    // 内存里拼起来
    profileMap := make(map[int]*Profile)
    for i := range profiles {
        profileMap[profiles[i].UserID] = &profiles[i]
    }
    for i := range users {
        users[i].Profile = profileMap[users[i].ID]
    }
    
    return users
}
```

---

### 优化 2: 批量插入/更新

```go
// ❌ 循环插入：1000 条就是 1000 次网络往返
func badInsert(users []User) {
    for _, u := range users {
        db.Create(&u)  // 每次都要网络往返！
    }
}

// ✅ 批量插入：一次搞定
func goodInsert(users []User) {
    db.Create(&users)  // GORM 自动批量插入
    // INSERT INTO users VALUES (), (), ()...  一条 SQL 搞定
}

// ✅ 分批次批量（数据太大避免 SQL 过长）
func batchInsert(users []User) {
    batchSize := 100
    for i := 0; i < len(users); i += batchSize {
        end := i + batchSize
        if end > len(users) {
            end = len(users)
        }
        db.Create(users[i:end])
    }
}
```

**性能提升：** 1000 条数据，批量比循环快 **50-100 倍**！

---

### 优化 3: 连接池配置（90% 的人配错）

```go
// ❌ 默认配置：最大 100 连接，高并发下不够用
db, _ := sql.Open("mysql", dsn)

// ✅ 正确配置（根据实际情况调）
db.SetMaxOpenConns(100)     // 最大打开连接数
db.SetMaxIdleConns(50)      // 最大空闲连接数
db.SetConnMaxLifetime(time.Hour)  // 连接最大存活时间
db.SetConnMaxIdleTime(10 * time.Minute)  // 最大空闲时间
```

**怎么算连接数？**

```
理想连接数 ≈ CPU 核心数 × 2 + 磁盘 spindle 数
```

**太多连接的坏处：**
- 每个连接占内存（约 2M），1000 个连接就 2G 没了
- 太多连接导致上下文切换增多，反而变慢

**太少连接的坏处：**
- 高并发下都在等连接，排队时间长

---

## 🔑 SQL 层优化：索引优化

### 怎么看索引走没走？

```sql
-- EXPLAIN 看执行计划
EXPLAIN SELECT * FROM users WHERE email = 'xxx@xx.com';
```

看输出里的 `type` 列：

| type | 含义 | 性能 |
|-----|------|-----|
| system | 系统表，只有一行 | 最好 |
| const | 主键/唯一索引匹配一行 | 很好 |
| eq_ref | 联表时用主键/唯一索引 | 好 |
| ref | 非唯一索引等值匹配 | 可以 |
| range | 索引范围查询 | 一般 |
| index | 扫描整个索引树 | 慢 |
| ALL | 全表扫描！ | 最慢 ❌ |

**看到 `type=ALL` 必须优化！**

---

### 索引最佳实践

✅ **1. 最左前缀原则**
```
索引 (a, b, c)
能用的查询：
  a = ?
  a = ? AND b = ?
  a = ? AND b = ? AND c = ?

不能用的查询：
  b = ?
  c = ?
  a = ? AND c = ? （只能用到 a）
```

✅ **2. 小表驱动大表**
```sql
-- 好：小表 100 行驱动大表 100000 行
SELECT * FROM small s JOIN big b ON s.id = b.id
-- 只循环 100 次

-- 不好：大表驱动小表
SELECT * FROM big b JOIN small s ON b.id = s.id
-- 循环 100000 次
```

✅ **3. 索引列不要用函数/运算**
```sql
-- ❌ 索引用不上！
WHERE DATE(created_at) = '2024-01-01'

-- ✅ 改成范围查询
WHERE created_at >= '2024-01-01' AND created_at < '2024-01-02'
```

✅ **4. LIKE '%xxx' 前缀不能用索引**
```sql
-- ❌ 全模糊，索引用不上
WHERE name LIKE '%张三%'

-- ✅ 前缀模糊，可以用索引
WHERE name LIKE '张三%'

-- 真的需要全文搜索？用 Elasticsearch！
```

❌ **5. 不要建太多索引**
- 索引占磁盘空间
- 写入（INSERT/UPDATE/DELETE）时要更新所有相关索引，变慢
- 一般单表索引不超过 5 个

---

### 覆盖索引：索引就是数据，不用回表

```sql
-- 普通索引：先查索引树，再回表查整行数据
SELECT email FROM users WHERE name = '张三'
-- 1. 查 name 索引树，找到主键 id
-- 2. 用 id 查主键索引树，拿到 email

-- ✅ 覆盖索引：索引里就有要查的字段，不用回表！
CREATE INDEX idx_name_email ON users(name, email)

-- 再查：
SELECT email FROM users WHERE name = '张三'
-- 直接从索引树里拿到 email，不用回表！性能提升 2-10 倍！
```

---

## 🏗️ 架构层优化

### 1. 读写分离

```go
// 主库：写
// 从库：读（多个从库分摊读压力）

type DB struct {
    master *gorm.DB
    slaves []*gorm.DB
}

func (db *DB) GetSlave() *gorm.DB {
    // 轮询选一个从库
    return db.slaves[rand.Intn(len(db.slaves))]
}

// 写操作用主库
func (db *DB) CreateUser(u *User) {
    db.master.Create(u)
}

// 读操作用从库
func (db *DB) GetUser(id int) *User {
    var u User
    db.GetSlave().First(&u, id)
    return &u
}
```

---

### 2. 分库分表（水平拆分）

**什么时候需要分表？**
- 单表超过 500 万行，或者 2G 以上
- 写入 QPS 超过 1000

**分表策略：**

| 策略 | 适用场景 | 优点 | 缺点 |
|-----|---------|------|-----|
| RANGE 范围 | 按时间、ID 范围 | 查询方便，容易扩容 | 容易热点 |
| HASH 哈希 | 用户 ID、订单 ID 取模 | 数据均匀 | 范围查麻烦 |
| LIST 列表 | 按地区、类型 | 灵活 | 需要维护映射 |

```go
// 按用户 ID 哈希分表示例
func getUserTable(userID int) string {
    tableID := userID % 100  // 分 100 张表
    return fmt.Sprintf("users_%02d", tableID)
}

func GetUser(userID int) *User {
    table := getUserTable(userID)
    var u User
    db.Table(table).First(&u, userID)
    return &u
}
```

---

## ⚠️ 常见 SQL 坑

### 坑 1: SELECT * 查所有字段

```go
// ❌ 查不需要的字段，浪费 IO 和内存
db.First(&user, id)  // SELECT * FROM users

// ✅ 只查需要的字段
db.Select("id, name, email").First(&user, id)
```

---

### 坑 2: COUNT(*) 太多

```go
// ❌ 全表扫，大表慢死
db.Model(&User{}).Count(&count)

// ✅ 方案 1: 缓存总数（不需要精确的场景）
count, _ := redis.Get("user_count").Int()

// ✅ 方案 2: 估算（MySQL）
SELECT table_rows FROM information_schema.tables
WHERE table_name = 'users'
```

---

### 坑 3: OFFSET 太深

```go
// ❌ 翻到第 1000 页，OFFSET 99900，要跳过 99900 行！
SELECT * FROM users LIMIT 100 OFFSET 99900

// ✅ 用 ID 定位，书签式分页
SELECT * FROM users WHERE id > last_id LIMIT 100
```

---

## 💯 面试回答模板

**Q: 遇到慢查询怎么优化？**

> "我一般按这个流程优化：
>
> **第一步：先定位是哪条 SQL 慢**
> 开启慢查询日志，或者在应用层监控超过 200ms 的查询，
> 先把慢 SQL 捞出来。
>
> **第二步：EXPLAIN 看执行计划**
> 看 `type` 是不是 `ALL` 全表扫描？
> 看 `rows` 是不是扫描行数太多？
> 看 `Extra` 有没有 `Using filesort` / `Using temporary`？
>
> **第三步：应用层优化（最优先）**
> 先看看是不是有 N+1 查询？能不能改成 JOIN 或者 IN 查询？
> 能不能批量操作减少网络往返？
> 连接池配置合理吗？会不会都在等连接？
>
> **第四步：索引优化**
> 看 where 条件的字段有没有索引？
> 符合最左前缀原则吗？
> 能不能建覆盖索引避免回表？
>
> **第五步：SQL 本身优化**
> 不要 SELECT *，只查需要的字段？
> 深分页能不能改成书签式分页？
> COUNT(*) 能不能缓存或者估算？
>
> **最后：架构层优化**
> 读写分离？分库分表？
> 热点数据能不能放 Redis 缓存？
>
> **记住：** 90% 的慢查询问题，
> 都能通过「消灭 N+1」和「加索引」解决！"

---

## ✅ 数据库优化 Checklist

- [ ] 开启慢查询日志了吗？
- [ ] 有 N+1 查询吗？循环里有没有查数据库？
- [ ] 能批量操作吗？不要循环插入/更新
- [ ] EXPLAIN 看了吗？type 是 ALL 吗？
- [ ] 索引符合最左前缀吗？
- [ ] 索引列上用了函数/运算吗？
- [ ] 连接池配置合理吗？
- [ ] SELECT * 了吗？只查需要的字段了吗？
- [ ] 深分页优化了吗？
- [ ] 热点数据加缓存了吗？

---

## 🚀 动手练习

1. 写一个 N+1 查询的例子，然后优化成 JOIN 和 IN 查询，对比性能
2. 用 EXPLAIN 分析一条慢查询，加索引看看性能提升多少
3. 配置合理的数据库连接池，压测看不同配置下的 QPS 差异

**思考题：** 为什么索引能加快查询？为什么不把所有字段都建索引？
