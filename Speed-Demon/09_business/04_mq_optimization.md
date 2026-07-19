# 消息队列性能优化最佳实践

> **面试必问**：消息积压怎么处理？消息丢失怎么办？

---

## 📦 消息队列核心问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| **消息丢失** | 发送/消费/持久化各环节可能丢 | 确认机制、持久化、重试 |
| **消息积压** | 消费速度跟不上生产速度 | 扩容、批量、优化消费逻辑 |
| **重复消费** | 网络重传、重试 | 幂等性 |
| **顺序性** | 分区/多线程消费乱序 | 单分区、有序消费 |

---

## 🚨 消息积压处理

### 第一步：先止血！

```go
// ❌ 消费者还在慢慢消费，积压越来越多
// 先把消费者停了，防止雪崩

// ✅ 紧急处理步骤：
// 1. 扩容消费者数量（前提是下游能扛住）
// 2. 临时把消息转存到新的 Topic，新消息正常消费
// 3. 后台慢慢消费积压的消息
```

---

### 第二步：定位根因

**常见积压原因：**

| 原因 | 现象 | 解决 |
|-----|------|------|
| 消费者挂了 | 消费组 lag 持续上涨 | 重启消费者 |
| 消费逻辑变慢 | 消费耗时突然从 10ms 变 1s | 优化消费逻辑、加缓存 |
| 生产突然放量 | 大促、活动、爬虫 | 扩容、限流 |
| 下游 DB 挂了 | 所有消费都超时重试 | 熔断降级、先入库 |

---

### 第三步：加速消费

```go
// ❌ 单条消费，慢死
func badConsumer(msg Message) error {
    return process(msg)  // 一条一条处理
}

// ✅ 批量消费，均摊 IO 开销
func goodConsumer(msgs []Message) error {
    // 攒 100 条批量处理
    // 比如批量入库、批量调用 RPC
    return batchProcess(msgs)  // 比单条快 10-100 倍！
}

// ✅ 并发消费
func concurrentConsumer(msgs []Message) error {
    var wg sync.WaitGroup
    for _, msg := range msgs {
        wg.Add(1)
        go func(m Message) {
            defer wg.Done()
            process(m)
        }(msg)
    }
    wg.Wait()
    return nil
}
```

---

## 🔒 消息不丢失保障

### 生产者端可靠性

```go
// ❌ 发出去就不管了，网络丢了也不知道
func badProducer(msg Message) {
    producer.Send(msg)  // 火鸡送！
}

// ✅ 同步发送 + 重试
func goodProducer(ctx context.Context, msg Message) error {
    var err error
    for i := 0; i < 3; i++ {
        err = producer.SendSync(ctx, msg)
        if err == nil {
            return nil
        }
        time.Sleep(100 * time.Millisecond)
    }
    // 还是失败，记录日志 + 告警
    log.Printf("send message failed: %v", err)
    return err
}

// ✅ 异步发送 + 回调确认
func asyncProducer(msg Message) {
    producer.SendAsync(msg, func(result Result, err error) {
        if err != nil {
            // 发送失败，记录下来，后续重试
            failedMessages.Push(msg)
        }
    })
}
```

---

### Broker 端可靠性

```
✅ 持久化：消息写入磁盘，不是只在内存
✅ 多副本：至少 3 副本，一个挂了还有其他
✅ 刷盘策略：至少异步刷盘，重要消息同步刷盘
✅ ack 机制：所有副本都收到才确认成功
```

---

### 消费者端可靠性

```go
// ❌ 收到消息就 ack，处理失败消息丢了
func badConsumer(msg Message) error {
    ack(msg)  // 先确认！
    
    err := process(msg)  // 这里 panic 消息就丢了！
    return err
}

// ✅ 处理成功才 ack
func goodConsumer(msg Message) error {
    err := process(msg)  // 先处理
    if err != nil {
        // 处理失败，nack，让消息重新入队重试
        nack(msg)
        return err
    }
    
    ack(msg)  // 处理成功才确认！
    return nil
}
```

---

## 🆔 幂等性：重复消费不可怕

**为什么一定会重复消费？**
- 网络抖动，ack 没发到 broker
- 消费者重启，最后一批消息重新消费
- 重试机制本身就是重复

---

### 常用幂等方案

| 方案 | 适用场景 | 优点 | 缺点 |
|-----|---------|------|------|
| **唯一键去重表** | 数据库操作 | 简单，强一致 | 依赖 DB |
| **Redis 去重** | 通用场景 | 快，性能好 | 有过期时间 |
| **版本号乐观锁** | 更新操作 | 无额外开销 | 只能用于更新 |
| **天然幂等** | 查询、删除 | 最好！ | 不是所有场景都有 |

---

### Redis 幂等实现

```go
// ✅ 用消息 ID 做去重
func isProcessed(msgID string) bool {
    // SETNX：如果 key 不存在才设置成功
    // 设置成功说明没处理过，返回 false
    // 设置失败说明已经处理过了，返回 true
    key := fmt.Sprintf("msg_processed:%s", msgID)
    
    // 设置 1 小时过期，防止内存爆炸
    success, _ := redis.SetNX(key, "1", 1*time.Hour).Result()
    return !success
}

func consumer(msg Message) error {
    if isProcessed(msg.ID) {
        return nil  // 已经处理过了，直接跳过
    }
    
    return process(msg)
}
```

---

### 数据库唯一键幂等

```go
// ✅ 用业务唯一键做幂等
func createOrder(ctx context.Context, order Order) error {
    // 订单号是唯一键
    _, err := db.ExecContext(ctx,
        "INSERT INTO orders (order_no, user_id, amount) VALUES (?, ?, ?)",
        order.OrderNo, order.UserID, order.Amount,
    )
    
    if err != nil {
        if mysqlErr, ok := err.(*mysql.MySQLError); ok {
            if mysqlErr.Number == 1062 {  // 重复键错误
                return nil  // 已经创建过了，直接返回成功
            }
        }
        return err
    }
    return nil
}
```

---

## 📊 顺序消息

### 什么时候需要顺序？

✅ 需要顺序的场景：
- 订单状态流转：创建 → 支付 → 发货 → 完成
- 日志流水：按时间顺序
- Binlog 同步：必须按顺序

❌ 不需要顺序的场景（大部分）：
- 发送短信、邮件
- 统计计数
- 异步任务

---

### 怎么保证顺序？

```go
// ✅ 方案 1：同一个业务 ID 的消息发到同一个分区
func getPartitionKey(msg Message) string {
    // 订单消息按订单号哈希，同一个订单的消息永远在同一个分区
    return msg.OrderNo
}

// 同一个分区的消息，消费时自然是按顺序的

// ✅ 方案 2：单线程消费一个分区
// 一个分区只给一个消费者，保证顺序
// 缺点：并发度受分区数限制
```

---

## ⚠️ 常见坑

### 坑 1: 重试无限循环

```go
// ❌ 处理失败就无限重试，消息永远堵在这里
func badConsumer(msg Message) error {
    for {
        err := process(msg)
        if err == nil {
            return nil
        }
        // 永远重试！如果是脏数据，永远卡在这里！
        time.Sleep(1 * time.Second)
    }
}

// ✅ 最多重试 N 次，超过就进死信队列
func goodConsumer(msg Message) error {
    retryCount := getRetryCount(msg)
    if retryCount >= 3 {
        // 超过重试次数，进死信队列，人工处理
        sendToDLQ(msg)
        ack(msg)
        return nil
    }
    
    err := process(msg)
    if err != nil {
        incrRetryCount(msg)
        nack(msg)  // 重新入队
        return err
    }
    
    ack(msg)
    return nil
}
```

---

### 坑 2: 消息体太大

```go
// ❌ 直接把 10MB 的图片放消息体里
// 网络传输慢，broker 内存压力大，消费也慢

// ✅ 消息体只放 ID，真正的数据存在 Redis/OSS
msg := Message{
    Type: "image_process",
    ImageID: "xxx",  // 只放 ID
    // 图片存在 OSS，消费者去取
}
```

---

## 💯 面试回答模板

**Q: 消息队列积压了几百万条消息，怎么处理？**

> "我会按这几步处理：
>
> **第一步：先止血，防止情况恶化**
> 先看是不是消费者挂了，能重启就先重启。
> 如果消费逻辑有问题，先临时降级，比如跳过非核心处理，
> 先把积压消下来再说。
>
> **第二步：定位根因**
> 是生产者突然放量了，还是消费者变慢了？
> 看监控：QPS 涨了多少，消费耗时涨了多少，下游 DB/Redis 有没有问题。
>
> **第三步：加速消费**
> 第一，扩容消费者数量，前提是下游能扛住。
> 第二，改批量消费，把单条处理改成批量处理，均摊 IO 开销。
> 第三，能并发消费的就并发，不要串行。
>
> **第四步：善后处理**
> 积压消完之后，复盘为什么会积压：
> 是监控没告警吗？还是消费速度没达标？
> 要不要加消息积压告警？要不要做限流保护？
>
> **极端情况：**
> 如果积压太多，几天都消费不完，可以考虑
> 把新消息切到新 Topic，旧消息后台慢慢消费，
> 先保证新消息不受影响。
>
> **记住：** 先恢复服务，再查问题根因！"

---

**Q: 怎么保证消息不丢失？**

> "要从三个环节都保证：
>
> **生产者端：**
> 用同步发送或者带回调的异步发送，
> 发送失败要重试，重试几次还失败要记录日志告警，
> 不能发出去就不管了。
>
> **Broker 端：**
> 要开持久化，消息写入磁盘，不能只在内存。
> 多副本，至少 3 副本，一个挂了还有其他副本。
> ack 机制要开 all，所有副本都收到才确认成功。
>
> **消费者端：**
> 最重要，必须处理完消息才 ack，不能收到就 ack。
> 处理失败要 nack，让消息重新入队重试。
>
> **最后：**
> 没有 100% 不丢的消息，只能做到 99.999%，
> 所以重要的消息一定要有对账机制，
> 每天跑批核对生产和消费的数量差。"

---

## ✅ MQ 优化 Checklist

- [ ] 生产者发送有重试机制吗？
- [ ] 消费者处理成功才 ack 吗？
- [ ] 幂等性做了吗？重复消费有问题吗？
- [ ] 重试有次数限制吗？有死信队列吗？
- [ ] 能批量消费吗？有批量接口吗？
- [ ] 消息体太大了吗？只放必要的字段吗？
- [ ] 有消息积压监控告警吗？
- [ ] 顺序消息真的需要顺序吗？能不做顺序吗？
- [ ] 有对账机制吗？能发现丢失的消息吗？

---

## 🚀 动手练习

1. 写一个带重试和死信队列的消费者
2. 用 Redis 实现一个简单的幂等去重
3. 模拟消息积压场景，看批量消费 vs 单条消费的速度差

**思考题：** 顺序消息和高并发为什么是矛盾的？怎么平衡？
