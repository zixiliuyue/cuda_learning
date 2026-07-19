package main

import (
	"errors"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ==============================================
// 模拟消息队列
// ==============================================

type Message struct {
	ID      string
	Content string
}

// 简单的内存 MQ
type MemoryMQ struct {
	messages chan Message
}

func NewMemoryMQ(size int) *MemoryMQ {
	return &MemoryMQ{
		messages: make(chan Message, size),
	}
}

func (mq *MemoryMQ) Publish(msg Message) {
	mq.messages <- msg
}

func (mq *MemoryMQ) Consume() (Message, bool) {
	select {
	case msg := <-mq.messages:
		return msg, true
	default:
		return Message{}, false
	}
}

// ==============================================
// 基准测试 1: 单条消费 vs 批量消费
// ==============================================

func BenchmarkMQ_SingleConsumer(b *testing.B) {
	mq := NewMemoryMQ(b.N)
	// 先生产消息
	for i := 0; i < b.N; i++ {
		mq.Publish(Message{ID: string(rune(i)), Content: "data"})
	}

	b.ReportAllocs()
	b.ResetTimer()

	// ❌ 单条消费
	for i := 0; i < b.N; i++ {
		msg, _ := mq.Consume()
		// 模拟处理单条消息
		time.Sleep(1 * time.Millisecond) // 单条处理开销
		_ = msg
	}
}

func BenchmarkMQ_BatchConsumer(b *testing.B) {
	mq := NewMemoryMQ(b.N)
	for i := 0; i < b.N; i++ {
		mq.Publish(Message{ID: string(rune(i)), Content: "data"})
	}

	b.ReportAllocs()
	b.ResetTimer()

	batchSize := 100
	// ✅ 批量消费
	for i := 0; i < b.N; i += batchSize {
		// 攒一批消息
		batch := make([]Message, 0, batchSize)
		for j := 0; j < batchSize && (i+j) < b.N; j++ {
			msg, _ := mq.Consume()
			batch = append(batch, msg)
		}
		// 批量处理，均摊开销（比如批量入库，网络开销均摊）
		time.Sleep(10 * time.Millisecond) // 100 条总共只花 10ms，均摊每条 0.1ms！
		_ = batch
	}
}

// ==============================================
// 基准测试 2: 并发消费 vs 串行消费
// ==============================================

func BenchmarkMQ_SerialConsumer(b *testing.B) {
	mq := NewMemoryMQ(b.N)
	for i := 0; i < b.N; i++ {
		mq.Publish(Message{ID: string(rune(i)), Content: "data"})
	}

	b.ReportAllocs()
	b.ResetTimer()

	// ❌ 串行消费，一个一个来
	for i := 0; i < b.N; i++ {
		msg, _ := mq.Consume()
		time.Sleep(1 * time.Millisecond) // 处理消息
		_ = msg
	}
}

func BenchmarkMQ_ConcurrentConsumer(b *testing.B) {
	mq := NewMemoryMQ(b.N)
	for i := 0; i < b.N; i++ {
		mq.Publish(Message{ID: string(rune(i)), Content: "data"})
	}

	b.ReportAllocs()
	b.ResetTimer()

	var wg sync.WaitGroup
	workerCount := 10

	// ✅ 10 个 goroutine 并发消费
	for w := 0; w < workerCount; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < b.N/workerCount; i++ {
				msg, _ := mq.Consume()
				time.Sleep(1 * time.Millisecond)
				_ = msg
			}
		}()
	}
	wg.Wait()
}

// ==============================================
// 基准测试 3: 幂等性 - Redis 去重 vs 无去重
// ==============================================

var processedMessages = sync.Map{}

// ❌ 没有幂等，重复处理
func BenchmarkMQ_NoIdempotent(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// 直接处理，不管是不是重复消息
			msgID := rand.Intn(1000) // 模拟可能重复的消息 ID
			_ = msgID
			time.Sleep(1 * time.Millisecond)
		}
	})
}

// ✅ 有幂等，用 SetNX 去重
func BenchmarkMQ_WithIdempotent(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			msgID := rand.Intn(1000)
			key := string(rune(msgID))

			// 检查是否已经处理过
			if _, loaded := processedMessages.LoadOrStore(key, true); loaded {
				continue // 已经处理过，跳过
			}

			// 第一次处理
			time.Sleep(1 * time.Millisecond)
		}
	})
}

// ==============================================
// 基准测试 4: 重试机制 - 有死信队列 vs 无限重试
// ==============================================

var retryCount int32

// 模拟一个总是失败的消息处理
func alwaysFail(msg Message) error {
	return errors.New("processing failed")
}

// ❌ 无限重试，卡住
func BenchmarkMQ_InfiniteRetry(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		msg := Message{ID: string(rune(i))}
		retry := 0
		for {
			err := alwaysFail(msg)
			if err == nil {
				break
			}
			retry++
			if retry > 10 { // 虽然有上限，但每次都重试 10 次
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
}

// ✅ 最多重试 3 次，超过进死信队列
func BenchmarkMQ_WithDLQ(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		msg := Message{ID: string(rune(i))}
		success := false

		for retry := 0; retry < 3; retry++ {
			err := alwaysFail(msg)
			if err == nil {
				success = true
				break
			}
			time.Sleep(10 * time.Millisecond)
		}

		if !success {
			// 进死信队列，人工处理，不阻塞消费
			_ = msg // 模拟写入 DLQ
		}
	}
}

// ==============================================
// 基准测试 5: 消息体大小 - 大消息 vs 小消息
// ==============================================

type SmallMessage struct { // 16 字节
	ID   int32
	Type int32
}

type BigMessage struct { // 大消息
	ID      int
	Content [10000]byte // 10KB 的内容
}

func BenchmarkMQ_SmallMessage(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// ✅ 小消息，传输快，序列化快
			msg := SmallMessage{ID: 1, Type: 2}
			_ = msg
		}
	})
}

func BenchmarkMQ_BigMessage(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// ❌ 大消息，传输慢，序列化慢
			msg := BigMessage{ID: 1}
			_ = msg
		}
	})
}

// ==============================================
// 基准测试 6: 预取缓存 - 一次拉多条 vs 一次拉一条
// ==============================================

func BenchmarkMQ_Prefetch1(b *testing.B) {
	mq := NewMemoryMQ(b.N)
	for i := 0; i < b.N; i++ {
		mq.Publish(Message{ID: string(rune(i)), Content: "data"})
	}

	b.ReportAllocs()
	b.ResetTimer()

	// ❌ 每次只拉 1 条
	for i := 0; i < b.N; i++ {
		// 模拟网络开销（每次拉取都有网络 RTT）
		time.Sleep(100 * time.Microsecond) // 模拟网络开销
		msg, _ := mq.Consume()
		time.Sleep(1 * time.Millisecond) // 处理
		_ = msg
	}
}

func BenchmarkMQ_Prefetch100(b *testing.B) {
	mq := NewMemoryMQ(b.N)
	for i := 0; i < b.N; i++ {
		mq.Publish(Message{ID: string(rune(i)), Content: "data"})
	}

	b.ReportAllocs()
	b.ResetTimer()

	// ✅ 每次拉 100 条，网络开销均摊
	for i := 0; i < b.N; i += 100 {
		// 模拟一次网络拉取 100 条
		time.Sleep(100 * time.Microsecond) // 只花一次网络开销

		// 处理 100 条
		for j := 0; j < 100 && (i+j) < b.N; j++ {
			msg, _ := mq.Consume()
			time.Sleep(1 * time.Millisecond)
			_ = msg
		}
	}
}

// ==============================================
// 基准测试 7: 消息积压处理 - 扩容消费者
// ==============================================

func BenchmarkMQ_Backlog_1Consumer(b *testing.B) {
	mq := NewMemoryMQ(b.N)
	for i := 0; i < b.N; i++ {
		mq.Publish(Message{ID: string(rune(i)), Content: "data"})
	}

	b.ReportAllocs()
	b.ResetTimer()

	// ❌ 只有 1 个消费者，慢
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < b.N; i++ {
			msg, _ := mq.Consume()
			time.Sleep(1 * time.Millisecond)
			_ = msg
		}
	}()
	wg.Wait()
}

func BenchmarkMQ_Backlog_100Consumers(b *testing.B) {
	mq := NewMemoryMQ(b.N)
	for i := 0; i < b.N; i++ {
		mq.Publish(Message{ID: string(rune(i)), Content: "data"})
	}

	b.ReportAllocs()
	b.ResetTimer()

	// ✅ 100 个消费者，快！
	var wg sync.WaitGroup
	consumerCount := 100
	for w := 0; w < consumerCount; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < b.N/consumerCount; i++ {
				msg, _ := mq.Consume()
				time.Sleep(1 * time.Millisecond)
				_ = msg
			}
		}()
	}
	wg.Wait()
}

// ==============================================
// 使用说明
// ==============================================
// 运行所有基准测试：
// go test -bench=. -benchmem -count=3 -timeout=10m
//
// 典型输出（预期）：
// BenchmarkMQ_SingleConsumer-8           1000      1000000 ns/op
// BenchmarkMQ_BatchConsumer-8           10000       100000 ns/op   (批量消费快 10 倍！)
//
// BenchmarkMQ_SerialConsumer-8           1000      1000000 ns/op
// BenchmarkMQ_ConcurrentConsumer-8      10000       100000 ns/op   (并发消费快 10 倍！)
//
// BenchmarkMQ_InfiniteRetry-8             100     10000000 ns/op
// BenchmarkMQ_WithDLQ-8                  1000      1000000 ns/op   (死信队列避免无限重试卡)
//
// BenchmarkMQ_Backlog_1Consumer-8        1000      1000000 ns/op
// BenchmarkMQ_Backlog_100Consumers-8    10000       100000 ns/op   (扩容快 100 倍！🚀)
//
// 结论：
// 1. 批量消费能显著提升吞吐量，均摊网络/IO 开销
// 2. 并发消费能线性提升消费速度（前提是下游能扛住）
// 3. 死信队列很重要，避免脏消息卡主整个消费流程
// 4. 消息积压时，第一反应应该是扩容消费者数量
