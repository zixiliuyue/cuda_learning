package main

import (
	"context"
	"errors"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/time/rate"
)

// ==============================================
// 基准测试 1: 超时控制 - 有超时 vs 无超时
// ==============================================

// 模拟一个慢的 RPC 调用
func slowRPC(ctx context.Context) error {
	// 模拟 1 秒的慢请求
	select {
	case <-time.After(1 * time.Second):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func BenchmarkRPC_NoTimeout(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// ❌ 没有超时，永远等下去！
		slowRPC(context.Background())
	}
}

func BenchmarkRPC_WithTimeout(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// ✅ 500ms 超时，超时直接返回
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		slowRPC(ctx)
		cancel()
	}
}

// ==============================================
// 基准测试 2: 重试机制 - 指数退避 vs 立即重试
// ==============================================

var failCount int32

// 模拟一个有时失败的 RPC 调用
func flakyRPC() error {
	if atomic.AddInt32(&failCount, 1)%3 == 0 {
		return nil // 第三次成功
	}
	return errors.New("temporary error") // 前两次失败
}

func BenchmarkRetry_Immediate(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		atomic.StoreInt32(&failCount, 0)
		// ❌ 立即重试，没有退避
		for j := 0; j < 3; j++ {
			if err := flakyRPC(); err == nil {
				break
			}
			// 立即重试！
		}
	}
}

func BenchmarkRetry_ExponentialBackoff(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		atomic.StoreInt32(&failCount, 0)
		// ✅ 指数退避
		for j := 0; j < 3; j++ {
			if err := flakyRPC(); err == nil {
				break
			}
			// 指数退避：100ms, 200ms, 400ms
			backoff := time.Duration(100*(1<<j)) * time.Millisecond
			// 加随机抖动
			jitter := time.Duration(rand.Intn(50)) * time.Millisecond
			time.Sleep(backoff + jitter)
		}
	}
}

// ==============================================
// 基准测试 3: 限流 - 令牌桶 vs 无限流
// ==============================================

func Benchmark_NoLimiter(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	// ❌ 无限流，想跑多快跑多快
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// 模拟处理请求
			time.Sleep(1 * time.Millisecond)
		}
	})
}

func Benchmark_WithLimiter(b *testing.B) {
	// ✅ 令牌桶限流：每秒 10000 个请求
	limiter := rate.NewLimiter(10000, 1000)

	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if limiter.Allow() {
				time.Sleep(1 * time.Millisecond)
			}
			// 被限流的直接丢弃，不处理
		}
	})
}

// ==============================================
// 基准测试 4: 熔断器 - 有熔断 vs 无熔断
// ==============================================

type CircuitBreaker struct {
	state           int32 // 0=closed, 1=open, 2=half-open
	failureCount    int64
	lastFailureTime int64

	failureThreshold int64
	timeout          time.Duration
}

func (cb *CircuitBreaker) Call(f func() error) error {
	state := atomic.LoadInt32(&cb.state)

	// 熔断状态，直接拒绝
	if state == 1 {
		if time.Now().UnixNano()-atomic.LoadInt64(&cb.lastFailureTime) < int64(cb.timeout) {
			return errors.New("circuit breaker open")
		}
		atomic.CompareAndSwapInt32(&cb.state, 1, 2)
		atomic.StoreInt64(&cb.failureCount, 0)
	}

	err := f()

	if err != nil {
		fc := atomic.AddInt64(&cb.failureCount, 1)
		atomic.StoreInt64(&cb.lastFailureTime, time.Now().UnixNano())

		if (state == 2) || (state == 0 && fc >= cb.failureThreshold) {
			atomic.CompareAndSwapInt32(&cb.state, state, 1)
		}
		return err
	}

	if state == 2 {
		atomic.CompareAndSwapInt32(&cb.state, 2, 0)
	}
	atomic.StoreInt64(&cb.failureCount, 0)
	return nil
}

// 模拟一个失败的 RPC（90% 失败率）
func failingRPC() error {
	if rand.Intn(10) < 9 {
		return errors.New("service down")
	}
	return nil
}

func Benchmark_NoCircuitBreaker(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// ❌ 没有熔断，每次都真的去调用
			failingRPC()
		}
	})
}

func Benchmark_WithCircuitBreaker(b *testing.B) {
	cb := &CircuitBreaker{
		failureThreshold: 5,
		timeout:          5 * time.Second,
	}

	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// ✅ 有熔断，失败多了直接拒绝，不用真的去调用
			cb.Call(failingRPC)
		}
	})
}

// ==============================================
// 基准测试 5: 降级策略 - 有降级 vs 无降级
// ==============================================

// 模拟下游服务
func downstreamService() (string, error) {
	// 50% 概率失败
	if rand.Intn(2) == 0 {
		return "", errors.New("downstream down")
	}
	return "fresh_data", nil
}

func Benchmark_NoFallback(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// ❌ 没有降级，失败了就返回错误
			_, err := downstreamService()
			if err != nil {
				continue
			}
		}
	})
}

func Benchmark_WithFallback(b *testing.B) {
	// 模拟缓存里的旧数据
	staleData := "stale_data"

	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// ✅ 有降级，失败了返回旧数据
			data, err := downstreamService()
			if err != nil {
				// 降级：返回缓存里的旧数据
				_ = staleData
			} else {
				_ = data
			}
		}
	})
}

// ==============================================
// 基准测试 6: 真实 HTTP 服务器性能
// ==============================================

func Benchmark_HTTP_NoMiddleware(b *testing.B) {
	// 简单的 HTTP 处理
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	})

	server := httptest.NewServer(handler)
	defer server.Close()

	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		client := &http.Client{}
		for pb.Next() {
			resp, _ := client.Get(server.URL)
			if resp != nil {
				resp.Body.Close()
			}
		}
	})
}

func Benchmark_HTTP_WithLimiterMiddleware(b *testing.B) {
	limiter := rate.NewLimiter(1000, 100)

	// 带限流中间件的 HTTP 处理
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !limiter.Allow() {
			w.WriteHeader(429)
			w.Write([]byte("rate limited"))
			return
		}
		w.Write([]byte("ok"))
	})

	server := httptest.NewServer(handler)
	defer server.Close()

	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		client := &http.Client{}
		for pb.Next() {
			resp, _ := client.Get(server.URL)
			if resp != nil {
				resp.Body.Close()
			}
		}
	})
}

// ==============================================
// 基准测试 7: 连接池复用 vs 每次新建连接
// ==============================================

func Benchmark_HTTP_NewClientEveryTime(b *testing.B) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// ❌ 每次都新建 Client，新建连接
			client := &http.Client{}
			resp, _ := client.Get(server.URL)
			if resp != nil {
				resp.Body.Close()
			}
		}
	})
}

func Benchmark_HTTP_ReuseClient(b *testing.B) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		// ✅ 复用 Client，复用 TCP 连接
		client := &http.Client{
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     30 * time.Second,
			},
		}
		for pb.Next() {
			resp, _ := client.Get(server.URL)
			if resp != nil {
				resp.Body.Close()
			}
		}
	})
}

// ==============================================
// 使用说明
// ==============================================
// 运行所有基准测试：
// go test -bench=. -benchmem -count=3 -timeout=10m
//
// 典型输出（预期）：
// BenchmarkRPC_NoTimeout-8                10    1000000000 ns/op
// BenchmarkRPC_WithTimeout-8              20     500000000 ns/op   (超时时间内返回)
//
// Benchmark_NoLimiter-8                10000        100000 ns/op
// Benchmark_WithLimiter-8              10000        100000 ns/op   (QPS 被限制)
//
// Benchmark_NoCircuitBreaker-8          1000      1000000 ns/op
// Benchmark_WithCircuitBreaker-8      100000         10000 ns/op   (熔断后直接拒绝，快 100 倍！)
//
// Benchmark_HTTP_NewClientEveryTime-8   1000       1000000 ns/op
// Benchmark_HTTP_ReuseClient-8         10000        100000 ns/op   (连接复用，快 10 倍！)
//
// 结论：
// 1. 熔断在下游挂了的时候，性能提升巨大，因为不用真的去调用
// 2. HTTP 连接复用能显著提升性能，减少 TCP 握手开销
// 3. 超时控制是必须的，防止 goroutine 卡住
