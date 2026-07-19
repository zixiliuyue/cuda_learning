package main

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-redis/redis/v8"
	"golang.org/x/net/context"
)

var ctx = context.Background()
var rdb *redis.Client

func setupRedis() {
	rdb = redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
		DB:   0,
	})
}

// ==============================================
// 基准测试 1: 缓存穿透 - 空值缓存 vs 无缓存
// ==============================================

func BenchmarkCachePenetration_NoNullCache(b *testing.B) {
	setupRedis()
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// ❌ 模拟：查不存在的 ID，每次都查 DB（这里用 sleep 模拟 DB 查询）
			id := rand.Intn(10000) + 100000 // 不存在的 ID 范围
			key := fmt.Sprintf("user:%d", id)
			rdb.Get(ctx, key)
			// 缓存没有，模拟查 DB
			time.Sleep(10 * time.Millisecond)
		}
	})
}

func BenchmarkCachePenetration_WithNullCache(b *testing.B) {
	setupRedis()
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// ✅ 模拟：查不存在的 ID，空值缓存命中，不用查 DB
			id := rand.Intn(10000) + 100000
			key := fmt.Sprintf("user:%d", id)
			val, _ := rdb.Get(ctx, key).Result()
			if val == "" {
				// 第一次查，缓存空值
				rdb.SetEX(ctx, key, "NULL", 5*time.Minute)
				time.Sleep(10 * time.Millisecond) // 模拟第一次查 DB
			}
			// 后续查直接命中空值缓存，不用查 DB
		}
	})
}

// ==============================================
// 基准测试 2: 缓存击穿 - 互斥锁 vs 无锁
// ==============================================

var (
	cacheHit   int64
	cacheMiss  int64
)

// ❌ 无互斥锁：缓存失效时，所有 goroutine 同时查 DB
func BenchmarkCacheBreakdown_NoLock(b *testing.B) {
	setupRedis()
	// 预热：先设置一个即将过期的缓存
	rdb.SetEX(ctx, "hot_product:123", "product_data", 1*time.Second)
	time.Sleep(2 * time.Second) // 等缓存过期

	b.ReportAllocs()
	b.ResetTimer()

	b.SetParallelism(100) // 100 个并发
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			key := "hot_product:123"
			val, _ := rdb.Get(ctx, key).Result()
			if val == "" {
				// 缓存失效，所有人同时查 DB
				time.Sleep(50 * time.Millisecond) // 模拟查 DB
				rdb.SetEX(ctx, key, "product_data", 1*time.Hour)
				atomic.AddInt64(&cacheMiss, 1)
			} else {
				atomic.AddInt64(&cacheHit, 1)
			}
		}
	})
}

// ✅ 互斥锁：只有一个 goroutine 查 DB，其他人等结果
func BenchmarkCacheBreakdown_WithLock(b *testing.B) {
	setupRedis()
	rdb.SetEX(ctx, "hot_product:456", "product_data", 1*time.Second)
	time.Sleep(2 * time.Second)

	b.ReportAllocs()
	b.ResetTimer()

	b.SetParallelism(100)
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			key := "hot_product:456"
			val, _ := rdb.Get(ctx, key).Result()
			if val == "" {
				// 抢锁
				lockKey := "lock:product:456"
				locked, _ := rdb.SetNX(ctx, lockKey, "1", 10*time.Second).Result()
				if locked {
					defer rdb.Del(ctx, lockKey)
					// 拿到锁，查 DB
					time.Sleep(50 * time.Millisecond)
					rdb.SetEX(ctx, key, "product_data", 1*time.Hour)
					atomic.AddInt64(&cacheMiss, 1)
				} else {
					// 没抢到，等一下再查
					time.Sleep(10 * time.Millisecond)
				}
			} else {
				atomic.AddInt64(&cacheHit, 1)
			}
		}
	})
}

// ==============================================
// 基准测试 3: Pipeline vs 循环单条
// ==============================================

func BenchmarkRedis_LoopGet(b *testing.B) {
	setupRedis()
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// ❌ 循环 100 次 Get，100 次网络往返
		for j := 0; j < 100; j++ {
			rdb.Get(ctx, fmt.Sprintf("key:%d", j))
		}
	}
}

func BenchmarkRedis_Pipeline(b *testing.B) {
	setupRedis()
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// ✅ Pipeline，1 次网络往返
		pipe := rdb.Pipeline()
		for j := 0; j < 100; j++ {
			pipe.Get(ctx, fmt.Sprintf("key:%d", j))
		}
		pipe.Exec(ctx)
	}
}

func BenchmarkRedis_MGet(b *testing.B) {
	setupRedis()
	b.ReportAllocs()
	b.ResetTimer()

	keys := make([]string, 100)
	for i := 0; i < 100; i++ {
		keys[i] = fmt.Sprintf("key:%d", i)
	}

	for i := 0; i < b.N; i++ {
		// ✅ MGet，1 次网络往返，最快！
		rdb.MGet(ctx, keys...)
	}
}

// ==============================================
// 基准测试 4: 缓存雪崩 - 随机过期 vs 同时过期
// ==============================================

func BenchmarkCacheAvalanche_SameExpire(b *testing.B) {
	setupRedis()
	// 预热：1000 个 key，都是 1 秒过期
	for i := 0; i < 1000; i++ {
		rdb.SetEX(ctx, fmt.Sprintf("product:%d", i), "data", 1*time.Second)
	}
	time.Sleep(2 * time.Second) // 等所有 key 同时过期

	b.ReportAllocs()
	b.ResetTimer()

	b.SetParallelism(100)
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			id := rand.Intn(1000)
			key := fmt.Sprintf("product:%d", id)
			val, _ := rdb.Get(ctx, key).Result()
			if val == "" {
				// 大量缓存同时失效，大量请求打 DB
				time.Sleep(10 * time.Millisecond)
				rdb.SetEX(ctx, key, "data", 1*time.Second)
			}
		}
	})
}

func BenchmarkCacheAvalanche_RandomExpire(b *testing.B) {
	setupRedis()
	// 预热：1000 个 key，过期时间 50~70 秒随机
	for i := 0; i < 1000; i++ {
		expire := 50 + rand.Intn(20)
		rdb.SetEX(ctx, fmt.Sprintf("product2:%d", i), "data", time.Duration(expire)*time.Second)
	}

	b.ReportAllocs()
	b.ResetTimer()

	b.SetParallelism(100)
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			id := rand.Intn(1000)
			key := fmt.Sprintf("product2:%d", id)
			val, _ := rdb.Get(ctx, key).Result()
			if val == "" {
				// 只有少量 key 过期，DB 压力小
				time.Sleep(10 * time.Millisecond)
				expire := 50 + rand.Intn(20)
				rdb.SetEX(ctx, key, "data", time.Duration(expire)*time.Second)
			}
		}
	})
}

// ==============================================
// 基准测试 5: 本地缓存 vs Redis
// ==============================================

var (
	localCache = sync.Map{}
)

func Benchmark_NoLocalCache(b *testing.B) {
	setupRedis()
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// ❌ 每次都查 Redis
			rdb.Get(ctx, "hot_key")
		}
	})
}

func Benchmark_WithLocalCache(b *testing.B) {
	setupRedis()
	// 预热本地缓存
	localCache.Store("hot_key", "local_data")

	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// ✅ 先查本地缓存，命中直接返回
			if val, ok := localCache.Load("hot_key"); ok {
				_ = val // 命中本地缓存，0 网络开销！
			} else {
				rdb.Get(ctx, "hot_key")
			}
		}
	})
}

// ==============================================
// 使用说明
// ==============================================
// 运行前先启动本地 Redis
//
// 运行所有基准测试：
// go test -bench=. -benchmem -count=3 -timeout=30m
//
// 典型输出（预期）：
// BenchmarkCachePenetration_NoNullCache-8        1000    1200000 ns/op
// BenchmarkCachePenetration_WithNullCache-8     10000      12000 ns/op   (快 100 倍！)
//
// BenchmarkRedis_LoopGet-8                        100   12000000 ns/op
// BenchmarkRedis_Pipeline-8                      5000     200000 ns/op   (快 60 倍！)
// BenchmarkRedis_MGet-8                         10000     100000 ns/op   (快 120 倍！)
//
// Benchmark_NoLocalCache-8                      20000      50000 ns/op
// Benchmark_WithLocalCache-8                  10000000        100 ns/op   (快 500 倍！🚀)
//
// 结论：
// 1. 空值缓存能有效防止缓存穿透，性能提升巨大
// 2. Pipeline/MGet 比循环单条快几十上百倍
// 3. 本地缓存（进程内缓存）比 Redis 快几百倍！
