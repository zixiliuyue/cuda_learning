package main

import (
    "sync"
    "sync/atomic"
    "testing"
)

// ==============================================
// 基准测试 1: 原子操作 vs 互斥锁 计数器对比
// ==============================================

func BenchmarkAtomicCounter(b *testing.B) {
    var counter int64
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            atomic.AddInt64(&counter, 1)
        }
    })
}

func BenchmarkMutexCounter(b *testing.B) {
    var counter int64
    var mu sync.Mutex
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            mu.Lock()
            counter++
            mu.Unlock()
        }
    })
}

func BenchmarkRWMutexRead(b *testing.B) {
    var counter int64
    var mu sync.RWMutex
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            mu.RLock()
            _ = counter
            mu.RUnlock()
        }
    })
}

func BenchmarkAtomicRead(b *testing.B) {
    var counter int64
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _ = atomic.LoadInt64(&counter)
        }
    })
}

// ==============================================
// 基准测试 2: 不同并发度下的性能对比
// ==============================================

func BenchmarkAtomic_1Goroutine(b *testing.B) {
    var counter int64
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        atomic.AddInt64(&counter, 1)
    }
}

func BenchmarkMutex_1Goroutine(b *testing.B) {
    var counter int64
    var mu sync.Mutex
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        mu.Lock()
        counter++
        mu.Unlock()
    }
}

// ==============================================
// 基准测试 3: 不同竞争激烈程度对比
// ==============================================

// 低竞争：临界区很小
func BenchmarkMutexLowContention(b *testing.B) {
    var counter int64
    var mu sync.Mutex
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            mu.Lock()
            counter++  // 非常快的操作
            mu.Unlock()
        }
    })
}

// 高竞争：临界区里有额外开销（模拟真实场景）
func BenchmarkMutexHighContention(b *testing.B) {
    var counter int64
    var mu sync.Mutex
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            mu.Lock()
            counter++
            // 模拟临界区有一点点开销
            for i := 0; i < 10; i++ {
                counter++
            }
            mu.Unlock()
        }
    })
}

// ==============================================
// 使用说明：
// ==============================================
// 运行所有基准测试：
// go test -bench=. -benchmem -count=5 -cpu 1,2,4,8 > atomic_vs_mutex.txt
//
// 对比分析：
// benchstat atomic_vs_mutex.txt
//
// 典型输出（8 核 CPU）：
// BenchmarkAtomicCounter-8        100000000    10.2 ns/op    0 B/op   0 allocs/op
// BenchmarkMutexCounter-8          30000000    45.6 ns/op    0 B/op   0 allocs/op
// BenchmarkRWMutexRead-8           50000000    22.3 ns/op    0 B/op   0 allocs/op
// BenchmarkAtomicRead-8           200000000     3.1 ns/op    0 B/op   0 allocs/op
//
// 结论：
// - 写入场景：atomic 比 mutex 快 4-5 倍
// - 读取场景：atomic 比 RWMutex 读快 7 倍
// - 竞争越激烈，atomic 的优势越明显
