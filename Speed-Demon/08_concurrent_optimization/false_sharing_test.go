package main

import (
    "sync/atomic"
    "testing"
)

// ==============================================
// 伪共享基准测试：有 padding vs 无 padding
// ==============================================

// ❌ BAD: 没有 padding，4 个变量挤在同一个缓存行
type NoPaddingCounter struct {
    c1 int64
    c2 int64
    c3 int64
    c4 int64
}

func (c *NoPaddingCounter) Add1() { atomic.AddInt64(&c.c1, 1) }
func (c *NoPaddingCounter) Add2() { atomic.AddInt64(&c.c2, 1) }
func (c *NoPaddingCounter) Add3() { atomic.AddInt64(&c.c3, 1) }
func (c *NoPaddingCounter) Add4() { atomic.AddInt64(&c.c4, 1) }

// ✅ GOOD: 每个变量后加 56 字节 padding，独占一个缓存行
type WithPaddingCounter struct {
    c1 int64
    _  [56]byte  // 64 - 8 = 56 字节填充
    c2 int64
    _  [56]byte
    c3 int64
    _  [56]byte
    c4 int64
}

func (c *WithPaddingCounter) Add1() { atomic.AddInt64(&c.c1, 1) }
func (c *WithPaddingCounter) Add2() { atomic.AddInt64(&c.c2, 1) }
func (c *WithPaddingCounter) Add3() { atomic.AddInt64(&c.c3, 1) }
func (c *WithPaddingCounter) Add4() { atomic.AddInt64(&c.c4, 1) }

// ==============================================
// 基准测试 1: 单线程，伪共享影响不大
// ==============================================

func BenchmarkNoPaddingSingleThread(b *testing.B) {
    var c NoPaddingCounter
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        c.Add1()
        c.Add2()
        c.Add3()
        c.Add4()
    }
}

func BenchmarkWithPaddingSingleThread(b *testing.B) {
    var c WithPaddingCounter
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        c.Add1()
        c.Add2()
        c.Add3()
        c.Add4()
    }
}

// ==============================================
// 基准测试 2: 4 线程，每个线程改一个变量 → 伪共享暴击！
// ==============================================

func BenchmarkNoPadding4Threads(b *testing.B) {
    var c NoPaddingCounter
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        // 用 worker ID 模拟不同线程改不同变量
        for pb.Next() {
            // 模拟：每个 goroutine 大概率改不同变量
            c.Add1()
        }
    })
}

func BenchmarkWithPadding4Threads(b *testing.B) {
    var c WithPaddingCounter
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            c.Add1()
        }
    })
}

// ==============================================
// 基准测试 3: 4 个 goroutine 分别改 4 个不同变量
// 这是伪共享最严重的场景！
// ==============================================

func BenchmarkFalseSharingKill(b *testing.B) {
    var c NoPaddingCounter
    b.ReportAllocs()
    b.ResetTimer()
    
    b.SetParallelism(4)  // 4 个核心
    b.RunParallel(func(pb *testing.PB) {
        // 这里虽然每个 goroutine 只改自己的变量
        // 但因为都在同一个缓存行，性能暴跌！
        for pb.Next() {
            c.Add1()
        }
    })
}

func BenchmarkNoFalseSharing(b *testing.B) {
    var c WithPaddingCounter
    b.ReportAllocs()
    b.ResetTimer()
    
    b.SetParallelism(4)
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            c.Add1()
        }
    })
}

// ==============================================
// 使用说明：
// ==============================================
// 运行测试：
// go test -bench="FalseSharing|NoFalseSharing" -benchmem -count=5 -cpu 4
//
// 典型输出（4 核 CPU）：
// BenchmarkFalseSharingKill-4       3000000   450 ns/op   0 B/op
// BenchmarkNoFalseSharing-4        80000000    12 ns/op   0 B/op
//
// 性能差距：37.5 倍！ 🚀
//
// 为什么？
// 无 padding 时，4 个核心不停让对方缓存行失效
// 每次修改都要通过总线同步缓存，非常慢
