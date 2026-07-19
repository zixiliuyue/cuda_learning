package main

import (
    "sync"
    "sync/atomic"
    "testing"
)

// ==============================================
// 并发常见 Bug 复现测试
// ==============================================

// ==============================================
// Bug 1: 数据竞争 - 不加锁并发写
// ==============================================

func BenchmarkDataRace_NoLock(b *testing.B) {
    var counter int64
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            // ❌ 直接写，没有任何同步！数据竞争！
            counter++
        }
    })
}

func BenchmarkDataRace_WithMutex(b *testing.B) {
    var counter int64
    var mu sync.Mutex
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            // ✅ 加锁，安全
            mu.Lock()
            counter++
            mu.Unlock()
        }
    })
}

func BenchmarkDataRace_WithAtomic(b *testing.B) {
    var counter int64
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            // ✅ 原子操作，安全
            atomic.AddInt64(&counter, 1)
        }
    })
}

// ==============================================
// Bug 2: 闭包捕获循环变量 - 经典坑！
// ==============================================

// ❌ 错误写法：所有 goroutine 拿到的是同一个变量
func BenchmarkClosureBug(b *testing.B) {
    var wg sync.WaitGroup
    results := make([]int, b.N)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            // ❌ 所有 goroutine 共享同一个 i！
            // 等 goroutine 跑起来，循环已经结束了，i = b.N
            results[i] = i
        }()
    }
    wg.Wait()
}

// ✅ 正确写法 1：参数传进去
func BenchmarkClosureFixed_Param(b *testing.B) {
    var wg sync.WaitGroup
    results := make([]int, b.N)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        wg.Add(1)
        go func(i int) { // ✅ 作为参数传进去，每次都会复制
            defer wg.Done()
            results[i] = i
        }(i)
    }
    wg.Wait()
}

// ✅ 正确写法 2：循环内局部变量
func BenchmarkClosureFixed_Local(b *testing.B) {
    var wg sync.WaitGroup
    results := make([]int, b.N)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        i := i // ✅ 循环内的局部变量，每次迭代都有新的
        wg.Add(1)
        go func() {
            defer wg.Done()
            results[i] = i
        }()
    }
    wg.Wait()
}

// ==============================================
// Bug 3: Goroutine 泄漏 - channel 没人接收
// ==============================================

func BenchmarkGoroutineLeak(b *testing.B) {
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        ch := make(chan int)
        go func() {
            // ❌ 如果没人接收，这个 goroutine 永远卡住！
            ch <- 123
        }()
        // 函数返回了，goroutine 泄漏了！
    }
}

func BenchmarkGoroutineNoLeak(b *testing.B) {
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        // ✅ 有缓冲，goroutine 放完就走，不阻塞
        ch := make(chan int, 1)
        go func() {
            ch <- 123
        }()
        // goroutine 发完就可以退出了
    }
}

// ==============================================
// Bug 4: 循环内使用 defer
// ==============================================

// ❌ 错误写法：循环内 defer，所有锁堆到最后一起释放
func BenchmarkDeferInLoop_Bad(b *testing.B) {
    var mu sync.Mutex
    var counter int64
    
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        mu.Lock()
        defer mu.Unlock() // ❌ 循环结束才会执行！等于永远占着锁！
        counter++
    }
}

// ✅ 正确写法：每次循环手动解锁
func BenchmarkDeferInLoop_Good(b *testing.B) {
    var mu sync.Mutex
    var counter int64
    
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        mu.Lock()
        counter++
        mu.Unlock() // ✅ 用完就解锁
    }
}

// ==============================================
// 使用说明：
// ==============================================
// 1. 运行数据竞争检测：
// go test -race -run="DataRace" -bench="DataRace"
//
// 输出应该会有：
// WARNING: DATA RACE
//
// 2. 运行闭包测试，看看结果对不对
//
// 3. Goroutine 泄漏检测：
// go tool pprof ... 看 goroutine 数量是不是一直涨
//
// 结论：
// - 并发写一定要加锁或者原子操作
// - 循环内启动 goroutine，一定要小心捕获变量！
// - channel 一定要有接收者，否则 goroutine 泄漏
// - 循环内不要 defer，defer 是函数级别的！
