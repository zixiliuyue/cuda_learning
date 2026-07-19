package main

import (
    "sync"
    "testing"
)

// ==============================================
// Channel 性能基准测试
// ==============================================

// ==============================================
// 基准测试 1: 无缓冲 vs 有缓冲 Channel
// ==============================================

func BenchmarkUnbufferedChannel(b *testing.B) {
    ch := make(chan int)
    var wg sync.WaitGroup
    wg.Add(1)
    
    go func() {
        defer wg.Done()
        for range ch {
        }
    }()
    
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        ch <- i
    }
    
    close(ch)
    wg.Wait()
}

func BenchmarkBufferedChannel_1(b *testing.B) {
    ch := make(chan int, 1)  // 缓冲大小 = 1
    var wg sync.WaitGroup
    wg.Add(1)
    
    go func() {
        defer wg.Done()
        for range ch {
        }
    }()
    
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        ch <- i
    }
    
    close(ch)
    wg.Wait()
}

func BenchmarkBufferedChannel_100(b *testing.B) {
    ch := make(chan int, 100)  // 缓冲大小 = 100
    var wg sync.WaitGroup
    wg.Add(1)
    
    go func() {
        defer wg.Done()
        for range ch {
        }
    }()
    
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        ch <- i
    }
    
    close(ch)
    wg.Wait()
}

// ==============================================
// 基准测试 2: Channel 消息大小影响
// ==============================================

type SmallMessage struct {  // 16 字节
    ID   int32
    Type int32
}

type BigMessage struct {  // 112 字节
    ID      int64
    Name    string
    Tags    [10]int
    Payload [64]byte
}

func BenchmarkChannelSmallMessage(b *testing.B) {
    ch := make(chan SmallMessage, 100)
    var wg sync.WaitGroup
    wg.Add(1)
    
    go func() {
        defer wg.Done()
        for range ch {
        }
    }()
    
    b.ReportAllocs()
    b.ResetTimer()
    
    msg := SmallMessage{ID: 1, Type: 2}
    for i := 0; i < b.N; i++ {
        ch <- msg
    }
    
    close(ch)
    wg.Wait()
}

func BenchmarkChannelBigMessage(b *testing.B) {
    ch := make(chan BigMessage, 100)
    var wg sync.WaitGroup
    wg.Add(1)
    
    go func() {
        defer wg.Done()
        for range ch {
        }
    }()
    
    b.ReportAllocs()
    b.ResetTimer()
    
    msg := BigMessage{ID: 1}
    for i := 0; i < b.N; i++ {
        ch <- msg
    }
    
    close(ch)
    wg.Wait()
}

func BenchmarkChannelBigMessagePointer(b *testing.B) {
    ch := make(chan *BigMessage, 100)  // 传指针！
    var wg sync.WaitGroup
    wg.Add(1)
    
    go func() {
        defer wg.Done()
        for range ch {
        }
    }()
    
    b.ReportAllocs()
    b.ResetTimer()
    
    msg := &BigMessage{ID: 1}
    for i := 0; i < b.N; i++ {
        ch <- msg
    }
    
    close(ch)
    wg.Wait()
}

// ==============================================
// 基准测试 3: 工作池性能
// ==============================================

func worker(ch chan int, wg *sync.WaitGroup) {
    defer wg.Done()
    for range ch {
    }
}

func BenchmarkWorkerPool_4(b *testing.B) {
    ch := make(chan int, 1000)
    var wg sync.WaitGroup
    
    for i := 0; i < 4; i++ {
        wg.Add(1)
        go worker(ch, &wg)
    }
    
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        ch <- i
    }
    
    close(ch)
    wg.Wait()
}

func BenchmarkWorkerPool_32(b *testing.B) {
    ch := make(chan int, 1000)
    var wg sync.WaitGroup
    
    for i := 0; i < 32; i++ {
        wg.Add(1)
        go worker(ch, &wg)
    }
    
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        ch <- i
    }
    
    close(ch)
    wg.Wait()
}

// ==============================================
// 使用说明：
// ==============================================
// 运行所有测试：
// go test -bench="Channel|WorkerPool" -benchmem -count=5
//
// 典型输出（8 核 CPU）：
// BenchmarkUnbufferedChannel-8       10000000   120 ns/op
// BenchmarkBufferedChannel_1-8       20000000    65 ns/op
// BenchmarkBufferedChannel_100-8     50000000    22 ns/op
//
// BenchmarkChannelSmallMessage-8     50000000    25 ns/op
// BenchmarkChannelBigMessage-8       10000000   150 ns/op
// BenchmarkChannelBigMessagePointer-8 40000000   30 ns/op
//
// 结论：
// 1. 有缓冲比无缓冲快 2-5 倍！
// 2. 大结构体传指针，性能提升 5 倍！
// 3. 缓冲大小不是越大越好，够用就行
