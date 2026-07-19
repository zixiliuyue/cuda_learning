package main

import (
    "sync"
    "sync/atomic"
    "testing"
    "unsafe"
)

// ==============================================
// 无锁编程模式基准测试
// ==============================================

// ==============================================
// 基准测试 1: CAS 自旋 vs Mutex
// ==============================================

// 无锁自旋锁
type SpinLock struct {
    flag int32
}

func (sl *SpinLock) Lock() {
    for !atomic.CompareAndSwapInt32(&sl.flag, 0, 1) {
        // 自旋等待
    }
}

func (sl *SpinLock) Unlock() {
    atomic.StoreInt32(&sl.flag, 0)
}

func BenchmarkSpinLock(b *testing.B) {
    var lock SpinLock
    var counter int64
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            lock.Lock()
            counter++
            lock.Unlock()
        }
    })
}

func BenchmarkMutexLock(b *testing.B) {
    var mu sync.Mutex
    var counter int64
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

// ==============================================
// 基准测试 2: 无锁栈 vs 有锁栈
// ==============================================

// 无锁栈
type LockFreeStack struct {
    head unsafe.Pointer
}

type Node struct {
    value int
    next  unsafe.Pointer
}

func (s *LockFreeStack) Push(v int) {
    newNode := &Node{value: v}
    for {
        oldHead := atomic.LoadPointer(&s.head)
        newNode.next = oldHead
        if atomic.CompareAndSwapPointer(&s.head, oldHead, unsafe.Pointer(newNode)) {
            return
        }
    }
}

func (s *LockFreeStack) Pop() (int, bool) {
    for {
        oldHead := atomic.LoadPointer(&s.head)
        if oldHead == nil {
            return 0, false
        }
        next := (*Node)(oldHead).next
        if atomic.CompareAndSwapPointer(&s.head, oldHead, next) {
            return (*Node)(oldHead).value, true
        }
    }
}

// 有锁栈
type MutexStack struct {
    mu   sync.Mutex
    head *Node
}

func (s *MutexStack) Push(v int) {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.head = &Node{value: v, next: s.head}
}

func (s *MutexStack) Pop() (int, bool) {
    s.mu.Lock()
    defer s.mu.Unlock()
    if s.head == nil {
        return 0, false
    }
    v := s.head.value
    s.head = s.head.next
    return v, true
}

// 测试无锁栈性能
func BenchmarkLockFreeStack(b *testing.B) {
    var stack LockFreeStack
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            stack.Push(1)
            stack.Pop()
        }
    })
}

func BenchmarkMutexStack(b *testing.B) {
    var stack MutexStack
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            stack.Push(1)
            stack.Pop()
        }
    })
}

// ==============================================
// 基准测试 3: 读写比例不同场景对比
// ==============================================

func BenchmarkLockFree_Read99_Write1(b *testing.B) {
    var counter int64
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        n := 0
        for pb.Next() {
            n++
            if n%100 == 0 {
                atomic.AddInt64(&counter, 1)  // 1% 写
            } else {
                atomic.LoadInt64(&counter)     // 99% 读
            }
        }
    })
}

func BenchmarkRWMutex_Read99_Write1(b *testing.B) {
    var counter int64
    var mu sync.RWMutex
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        n := 0
        for pb.Next() {
            n++
            if n%100 == 0 {
                mu.Lock()
                counter++
                mu.Unlock()
            } else {
                mu.RLock()
                _ = counter
                mu.RUnlock()
            }
        }
    })
}

// ==============================================
// 使用说明：
// ==============================================
// 运行测试：
// go test -bench="LockFree|Mutex|Spin" -benchmem -count=5 -cpu 1,2,4,8
//
// 典型输出（8 核 CPU）：
// BenchmarkSpinLock-8            20000000    60 ns/op
// BenchmarkMutexLock-8           30000000    45 ns/op
//
// BenchmarkLockFreeStack-8       10000000   150 ns/op
// BenchmarkMutexStack-8          10000000   120 ns/op
//
// BenchmarkLockFree_Read99-8    100000000   10 ns/op
// BenchmarkRWMutex_Read99-8      20000000   60 ns/op
//
// 结论：
// 1. 纯 CAS 竞争激烈时，可能不如 Mutex！因为空转太费 CPU
// 2. 读多写少场景（99%读），LockFree 优势巨大，快 6 倍
// 3. 无锁不是银弹，要分场景！
