package main

import (
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func BenchmarkNaiveConcat(b *testing.B) {
	for i := 0; i < b.N; i++ {
		result := ""
		for j := 0; j < 1000; j++ {
			result += "hello "
		}
		_ = result
	}
}

func BenchmarkBuilderConcat(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var builder strings.Builder
		builder.Grow(1000 * 6)
		for j := 0; j < 1000; j++ {
			builder.WriteString("hello ")
		}
		_ = builder.String()
	}
}

func BenchmarkMutex(b *testing.B) {
	var mu sync.Mutex
	var counter int64
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			mu.Lock()
			counter++
			mu.Unlock()
		}
	})
}

func BenchmarkAtomic(b *testing.B) {
	var counter int64
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			atomic.AddInt64(&counter, 1)
		}
	})
}

func BenchmarkChannelUnbuffered(b *testing.B) {
	ch := make(chan int)
	
	go func() {
		for val := range ch {
			_ = val
		}
	}()
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			ch <- 1
		}
	})
}

func BenchmarkChannelBuffered(b *testing.B) {
	ch := make(chan int, 100)
	
	go func() {
		for val := range ch {
			_ = val
		}
	}()
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			ch <- 1
		}
	})
}

func BenchmarkSliceAppendNoCap(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var s []int
		for j := 0; j < 10000; j++ {
			s = append(s, j)
		}
		_ = s
	}
}

func BenchmarkSliceAppendWithCap(b *testing.B) {
	for i := 0; i < b.N; i++ {
		s := make([]int, 0, 10000)
		for j := 0; j < 10000; j++ {
			s = append(s, j)
		}
		_ = s
	}
}

func BenchmarkMakeSliceSmall(b *testing.B) {
	for i := 0; i < b.N; i++ {
		s := make([]int, 16)
		_ = s
	}
}

func BenchmarkMakeSliceLarge(b *testing.B) {
	for i := 0; i < b.N; i++ {
		s := make([]int, 1024)
		_ = s
	}
}

func BenchmarkMap(b *testing.B) {
	m := make(map[int]int)
	
	for i := 0; i < 1000; i++ {
		m[i] = i
	}
	
	b.ResetTimer()
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			for i := 0; i < 100; i++ {
				_ = m[i]
			}
		}
	})
}

func BenchmarkSyncMap(b *testing.B) {
	var m sync.Map
	
	for i := 0; i < 1000; i++ {
		m.Store(i, i)
	}
	
	b.ResetTimer()
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			for i := 0; i < 100; i++ {
				_, _ = m.Load(i)
			}
		}
	})
}

func BenchmarkDefer(b *testing.B) {
	var mu sync.Mutex
	
	for i := 0; i < b.N; i++ {
		mu.Lock()
		mu.Unlock()
	}
}

func BenchmarkDeferUnlock(b *testing.B) {
	var mu sync.Mutex
	
	for i := 0; i < b.N; i++ {
		mu.Lock()
		defer mu.Unlock()
	}
}

func BenchmarkPool(b *testing.B) {
	pool := sync.Pool{
		New: func() interface{} {
			return make([]byte, 4096)
		},
	}
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf := pool.Get().([]byte)
			pool.Put(buf)
		}
	})
}

func BenchmarkNoPool(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf := make([]byte, 4096)
			_ = buf
		}
	})
}

func BenchmarkInterfaceCall(b *testing.B) {
	var s fmt.Stringer = &MyStruct{val: 42}
	
	for i := 0; i < b.N; i++ {
		_ = s.String()
	}
}

func BenchmarkDirectCall(b *testing.B) {
	s := &MyStruct{val: 42}
	
	for i := 0; i < b.N; i++ {
		_ = s.String()
	}
}

type MyStruct struct {
	val int
}

func (m *MyStruct) String() string {
	return "hello"
}

var bufPool = sync.Pool{
	New: func() interface{} {
		buf := make([]byte, 4096)
		return &buf
	},
}

func BenchmarkPoolUsage(b *testing.B) {
	for i := 0; i < b.N; i++ {
		buf := bufPool.Get().(*[]byte)
		for j := range *buf {
			(*buf)[j] = byte(j)
		}
		bufPool.Put(buf)
	}
}

func BenchmarkNoPoolUsage(b *testing.B) {
	for i := 0; i < b.N; i++ {
		buf := make([]byte, 4096)
		for j := range buf {
			buf[j] = byte(j)
		}
	}
}
