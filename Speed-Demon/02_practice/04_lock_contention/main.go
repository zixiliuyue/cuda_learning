package main

import (
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

var (
	counter    int64
	counterMu  sync.Mutex
	counterRW  sync.RWMutex
)

func main() {
	fmt.Println("Lock Contention Demo 启动在 :6060")
	fmt.Println("锁竞争分析演示:")
	fmt.Println("  /mutex/bad     - 高并发下的 Mutex 竞争")
	fmt.Println("  /mutex/good    - 优化后的版本")
	fmt.Println("  /rwmutex/bad   - RWMutex 使用不当")
	fmt.Println("  /rwmutex/good  - 正确使用 RWMutex")
	fmt.Println("  /atomic        - Atomic 操作")
	fmt.Println("  /stats         - 查看统计")
	
	http.HandleFunc("/mutex/bad", badMutexHandler)
	http.HandleFunc("/mutex/good", goodMutexHandler)
	http.HandleFunc("/rwmutex/bad", badRWMutexHandler)
	http.HandleFunc("/rwmutex/good", goodRWMutexHandler)
	http.HandleFunc("/atomic", atomicHandler)
	http.HandleFunc("/stats", statsHandler)
	
	log.Fatal(http.ListenAndServe(":6060", nil))
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	fmt.Fprintf(w, "===== Stats =====\n")
	fmt.Fprintf(w, "Counter: %d\n", atomic.LoadInt64(&counter))
	fmt.Fprintf(w, "Goroutines: %d\n", runtime.NumGoroutine())
	fmt.Fprintf(w, "\n===== Profiling URLs =====\n")
	fmt.Fprintf(w, "Mutex: http://localhost:6060/debug/pprof/mutex?debug=1\n")
	fmt.Fprintf(w, "Block: http://localhost:6060/debug/pprof/block?debug=1\n")
	fmt.Fprintf(w, "Goroutine: http://localhost:6060/debug/pprof/goroutine?debug=1\n")
}

func badMutexHandler(w http.ResponseWriter, r *http.Request) {
	count := 1000
	
	var wg sync.WaitGroup
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				counterMu.Lock()
				counter++
				counterMu.Unlock()
			}
		}()
	}
	
	wg.Wait()
	fmt.Fprintf(w, "Bad Mutex: Counter = %d\n", counter)
}

func goodMutexHandler(w http.ResponseWriter, r *http.Request) {
	count := 1000
	
	var wg sync.WaitGroup
	localCounters := make([]int64, runtime.NumCPU())
	
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			localIdx := idx % len(localCounters)
			for j := 0; j < 100; j++ {
				atomic.AddInt64(&localCounters[localIdx], 1)
			}
		}(i)
	}
	
	wg.Wait()
	
	var total int64
	for _, c := range localCounters {
		total += c
	}
	atomic.AddInt64(&counter, total)
	
	fmt.Fprintf(w, "Good Mutex: Counter = %d\n", atomic.LoadInt64(&counter))
}

func badRWMutexHandler(w http.ResponseWriter, r *http.Request) {
	count := 100
	
	var wg sync.WaitGroup
	
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < count; j++ {
				counterRW.Lock()
				counter++
				counterRW.Unlock()
			}
		}()
	}
	
	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			counterRW.RLock()
			_ = counter
			counterRW.RUnlock()
		}()
	}
	
	wg.Wait()
	fmt.Fprintf(w, "Bad RWMutex: Counter = %d\n", counter)
}

func goodRWMutexHandler(w http.ResponseWriter, r *http.Request) {
	count := 100
	
	var wg sync.WaitGroup
	
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < count; j++ {
				atomic.AddInt64(&counter, 1)
			}
		}()
	}
	
	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = atomic.LoadInt64(&counter)
		}()
	}
	
	wg.Wait()
	fmt.Fprintf(w, "Good RWMutex: Counter = %d\n", atomic.LoadInt64(&counter))
}

func atomicHandler(w http.ResponseWriter, r *http.Request) {
	count := 10000
	
	var wg sync.WaitGroup
	start := time.Now()
	
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				atomic.AddInt64(&counter, 1)
			}
		}()
	}
	
	wg.Wait()
	
	elapsed := time.Since(start)
	fmt.Fprintf(w, "Atomic: Counter = %d, Time = %v\n", atomic.LoadInt64(&counter), elapsed)
}
