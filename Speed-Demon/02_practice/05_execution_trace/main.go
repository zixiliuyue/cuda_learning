package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"runtime/trace"
	"sync"
	"sync/atomic"
	"time"
)

var counter int64

func main() {
	fmt.Println("Execution Trace Demo 启动在 :6060")
	fmt.Println("访问 /trace 采集 trace 文件")
	fmt.Println("访问 /demo/bad 运行低效代码")
	fmt.Println("访问 /demo/good 运行高效代码")
	fmt.Println("访问 /save 保存 trace 到文件")
	
	http.HandleFunc("/trace", traceHandler)
	http.HandleFunc("/save", saveTraceHandler)
	http.HandleFunc("/demo/bad", badDemoHandler)
	http.HandleFunc("/demo/good", goodDemoHandler)
	http.HandleFunc("/stats", statsHandler)
	
	log.Fatal(http.ListenAndServe(":6060", nil))
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	fmt.Fprintf(w, "===== Stats =====\n")
	fmt.Fprintf(w, "Goroutines: %d\n", runtime.NumGoroutine())
	fmt.Fprintf(w, "Counter: %d\n", atomic.LoadInt64(&counter))
	fmt.Fprintf(w, "\n===== Trace Commands =====\n")
	fmt.Fprintf(w, "go tool trace trace.out\n")
	fmt.Fprintf(w, "\n视图说明:\n")
	fmt.Fprintf(w, "  - View trace: 时间轴视图\n")
	fmt.Fprintf(w, "  - Goroutine analysis: Goroutine 分析\n")
	fmt.Fprintf(w, "  - Network/Sync blocking profile: 阻塞分析\n")
	fmt.Fprintf(w, "  - Minimum mutator utilization: GC 影响分析\n")
}

func traceHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Disposition", "attachment; filename=\"trace.out\"")
	
	if err := trace.Start(w); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer trace.Stop()
	
	duration := 5 * time.Second
	if d := r.URL.Query().Get("duration"); d != "" {
		if parsed, err := time.ParseDuration(d); err == nil {
			duration = parsed
		}
	}
	
	fmt.Fprintf(os.Stderr, "采集 %v trace...\n", duration)
	
	ctx, task := trace.NewTask(context.Background(), "DemoTask")
	defer task.End()
	
	region := trace.StartRegion(ctx, "DemoRegion")
	defer region.End()
	
	for i := 0; i < 10; i++ {
		trace.WithRegion(ctx, "LoopIteration", func() {
			runWorkers(100)
		})
	}
	
	time.Sleep(duration)
}

func saveTraceHandler(w http.ResponseWriter, r *http.Request) {
	f, err := os.Create("trace.out")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()
	
	if err := trace.Start(f); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer trace.Stop()
	
	fmt.Println("采集 10 秒 trace...")
	
	ctx, task := trace.NewTask(context.Background(), "SaveTraceTask")
	defer task.End()
	
	for i := 0; i < 5; i++ {
		trace.WithRegion(ctx, "WorkBatch", func() {
			runWorkers(200)
		})
		time.Sleep(100 * time.Millisecond)
	}
	
	fmt.Fprintf(w, "Trace 已保存到 trace.out\n")
	fmt.Fprintf(w, "运行: go tool trace trace.out\n")
}

func badDemoHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	
	var wg sync.WaitGroup
	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				atomic.AddInt64(&counter, 1)
			}
		}()
	}
	wg.Wait()
	
	elapsed := time.Since(start)
	fmt.Fprintf(w, "Bad Demo: Counter = %d, Time = %v\n", atomic.LoadInt64(&counter), elapsed)
}

func goodDemoHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	
	numCPU := runtime.NumCPU()
	localCounters := make([]int64, numCPU)
	
	var wg sync.WaitGroup
	for i := 0; i < numCPU; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			for j := 0; j < 1000*1000/numCPU; j++ {
				localCounters[idx]++
			}
		}(i)
	}
	wg.Wait()
	
	var total int64
	for _, c := range localCounters {
		total += c
	}
	atomic.AddInt64(&counter, total)
	
	elapsed := time.Since(start)
	fmt.Fprintf(w, "Good Demo: Counter = %d, Time = %v\n", atomic.LoadInt64(&counter), elapsed)
}

func runWorkers(count int) {
	var wg sync.WaitGroup
	
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			for j := 0; j < 100; j++ {
				atomic.AddInt64(&counter, 1)
			}
			
			if id%10 == 0 {
				time.Sleep(1 * time.Millisecond)
			}
		}(i)
	}
	
	wg.Wait()
}
