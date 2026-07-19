package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"time"
)

var (
	mode        = flag.String("mode", "server", "运行模式: server | leak | bench")
	heapProfile = flag.String("heapprofile", "", "写入 heap profile 到文件")
	allocProfile = flag.String("allocprofile", "", "写入 allocation profile 到文件")
)

func main() {
	flag.Parse()

	switch *mode {
	case "server":
		runServer()
	case "leak":
		runLeakDemo()
	case "bench":
		runBenchmark()
	default:
		fmt.Println("未知模式，使用 -h 查看帮助")
	}
}

func runServer() {
	fmt.Println("Memory Profiling 演示服务启动在 :6060")
	fmt.Println("访问以下地址进行性能分析:")
	fmt.Println("  http://localhost:6060/debug/pprof/heap")
	fmt.Println("  http://localhost:6060/debug/pprof/allocs")

	http.HandleFunc("/alloc", allocHandler)
	http.HandleFunc("/leak", leakHandler)
	http.HandleFunc("/pool", poolHandler)
	http.HandleFunc("/stats", statsHandler)
	http.HandleFunc("/large", largeAllocHandler)

	log.Fatal(http.ListenAndServe(":6060", nil))
}

func runLeakDemo() {
	fmt.Println("运行内存泄漏演示，按 Ctrl+C 停止...")
	
	if *heapProfile != "" {
		defer func() {
			f, err := os.Create(*heapProfile)
			if err != nil {
				log.Fatal(err)
			}
			defer f.Close()
			runtime.GC()
			pprof.WriteHeapProfile(f)
			fmt.Println("Heap profile 已写入", *heapProfile)
		}()
	}

	leakData := make([][]byte, 0)
	
	for i := 0; i < 100; i++ {
		buf := make([]byte, 1024*1024)
		for j := range buf {
			buf[j] = byte(i)
		}
		leakData = append(leakData, buf)
		printMemStats()
		time.Sleep(500 * time.Millisecond)
	}
	
	fmt.Printf("已分配 %d MB 内存\n", len(leakData))
	fmt.Println("这些内存不会被释放，因为 leakData 持有引用...")
	select {}
}

func runBenchmark() {
	fmt.Println("运行内存基准测试...")
	
	tests := []struct {
		name string
		fn   func()
	}{
		{"NaiveConcat", func() { naiveConcat(1000) }},
		{"BuilderConcat", func() { builderConcat(1000) }},
		{"NoPool", func() { noPool(1000) }},
		{"WithPool", func() { withPool(1000) }},
		{"AppendNoCap", func() { appendNoCap(10000) }},
		{"AppendWithCap", func() { appendWithCap(10000) }},
	}
	
	for _, test := range tests {
		runtime.GC()
		var before, after runtime.MemStats
		
		runtime.ReadMemStats(&before)
		start := time.Now()
		iterations := 0
		
		for time.Since(start) < 2*time.Second {
			test.fn()
			iterations++
		}
		
		runtime.GC()
		runtime.ReadMemStats(&after)
		
		elapsed := time.Since(start)
		perOp := float64(elapsed.Nanoseconds()) / float64(iterations)
		allocs := (after.Mallocs - before.Mallocs) / uint64(iterations)
		allocBytes := (after.TotalAlloc - before.TotalAlloc) / uint64(iterations)
		
		fmt.Printf("%-20s %8d 次  %10.2f ns/op  %4d allocs/op  %6d B/op\n",
			test.name, iterations, perOp, allocs, allocBytes)
	}
}

func allocHandler(w http.ResponseWriter, r *http.Request) {
	data := naiveConcat(1000)
	fmt.Fprintf(w, "Generated string length: %d\n", len(data))
}

func leakHandler(w http.ResponseWriter, r *http.Request) {
	buf := make([]byte, 1024*1024)
	globalCache = append(globalCache, buf)
	fmt.Fprintf(w, "Leaked 1MB, total leaked: %d MB\n", len(globalCache))
}

var globalCache [][]byte

func poolHandler(w http.ResponseWriter, r *http.Request) {
	result := withPool(1000)
	fmt.Fprintf(w, "Pool result length: %d\n", len(result))
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	fmt.Fprintf(w, "===== Memory Stats =====\n")
	fmt.Fprintf(w, "Alloc: %d MB\n", m.Alloc/1024/1024)
	fmt.Fprintf(w, "TotalAlloc: %d MB\n", m.TotalAlloc/1024/1024)
	fmt.Fprintf(w, "Sys: %d MB\n", m.Sys/1024/1024)
	fmt.Fprintf(w, "Lookups: %d\n", m.Lookups)
	fmt.Fprintf(w, "Mallocs: %d\n", m.Mallocs)
	fmt.Fprintf(w, "Frees: %d\n", m.Frees)
	fmt.Fprintf(w, "HeapAlloc: %d MB\n", m.HeapAlloc/1024/1024)
	fmt.Fprintf(w, "HeapSys: %d MB\n", m.HeapSys/1024/1024)
	fmt.Fprintf(w, "HeapIdle: %d MB\n", m.HeapIdle/1024/1024)
	fmt.Fprintf(w, "HeapInuse: %d MB\n", m.HeapInuse/1024/1024)
	fmt.Fprintf(w, "HeapReleased: %d MB\n", m.HeapReleased/1024/1024)
	fmt.Fprintf(w, "HeapObjects: %d\n", m.HeapObjects)
	fmt.Fprintf(w, "NumGC: %d\n", m.NumGC)
}

func largeAllocHandler(w http.ResponseWriter, r *http.Request) {
	size := 32 * 1024 * 1024
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i % 256)
	}
	fmt.Fprintf(w, "Allocated %d MB\n", size/1024/1024)
}

func naiveConcat(n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += "hello "
	}
	return result
}

func builderConcat(n int) string {
	var builder strings.Builder
	builder.Grow(n * 6)
	for i := 0; i < n; i++ {
		builder.WriteString("hello ")
	}
	return builder.String()
}

func noPool(n int) [][]byte {
	result := make([][]byte, n)
	for i := 0; i < n; i++ {
		buf := make([]byte, 4096)
		for j := range buf {
			buf[j] = byte(i)
		}
		result[i] = buf
	}
	return result
}

var bufPool = sync.Pool{
	New: func() interface{} {
		return make([]byte, 4096)
	},
}

func withPool(n int) [][]byte {
	result := make([][]byte, n)
	for i := 0; i < n; i++ {
		buf := bufPool.Get().([]byte)
		for j := range buf {
			buf[j] = byte(i)
		}
		result[i] = buf
	}
	for _, buf := range result {
		bufPool.Put(buf)
	}
	return result
}

func appendNoCap(n int) []int {
	var result []int
	for i := 0; i < n; i++ {
		result = append(result, i)
	}
	return result
}

func appendWithCap(n int) []int {
	result := make([]int, 0, n)
	for i := 0; i < n; i++ {
		result = append(result, i)
	}
	return result
}

func printMemStats() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Alloc=%5dMB HeapObjects=%7d NumGC=%3d\n",
		m.Alloc/1024/1024,
		m.HeapObjects,
		m.NumGC)
}

func readLines(filename string) ([]string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}
