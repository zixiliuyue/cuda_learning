package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
)

var (
	mode     = flag.String("mode", "server", "运行模式: server | cli | bench")
	duration = flag.Int("duration", 30, "运行时长（秒）")
	cpuprofile = flag.String("cpuprofile", "", "写入 CPU profile 到文件")
)

func main() {
	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	switch *mode {
	case "server":
		runServer()
	case "cli":
		runCLI()
	case "bench":
		runBenchmark()
	default:
		fmt.Println("未知模式，使用 -h 查看帮助")
	}
}

func runServer() {
	fmt.Println("CPU Profiling 演示服务启动在 :6060")
	fmt.Println("访问以下地址进行性能分析:")
	fmt.Println("  http://localhost:6060/debug/pprof/")
	fmt.Println("  http://localhost:6060/debug/pprof/profile?seconds=10")

	http.HandleFunc("/heavy", heavyHandler)
	http.HandleFunc("/fib", fibHandler)
	http.HandleFunc("/prime", primeHandler)
	http.HandleFunc("/matrix", matrixHandler)

	log.Fatal(http.ListenAndServe(":6060", nil))
}

func runCLI() {
	fmt.Printf("运行 CPU 密集型任务，持续 %d 秒...\n", *duration)
	
	end := time.Now().Add(time.Duration(*duration) * time.Second)
	counter := 0
	
	for time.Now().Before(end) {
		heavyComputation(100000)
		fibonacci(35)
		findPrimes(10000)
		counter++
	}
	
	fmt.Printf("完成 %d 轮计算\n", counter)
}

func runBenchmark() {
	fmt.Println("运行基准测试...")
	
	tests := []struct {
		name string
		fn   func()
	}{
		{"HeavyComputation", func() { heavyComputation(1000) }},
		{"Fibonacci(30)", func() { fibonacci(30) }},
		{"FindPrimes(1000)", func() { findPrimes(1000) }},
		{"MatrixMultiply(100)", func() { matrixMultiply(100) }},
	}
	
	for _, test := range tests {
		start := time.Now()
		iterations := 0
		
		for time.Since(start) < 2*time.Second {
			test.fn()
			iterations++
		}
		
		elapsed := time.Since(start)
		perOp := float64(elapsed.Nanoseconds()) / float64(iterations)
		fmt.Printf("%-25s %8d 次  %10.2f ns/op\n", 
			test.name, iterations, perOp)
	}
}

func heavyHandler(w http.ResponseWriter, r *http.Request) {
	result := heavyComputation(50000)
	fmt.Fprintf(w, "Heavy computation result: %f\n", result)
}

func fibHandler(w http.ResponseWriter, r *http.Request) {
	result := fibonacci(35)
	fmt.Fprintf(w, "Fibonacci(35) = %d\n", result)
}

func primeHandler(w http.ResponseWriter, r *http.Request) {
	result := findPrimes(10000)
	fmt.Fprintf(w, "Found %d primes below 10000\n", len(result))
}

func matrixHandler(w http.ResponseWriter, r *http.Request) {
	result := matrixMultiply(100)
	fmt.Fprintf(w, "Matrix multiply result[0][0] = %f\n", result[0][0])
}

func heavyComputation(n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		for j := 0; j < 100; j++ {
			sum += math.Sin(float64(i*j)) * math.Cos(float64(i+j))
			sum += math.Sqrt(float64(i*i + j*j))
		}
	}
	return sum
}

func fibonacci(n int) int {
	if n <= 1 {
		return n
	}
	return fibonacci(n-1) + fibonacci(n-2)
}

func findPrimes(n int) []int {
	var primes []int
	for i := 2; i < n; i++ {
		if isPrime(i) {
			primes = append(primes, i)
		}
	}
	return primes
}

func isPrime(n int) bool {
	if n <= 1 {
		return false
	}
	for i := 2; i*i <= n; i++ {
		if n%i == 0 {
			return false
		}
	}
	return true
}

func matrixMultiply(size int) [][]float64 {
	a := make([][]float64, size)
	b := make([][]float64, size)
	
	for i := range a {
		a[i] = make([]float64, size)
		b[i] = make([]float64, size)
		for j := range a[i] {
			a[i][j] = rand.Float64()
			b[i][j] = rand.Float64()
		}
	}
	
	result := make([][]float64, size)
	for i := range result {
		result[i] = make([]float64, size)
		for j := range result[i] {
			for k := 0; k < size; k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	
	return result
}

func printMemStats() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Alloc = %v MiB", bToMb(m.Alloc))
	fmt.Printf("\tTotalAlloc = %v MiB", bToMb(m.TotalAlloc))
	fmt.Printf("\tSys = %v MiB", bToMb(m.Sys))
	fmt.Printf("\tNumGC = %v\n", m.NumGC)
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}
