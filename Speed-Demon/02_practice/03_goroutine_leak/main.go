package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"sync"
	"time"
)

func main() {
	fmt.Println("Goroutine Leak 演示服务启动在 :6060")
	fmt.Println("当前 goroutine 数量:", runtime.NumGoroutine())

	http.HandleFunc("/leak1", leakHandler1)
	http.HandleFunc("/leak2", leakHandler2)
	http.HandleFunc("/leak3", leakHandler3)
	http.HandleFunc("/leak4", leakHandler4)
	http.HandleFunc("/good1", goodHandler1)
	http.HandleFunc("/good2", goodHandler2)
	http.HandleFunc("/good3", goodHandler3)
	http.HandleFunc("/stats", statsHandler)

	log.Fatal(http.ListenAndServe(":6060", nil))
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "当前 goroutine 数量: %d\n", runtime.NumGoroutine())
}

func leakHandler1(w http.ResponseWriter, r *http.Request) {
	ch := make(chan int)
	
	go func() {
		val := <-ch
		fmt.Println("Got:", val)
	}()
	
	fmt.Fprintf(w, "已启动泄漏 goroutine (永久阻塞在 channel 接收)\n")
	fmt.Fprintf(w, "当前 goroutine 数量: %d\n", runtime.NumGoroutine())
}

func goodHandler1(w http.ResponseWriter, r *http.Request) {
	ch := make(chan int)
	
	go func() {
		select {
		case val := <-ch:
			fmt.Println("Got:", val)
		case <-time.After(1 * time.Second):
			fmt.Println("Timeout")
		}
	}()
	
	ch <- 42
	fmt.Fprintf(w, "正常模式，不会泄漏\n")
}

func leakHandler2(w http.ResponseWriter, r *http.Request) {
	timer := time.NewTimer(5 * time.Second)
	done := make(chan struct{})
	
	go func() {
		select {
		case <-timer.C:
			fmt.Println("Timeout")
		case <-done:
			fmt.Println("Done")
		}
	}()
	
	close(done)
	
	fmt.Fprintf(w, "泄漏: timer 没停止，goroutine 可能泄漏\n")
	fmt.Fprintf(w, "当前 goroutine 数量: %d\n", runtime.NumGoroutine())
}

func goodHandler2(w http.ResponseWriter, r *http.Request) {
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	
	done := make(chan struct{})
	
	go func() {
		select {
		case <-timer.C:
			fmt.Println("Timeout")
		case <-done:
			fmt.Println("Done")
		}
	}()
	
	close(done)
	
	fmt.Fprintf(w, "正确停止 timer，不会泄漏\n")
}

func leakHandler3(w http.ResponseWriter, r *http.Request) {
	for i := 0; i < 10; i++ {
		go func(id int) {
			for {
				time.Sleep(1 * time.Second)
			}
		}(i)
	}
	
	fmt.Fprintf(w, "启动了 10 个无限循环的 goroutine\n")
	fmt.Fprintf(w, "当前 goroutine 数量: %d\n", runtime.NumGoroutine())
}

func goodHandler3(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithCancel(context.Background())
	
	for i := 0; i < 10; i++ {
		go func(id int) {
			for {
				select {
				case <-ctx.Done():
					fmt.Printf("Goroutine %d exiting\n", id)
					return
				default:
					time.Sleep(1 * time.Second)
				}
			}
		}(i)
	}
	
	go func() {
		time.Sleep(3 * time.Second)
		cancel()
	}()
	
	fmt.Fprintf(w, "启动了 10 个 goroutine，3秒后会自动退出\n")
	fmt.Fprintf(w, "当前 goroutine 数量: %d\n", runtime.NumGoroutine())
}

func leakHandler4(w http.ResponseWriter, r *http.Request) {
	var wg sync.WaitGroup
	
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			if id == 3 {
				return
			}
			select {}
		}(i)
	}
	
	go func() {
		wg.Wait()
		fmt.Println("All done")
	}()
	
	fmt.Fprintf(w, "泄漏: 某些 goroutine 永远阻塞\n")
	fmt.Fprintf(w, "当前 goroutine 数量: %d\n", runtime.NumGoroutine())
}
