package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"strconv"
	"sync"
	"time"
)

type Product struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Price int    `json:"price"`
	Stock int    `json:"stock"`
}

// ============================================================
// 问题 1: 全局大锁！锁里还做慢操作！
var cacheLock sync.Mutex
var localCache = make(map[int]*Product)

func GetProductBad(id int) (*Product, error) {
	cacheLock.Lock() // ❌ 进来就上锁！
	defer cacheLock.Unlock()

	p, ok := localCache[id]
	if ok {
		return p, nil
	}

	// ❌ 慢操作在锁里面！！！
	time.Sleep(20 * time.Millisecond) // 模拟 DB 查询延迟
	p = &Product{
		ID:    id,
		Name:  "Product " + strconv.Itoa(id),
		Price: 99,
		Stock: 100,
	}
	localCache[id] = p
	return p, nil
}

// ============================================================
// 问题 2: channel 没缓冲，超时导致 goroutine 泄漏！
func CheckStockBad(ctx context.Context, id int) int {
	ch := make(chan int) // ❌ 无缓冲！

	go func() {
		time.Sleep(30 * time.Millisecond) // 模拟 RPC 延迟
		ch <- rand.Intn(100)
	}()

	select {
	case stock := <-ch:
		return stock
	case <-ctx.Done(): // ❌ 超时了，但是 goroutine 写不进去！
		return 0
	}
}

// ============================================================
// 问题 3: 每次都 json.Marshal，大量分配！
func (p *Product) ToJSONBad() []byte {
	data, _ := json.Marshal(p)
	return data
}

// ============================================================
// 问题 4: 没设置 DB 连接池！
func queryFromDB(id int) *Product {
	time.Sleep(20 * time.Millisecond)
	return &Product{
		ID:    id,
		Name:  "Product " + strconv.Itoa(id),
		Price: 99,
		Stock: 100,
	}
}

// ============================================================
// 问题 5: 没有 singleflight，相同 id 并发查 N 次 DB！
func productHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	idStr := r.URL.Query().Get("id")
	id, _ := strconv.Atoi(idStr)
	if id == 0 {
		id = 1
	}

	ctx, cancel := context.WithTimeout(r.Context(), 100*time.Millisecond)
	defer cancel()

	// 查商品
	p, err := GetProductBad(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// 查库存
	p.Stock = CheckStockBad(ctx, id)

	// 序列化
	data := p.ToJSONBad()

	w.Header().Set("Content-Type", "application/json")
	w.Write(data)

	log.Printf("请求耗时: %vms, goroutines: %d",
		time.Since(start).Milliseconds(), runtime.NumGoroutine())
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	fmt.Fprintf(w, "Goroutines: %d\n", runtime.NumGoroutine())
	fmt.Fprintf(w, "Alloc: %d MB\n", m.Alloc/1024/1024)
	fmt.Fprintf(w, "HeapObjects: %d\n", m.HeapObjects)
	fmt.Fprintf(w, "NumGC: %d\n", m.NumGC)
}

func main() {
	http.HandleFunc("/api/product", productHandler)
	http.HandleFunc("/stats", statsHandler)

	fmt.Println("Server started on :8080")
	fmt.Println("压测命令: hey -n 3000 -c 100 http://localhost:8080/api/product?id=123")
	fmt.Println("观察指标: curl http://localhost:8080/stats")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
