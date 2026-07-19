package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"strconv"
	"sync"
	"time"

	jsoniter "github.com/json-iterator/go"
	"golang.org/x/sync/singleflight"
)

type Product struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Price int    `json:"price"`
	Stock int    `json:"stock"`
}

// ============================================================
// 优化 1: 读写锁 + DB 操作在锁外面！
var cacheLock sync.RWMutex
var localCache = make(map[int]*Product)

var sf singleflight.Group // 相同 id 的请求合并！

func GetProductGood(id int) (*Product, error) {
	// 1. 先读缓存（读锁，并发可以进）
	cacheLock.RLock()
	p, ok := localCache[id]
	cacheLock.RUnlock()
	if ok {
		return p, nil
	}

	// 2. Singleflight 合并相同请求！
	result, err, _ := sf.Do(strconv.Itoa(id), func() (interface{}, error) {
		// 3. 没命中才去 DB（锁外面！！！）
		time.Sleep(20 * time.Millisecond) // 模拟 DB 查询
		p := &Product{
			ID:    id,
			Name:  "Product " + strconv.Itoa(id),
			Price: 99,
			Stock: 100,
		}

		// 4. 写回缓存（只锁写操作）
		cacheLock.Lock()
		localCache[id] = p
		cacheLock.Unlock()

		return p, nil
	})

	if err != nil {
		return nil, err
	}
	return result.(*Product), nil
}

// ============================================================
// 优化 2: channel 加缓冲，超时不会泄露 goroutine！
func CheckStockGood(ctx context.Context, id int) int {
	ch := make(chan int, 1) // ✅ 有缓冲！写完直接退出！

	go func() {
		time.Sleep(30 * time.Millisecond)
		ch <- rand.Intn(100)
	}()

	select {
	case stock := <-ch:
		return stock
	case <-ctx.Done():
		return 0
	}
}

// ============================================================
// 优化 3: 用更快的 json 库 + 对象池复用 buffer！
var bufPool = sync.Pool{
	New: func() interface{} {
		b := make([]byte, 0, 1024)
		return &b
	},
}

var json = jsoniter.ConfigFastest

func (p *Product) ToJSONGood() []byte {
	bufPtr := bufPool.Get().(*[]byte)
	buf := *bufPtr
	buf = buf[:0] // 重置

	defer func() {
		bufPool.Put(&buf) // 用完放回去
	}()

	data, err := json.Marshal(p)
	if err != nil {
		return nil
	}
	return data
}

// ============================================================
// 接口主逻辑
func productHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	idStr := r.URL.Query().Get("id")
	id, _ := strconv.Atoi(idStr)
	if id == 0 {
		id = 1
	}

	ctx, cancel := context.WithTimeout(r.Context(), 100*time.Millisecond)
	defer cancel()

	// 查商品（加了读写锁 + singleflight）
	p, err := GetProductGood(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// 查库存（channel 加了缓冲）
	p.Stock = CheckStockGood(ctx, id)

	// 序列化（用了对象池 + fast json）
	data := p.ToJSONGood()

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

	fmt.Println("优化后版本启动在 :8080")
	fmt.Println("压测命令: hey -n 10000 -c 100 http://localhost:8080/api/product?id=123")
	fmt.Println("观察指标: curl http://localhost:8080/stats")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
