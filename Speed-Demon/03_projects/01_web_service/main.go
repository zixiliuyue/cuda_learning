package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"strconv"
	"sync"
	"time"
)

type User struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}

var (
	userCache = make(map[int]*User)
	cacheMu   sync.RWMutex
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	
	mux := http.NewServeMux()
	
	mux.HandleFunc("/user/bad", badUserHandler)
	mux.HandleFunc("/user/good", goodUserHandler)
	mux.HandleFunc("/user/batch/bad", badBatchHandler)
	mux.HandleFunc("/user/batch/good", goodBatchHandler)
	mux.HandleFunc("/stats", statsHandler)
	
	log.Println("Web Service Demo 启动在 :6060")
	log.Fatal(http.ListenAndServe(":6060", mux))
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	cacheMu.RLock()
	cacheSize := len(userCache)
	cacheMu.RUnlock()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"goroutines":    runtime.NumGoroutine(),
		"heap_alloc_mb": m.HeapAlloc / 1024 / 1024,
		"num_gc":        m.NumGC,
		"cache_size":    cacheSize,
	})
}

func badUserHandler(w http.ResponseWriter, r *http.Request) {
	idStr := r.URL.Query().Get("id")
	id, _ := strconv.Atoi(idStr)
	
	user := &User{
		ID:   id,
		Name: "User " + strconv.Itoa(id),
	}
	
	cacheMu.Lock()
	userCache[id] = user
	cacheMu.Unlock()
	
	time.Sleep(10 * time.Millisecond)
	
	result := make([]byte, 1024)
	for i := range result {
		result[i] = byte(i % 256)
	}
	
	json.NewEncoder(w).Encode(user)
}

func goodUserHandler(w http.ResponseWriter, r *http.Request) {
	idStr := r.URL.Query().Get("id")
	id, _ := strconv.Atoi(idStr)
	
	ctx := r.Context()
	
	user, err := getUser(ctx, id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	pooledBuf := bufPool.Get().(*[]byte)
	defer bufPool.Put(pooledBuf)
	
	json.NewEncoder(w).Encode(user)
}

var bufPool = sync.Pool{
	New: func() interface{} {
		b := make([]byte, 1024)
		return &b
	},
}

func getUser(ctx context.Context, id int) (*User, error) {
	cacheMu.RLock()
	user, ok := userCache[id]
	cacheMu.RUnlock()
	
	if ok {
		return user, nil
	}
	
	user = &User{
		ID:   id,
		Name: "User " + strconv.Itoa(id),
	}
	
	cacheMu.Lock()
	if len(userCache) > 1000 {
		for k := range userCache {
			delete(userCache, k)
			if len(userCache) <= 500 {
				break
			}
		}
	}
	userCache[id] = user
	cacheMu.Unlock()
	
	return user, nil
}

func badBatchHandler(w http.ResponseWriter, r *http.Request) {
	countStr := r.URL.Query().Get("count")
	count, _ := strconv.Atoi(countStr)
	if count == 0 {
		count = 100
	}
	
	var wg sync.WaitGroup
	results := make([]*User, count)
	
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = &User{
				ID:   idx,
				Name: "User " + strconv.Itoa(idx),
			}
			time.Sleep(1 * time.Millisecond)
		}(i)
	}
	
	wg.Wait()
	
	json.NewEncoder(w).Encode(results)
}

func goodBatchHandler(w http.ResponseWriter, r *http.Request) {
	countStr := r.URL.Query().Get("count")
	count, _ := strconv.Atoi(countStr)
	if count == 0 {
		count = 100
	}
	
	const numWorkers = 10
	work := make(chan int, count)
	results := make([]*User, count)
	
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range work {
				results[idx] = &User{
					ID:   idx,
					Name: "User " + strconv.Itoa(idx),
				}
				time.Sleep(1 * time.Millisecond)
			}
		}()
	}
	
	for i := 0; i < count; i++ {
		work <- i
	}
	close(work)
	
	wg.Wait()
	
	json.NewEncoder(w).Encode(results)
}
