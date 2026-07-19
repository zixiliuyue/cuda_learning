package main

import (
	"context"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/sync/singleflight"
)

// ============================================================
// Prometheus Metrics 定义
// ============================================================

var (
	// http_requests_total: HTTP 请求总数
	HttpRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"path", "method", "status"},
	)

	// http_request_duration_seconds: HTTP 请求延迟直方图
	HttpRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request latency distributions",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"path", "method"},
	)

	// http_request_in_flight: 正在处理的请求数
	HttpRequestInFlight = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "http_request_in_flight",
			Help: "Current number of in-flight HTTP requests",
		},
	)

	// cache_hit_total: 缓存命中次数
	CacheHitTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "cache_hit_total",
			Help: "Total number of cache hits",
		},
	)

	// cache_miss_total: 缓存未命中次数
	CacheMissTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "cache_miss_total",
			Help: "Total number of cache misses",
		},
	)

	// db_query_duration_seconds: DB 查询延迟
	DbQueryDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "db_query_duration_seconds",
			Help:    "DB query latency distributions",
			Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1},
		},
		[]string{"operation"},
	)

	// goroutine_leak_detector: Goroutine 泄露检测（模拟）
	GoroutineLeakDetector = prometheus.NewGaugeFunc(
		prometheus.GaugeOpts{
			Name: "goroutine_leak_risk",
			Help: "Indicator of potential goroutine leak risk",
		},
		func() float64 {
			return 0
		},
	)
)

func init() {
	prometheus.MustRegister(HttpRequestsTotal)
	prometheus.MustRegister(HttpRequestDuration)
	prometheus.MustRegister(HttpRequestInFlight)
	prometheus.MustRegister(CacheHitTotal)
	prometheus.MustRegister(CacheMissTotal)
	prometheus.MustRegister(DbQueryDuration)
	prometheus.MustRegister(GoroutineLeakDetector)
}

// ============================================================
// Metrics 中间件
// ============================================================

type MetricsResponseWriter struct {
	http.ResponseWriter
	statusCode int
}

func NewMetricsResponseWriter(w http.ResponseWriter) *MetricsResponseWriter {
	return &MetricsResponseWriter{w, http.StatusOK}
}

func (m *MetricsResponseWriter) WriteHeader(code int) {
	m.statusCode = code
	m.ResponseWriter.WriteHeader(code)
}

func MetricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		HttpRequestInFlight.Inc()
		defer HttpRequestInFlight.Dec()

		mw := NewMetricsResponseWriter(w)

		next.ServeHTTP(mw, r)

		duration := time.Since(start).Seconds()
		path := normalizePath(r.URL.Path)
		status := strconv.Itoa(mw.statusCode)

		HttpRequestsTotal.WithLabelValues(path, r.Method, status).Inc()
		HttpRequestDuration.WithLabelValues(path, r.Method).Observe(duration)
	})
}

func normalizePath(path string) string {
	if strings.HasPrefix(path, "/api/product") {
		return "/api/product"
	}
	if strings.HasPrefix(path, "/debug/pprof") {
		return "/debug/pprof"
	}
	return path
}

// ============================================================
// 业务逻辑（带 Metrics）
// ============================================================

type Product struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Price int    `json:"price"`
	Stock int    `json:"stock"`
}

var (
	cache       = make(map[int]*Product)
	singleGroup singleflight.Group
)

func getProductFromDB(ctx context.Context, id int) (*Product, error) {
	start := time.Now()
	defer func() {
		duration := time.Since(start).Seconds()
		DbQueryDuration.WithLabelValues("select").Observe(duration)
	}()

	time.Sleep(20 * time.Millisecond)
	return &Product{
		ID:    id,
		Name:  "Product " + strconv.Itoa(id),
		Price: 99,
		Stock: 100,
	}, nil
}

func getProductFromCache(id int) (*Product, bool) {
	p, ok := cache[id]
	if ok {
		CacheHitTotal.Inc()
		return p, true
	}
	CacheMissTotal.Inc()
	return nil, false
}

func productHandler(w http.ResponseWriter, r *http.Request) {
	idStr := r.URL.Query().Get("id")
	id, _ := strconv.Atoi(idStr)
	if id == 0 {
		id = 1
	}

	if p, ok := getProductFromCache(id); ok {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":` + strconv.Itoa(p.ID) + `,"name":"` + p.Name + `","price":` + strconv.Itoa(p.Price) + `,"stock":` + strconv.Itoa(p.Stock) + `}`))
		return
	}

	result, err, _ := singleGroup.Do(strconv.Itoa(id), func() (interface{}, error) {
		return getProductFromDB(r.Context(), id)
	})

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	p := result.(*Product)
	cache[id] = p

	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"id":` + strconv.Itoa(p.ID) + `,"name":"` + p.Name + `","price":` + strconv.Itoa(p.Price) + `,"stock":` + strconv.Itoa(p.Stock) + `}`))
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	w.Write([]byte("Prometheus Metrics Example\n"))
	w.Write([]byte("=============================\n\n"))
	w.Write([]byte("Metrics available at /metrics\n\n"))
	w.Write([]byte("API endpoints:\n"))
	w.Write([]byte("  - /api/product?id=123\n"))
	w.Write([]byte("  - /metrics\n"))
}

// ============================================================
// Main
// ============================================================

func main() {
	mux := http.NewServeMux()

	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/api/product", productHandler)
	mux.HandleFunc("/stats", statsHandler)

	handler := MetricsMiddleware(mux)

	log.Println("Metrics Example Server starting on :8080")
	log.Println("Metrics endpoint: http://localhost:8080/metrics")
	log.Println("API endpoint: http://localhost:8080/api/product?id=123")
	log.Fatal(http.ListenAndServe(":8080", handler))
}
