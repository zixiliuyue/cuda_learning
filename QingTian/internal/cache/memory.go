// 本文件提供 Cache / PriorityQueue / Locker 的内存实现。
//
// 内存实现用于单元测试与本地单机演示，通过互斥锁保证并发安全，
// 并模拟 TTL 过期语义。生产环境应使用 Redis 实现。
package cache

import (
	"context"
	"sort"
	"strings"
	"sync"
	"time"
)

// entry 表示一个带过期时间的缓存条目。
type entry struct {
	value     string
	expiresAt time.Time // 零值表示永不过期
}

// expired 判断条目是否已过期。
func (e entry) expired(now time.Time) bool {
	return !e.expiresAt.IsZero() && now.After(e.expiresAt)
}

// MemoryCache 是 Cache 的内存实现。
type MemoryCache struct {
	mu   sync.RWMutex
	data map[string]entry
	// now 用于获取当前时间，测试可替换以模拟过期。
	now func() time.Time
}

// NewMemoryCache 创建内存缓存。
func NewMemoryCache() *MemoryCache {
	return &MemoryCache{data: make(map[string]entry), now: time.Now}
}

// Set 实现 Cache.Set。
func (c *MemoryCache) Set(_ context.Context, key, value string, ttl time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	e := entry{value: value}
	if ttl > 0 {
		e.expiresAt = c.now().Add(ttl)
	}
	c.data[key] = e
	return nil
}

// Get 实现 Cache.Get，读取时惰性清理过期条目。
func (c *MemoryCache) Get(_ context.Context, key string) (string, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.data[key]
	if !ok {
		return "", false, nil
	}
	if e.expired(c.now()) {
		delete(c.data, key)
		return "", false, nil
	}
	return e.value, true, nil
}

// Del 实现 Cache.Del。
func (c *MemoryCache) Del(_ context.Context, key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.data, key)
	return nil
}

// Keys 实现 Cache.Keys，返回匹配前缀且未过期的键。
func (c *MemoryCache) Keys(_ context.Context, prefix string) ([]string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()
	out := make([]string, 0)
	for k, e := range c.data {
		if e.expired(now) {
			delete(c.data, k)
			continue
		}
		if strings.HasPrefix(k, prefix) {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out, nil
}

// pqItem 是优先级队列中的一个元素。
type pqItem struct {
	taskID string
	score  float64
	seq    uint64 // 入队序号，用于同分值时保持 FIFO
}

// MemoryPriorityQueue 是 PriorityQueue 的内存实现。
type MemoryPriorityQueue struct {
	mu    sync.Mutex
	items []pqItem
	seq   uint64
}

// NewMemoryPriorityQueue 创建内存优先级队列。
func NewMemoryPriorityQueue() *MemoryPriorityQueue {
	return &MemoryPriorityQueue{items: make([]pqItem, 0)}
}

// Push 实现 PriorityQueue.Push。
func (q *MemoryPriorityQueue) Push(_ context.Context, taskID string, score float64) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.seq++
	q.items = append(q.items, pqItem{taskID: taskID, score: score, seq: q.seq})
	return nil
}

// Pop 实现 PriorityQueue.Pop。
//
// 选出 score 最大者；score 相同时选择入队序号最小者（FIFO）。
func (q *MemoryPriorityQueue) Pop(_ context.Context) (string, bool, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return "", false, nil
	}
	best := 0
	for i := 1; i < len(q.items); i++ {
		if q.items[i].score > q.items[best].score ||
			(q.items[i].score == q.items[best].score && q.items[i].seq < q.items[best].seq) {
			best = i
		}
	}
	item := q.items[best]
	// 从切片中移除该元素。
	q.items = append(q.items[:best], q.items[best+1:]...)
	return item.taskID, true, nil
}

// Len 实现 PriorityQueue.Len。
func (q *MemoryPriorityQueue) Len(_ context.Context) (int, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items), nil
}

// MemoryLocker 是 Locker 的内存实现。
type MemoryLocker struct {
	mu    sync.Mutex
	locks map[string]time.Time // key -> 过期时间
	now   func() time.Time
}

// NewMemoryLocker 创建内存分布式锁。
func NewMemoryLocker() *MemoryLocker {
	return &MemoryLocker{locks: make(map[string]time.Time), now: time.Now}
}

// Acquire 实现 Locker.Acquire。
func (l *MemoryLocker) Acquire(_ context.Context, key string, ttl time.Duration) (bool, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	if exp, ok := l.locks[key]; ok && now.Before(exp) {
		return false, nil // 锁仍被持有
	}
	l.locks[key] = now.Add(ttl)
	return true, nil
}

// Release 实现 Locker.Release。
func (l *MemoryLocker) Release(_ context.Context, key string) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.locks, key)
	return nil
}
