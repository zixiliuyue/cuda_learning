package cache

import (
	"context"
	"testing"
	"time"
)

// TestMemoryCacheSetGet 验证基本读写。
func TestMemoryCacheSetGet(t *testing.T) {
	c := NewMemoryCache()
	ctx := context.Background()
	if err := c.Set(ctx, "k1", "v1", 0); err != nil {
		t.Fatal(err)
	}
	v, ok, _ := c.Get(ctx, "k1")
	if !ok || v != "v1" {
		t.Errorf("读取应返回 v1，实际 ok=%v v=%s", ok, v)
	}
	// 不存在的键。
	if _, ok, _ := c.Get(ctx, "none"); ok {
		t.Error("不存在的键应返回 ok=false")
	}
}

// TestMemoryCacheTTL 验证 TTL 过期（通过注入时间函数模拟）。
func TestMemoryCacheTTL(t *testing.T) {
	c := NewMemoryCache()
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	current := base
	c.now = func() time.Time { return current }

	ctx := context.Background()
	_ = c.Set(ctx, "k", "v", 10*time.Second)
	// 未过期。
	if _, ok, _ := c.Get(ctx, "k"); !ok {
		t.Error("未到期应可读取")
	}
	// 快进 11 秒。
	current = base.Add(11 * time.Second)
	if _, ok, _ := c.Get(ctx, "k"); ok {
		t.Error("过期后不应可读取")
	}
}

// TestPriorityQueueOrder 验证按优先级出队，同优先级 FIFO。
func TestPriorityQueueOrder(t *testing.T) {
	q := NewMemoryPriorityQueue()
	ctx := context.Background()
	_ = q.Push(ctx, "low", 1)
	_ = q.Push(ctx, "high", 10)
	_ = q.Push(ctx, "mid1", 5)
	_ = q.Push(ctx, "mid2", 5) // 与 mid1 同分，应后出

	order := []string{"high", "mid1", "mid2", "low"}
	for _, want := range order {
		got, ok, _ := q.Pop(ctx)
		if !ok || got != want {
			t.Errorf("出队顺序错误，期望 %s，实际 %s(ok=%v)", want, got, ok)
		}
	}
	// 队列应已空。
	if _, ok, _ := q.Pop(ctx); ok {
		t.Error("队列应已空")
	}
}

// TestLockerMutualExclusion 验证锁互斥与释放。
func TestLockerMutualExclusion(t *testing.T) {
	l := NewMemoryLocker()
	ctx := context.Background()
	ok, _ := l.Acquire(ctx, "key", time.Minute)
	if !ok {
		t.Fatal("首次获取锁应成功")
	}
	// 未释放前再次获取应失败。
	if ok2, _ := l.Acquire(ctx, "key", time.Minute); ok2 {
		t.Error("锁被持有时不应再次获取成功")
	}
	// 释放后可再次获取。
	_ = l.Release(ctx, "key")
	if ok3, _ := l.Acquire(ctx, "key", time.Minute); !ok3 {
		t.Error("释放后应可再次获取")
	}
}

// TestLockerTTLExpiry 验证锁 TTL 到期后自动可获取。
func TestLockerTTLExpiry(t *testing.T) {
	l := NewMemoryLocker()
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	current := base
	l.now = func() time.Time { return current }
	ctx := context.Background()

	_, _ = l.Acquire(ctx, "key", 5*time.Second)
	current = base.Add(6 * time.Second) // 锁已过期
	if ok, _ := l.Acquire(ctx, "key", 5*time.Second); !ok {
		t.Error("锁到期后应可再次获取")
	}
}
