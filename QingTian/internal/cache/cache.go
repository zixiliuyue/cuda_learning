// Package cache 定义缓存与协调能力的抽象接口。
//
// 平台使用 Redis 实现节点心跳（带 TTL）、待调度优先级队列、分布式锁等。
// 通过接口抽象，业务代码不直接依赖 Redis，单元测试可使用内存实现。
package cache

import (
	"context"
	"time"
)

// Cache 定义键值缓存能力（带过期）。
type Cache interface {
	// Set 写入键值，ttl 为 0 表示永不过期。
	Set(ctx context.Context, key, value string, ttl time.Duration) error
	// Get 读取键值，不存在时返回 (\"\", false, nil)。
	Get(ctx context.Context, key string) (string, bool, error)
	// Del 删除键。
	Del(ctx context.Context, key string) error
	// Keys 返回匹配前缀的所有键。
	Keys(ctx context.Context, prefix string) ([]string, error)
}

// PriorityQueue 定义带优先级的待调度任务队列能力。
//
// 队列按 score 从大到小出队（score 通常取任务优先级），score 相同时
// 先入队者先出，从而实现"高优先级优先、同优先级 FIFO"的调度语义。
type PriorityQueue interface {
	// Push 入队一个任务 ID，附带优先级分值。
	Push(ctx context.Context, taskID string, score float64) error
	// Pop 出队优先级最高的任务 ID，队列为空时返回 (\"\", false, nil)。
	Pop(ctx context.Context) (string, bool, error)
	// Len 返回队列长度。
	Len(ctx context.Context) (int, error)
}

// Locker 定义分布式锁能力，用于调度器互斥、防止资源超卖。
type Locker interface {
	// Acquire 尝试获取指定 key 的锁，ttl 为锁自动过期时间（防死锁）。
	//
	// 返回值：
	//   - acquired: 是否成功获取。
	//   - err:      底层错误。
	Acquire(ctx context.Context, key string, ttl time.Duration) (bool, error)
	// Release 释放锁。
	Release(ctx context.Context, key string) error
}
