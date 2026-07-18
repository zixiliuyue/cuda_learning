// 本文件提供基于 Redis 的 Cache / PriorityQueue / Locker 实现。
//
// 生产环境使用该实现：
//   - Cache 使用 Redis 字符串 + TTL。
//   - PriorityQueue 使用 Redis ZSet，score 为优先级，按 score 降序出队。
//   - Locker 使用 SET NX + TTL 实现分布式锁。
package cache

import (
	"context"
	"time"

	"qingtian/internal/config"

	"github.com/redis/go-redis/v9"
)

// RedisCache 基于 Redis 实现 Cache、PriorityQueue、Locker 三个接口。
//
// 通过一个结构统一持有 *redis.Client，并为优先级队列指定固定的 ZSet 键。
type RedisCache struct {
	client *redis.Client
	// queueKey 是优先级队列使用的 ZSet 键名。
	queueKey string
}

// NewRedisCache 根据配置创建 Redis 客户端并封装为 RedisCache。
//
// 参数：
//   - cfg: Redis 连接配置。
//
// 返回值：
//   - *RedisCache: 封装实例。
//   - error:       连接创建失败时返回（此处仅构建客户端，真正连接在首次命令时建立）。
//
// 注意事项：
//   - 调用方应在使用前通过 Ping 校验连通性；本函数不做网络往返。
func NewRedisCache(cfg config.RedisConfig) (*RedisCache, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     cfg.Addr,
		Password: cfg.Password,
		DB:       cfg.DB,
	})
	return &RedisCache{client: client, queueKey: "qingtian:pending_queue"}, nil
}

// Ping 校验 Redis 连通性。
func (r *RedisCache) Ping(ctx context.Context) error {
	return r.client.Ping(ctx).Err()
}

// Close 关闭 Redis 连接。
func (r *RedisCache) Close() error { return r.client.Close() }

// Set 实现 Cache.Set。
func (r *RedisCache) Set(ctx context.Context, key, value string, ttl time.Duration) error {
	return r.client.Set(ctx, key, value, ttl).Err()
}

// Get 实现 Cache.Get。
func (r *RedisCache) Get(ctx context.Context, key string) (string, bool, error) {
	v, err := r.client.Get(ctx, key).Result()
	if err == redis.Nil {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return v, true, nil
}

// Del 实现 Cache.Del。
func (r *RedisCache) Del(ctx context.Context, key string) error {
	return r.client.Del(ctx, key).Err()
}

// Keys 实现 Cache.Keys。
//
// 注意：KEYS 命令在大数据量下会阻塞 Redis，生产可改用 SCAN；这里为语义完整保留 KEYS。
func (r *RedisCache) Keys(ctx context.Context, prefix string) ([]string, error) {
	return r.client.Keys(ctx, prefix+"*").Result()
}

// Push 实现 PriorityQueue.Push，写入 ZSet。
func (r *RedisCache) Push(ctx context.Context, taskID string, score float64) error {
	return r.client.ZAdd(ctx, r.queueKey, redis.Z{Score: score, Member: taskID}).Err()
}

// Pop 实现 PriorityQueue.Pop，弹出 score 最大的成员。
//
// 使用 ZPOPMAX 原子弹出，保证并发调度器之间不会取到同一任务。
func (r *RedisCache) Pop(ctx context.Context) (string, bool, error) {
	res, err := r.client.ZPopMax(ctx, r.queueKey, 1).Result()
	if err != nil {
		return "", false, err
	}
	if len(res) == 0 {
		return "", false, nil
	}
	return res[0].Member.(string), true, nil
}

// Len 实现 PriorityQueue.Len。
func (r *RedisCache) Len(ctx context.Context) (int, error) {
	n, err := r.client.ZCard(ctx, r.queueKey).Result()
	return int(n), err
}

// Acquire 实现 Locker.Acquire，使用 SET NX 保证原子性。
func (r *RedisCache) Acquire(ctx context.Context, key string, ttl time.Duration) (bool, error) {
	return r.client.SetNX(ctx, "lock:"+key, "1", ttl).Result()
}

// Release 实现 Locker.Release。
func (r *RedisCache) Release(ctx context.Context, key string) error {
	return r.client.Del(ctx, "lock:"+key).Err()
}
