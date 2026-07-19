// 本文件提供基于 Redis 的 Streamer 实现。
//
// 生产环境使用该实现：
//   - 日志行持久化在 Redis List（qingtian:logs:<taskID>），支持按序号回溯。
//   - 实时推送使用 Redis Pub/Sub（qingtian:logpub:<taskID>），多副本 API 均可订阅。
//   - 列表设置 TTL 防止无限增长；任务结束后日志应归档到对象存储长期留存。
package logstream

import (
	"context"
	"encoding/json"
	"time"

	"qingtian/internal/config"

	"github.com/redis/go-redis/v9"
)

// RedisStreamer 基于 Redis List + Pub/Sub 实现 Streamer。
type RedisStreamer struct {
	client *redis.Client
	ttl    time.Duration
}

// NewRedisStreamer 根据配置创建 Redis 日志流。
//
// 参数：
//   - cfg: Redis 连接配置。
//   - ttl: 日志在 Redis 中的保留时长（超过后依赖对象存储归档）。
func NewRedisStreamer(cfg config.RedisConfig, ttl time.Duration) *RedisStreamer {
	client := redis.NewClient(&redis.Options{
		Addr:     cfg.Addr,
		Password: cfg.Password,
		DB:       cfg.DB,
	})
	if ttl <= 0 {
		ttl = 24 * time.Hour
	}
	return &RedisStreamer{client: client, ttl: ttl}
}

// Close 关闭 Redis 连接（注意：与 Streamer.Close 语义不同，此为释放客户端）。
func (r *RedisStreamer) CloseClient() error { return r.client.Close() }

// listKey 返回任务日志列表键。
func (r *RedisStreamer) listKey(taskID string) string { return "qingtian:logs:" + taskID }

// pubChannel 返回任务日志发布通道名。
func (r *RedisStreamer) pubChannel(taskID string) string { return "qingtian:logpub:" + taskID }

// Append 实现 Streamer.Append。
func (r *RedisStreamer) Append(ctx context.Context, taskID, content string) (int64, error) {
	key := r.listKey(taskID)
	// RPUSH 返回列表新长度，即该行序号。
	seq, err := r.client.RPush(ctx, key, content).Result()
	if err != nil {
		return 0, err
	}
	r.client.Expire(ctx, key, r.ttl)
	line := LogLine{Seq: seq, Content: content}
	body, _ := json.Marshal(line)
	// 发布给实时订阅者，失败不影响持久化。
	_ = r.client.Publish(ctx, r.pubChannel(taskID), body).Err()
	return seq, nil
}

// ReadFrom 实现 Streamer.ReadFrom。
func (r *RedisStreamer) ReadFrom(ctx context.Context, taskID string, afterSeq int64) ([]LogLine, error) {
	// List 下标从 0 开始，序号从 1 开始；afterSeq 之后即下标 afterSeq 起。
	start := afterSeq
	if start < 0 {
		start = 0
	}
	items, err := r.client.LRange(ctx, r.listKey(taskID), start, -1).Result()
	if err != nil {
		return nil, err
	}
	out := make([]LogLine, 0, len(items))
	for i, content := range items {
		out = append(out, LogLine{Seq: start + int64(i) + 1, Content: content})
	}
	return out, nil
}

// Subscribe 实现 Streamer.Subscribe。
func (r *RedisStreamer) Subscribe(ctx context.Context, taskID string) (<-chan LogLine, error) {
	sub := r.client.Subscribe(ctx, r.pubChannel(taskID))
	out := make(chan LogLine, 256)
	go func() {
		defer close(out)
		defer sub.Close()
		ch := sub.Channel()
		for {
			select {
			case <-ctx.Done():
				return
			case msg, ok := <-ch:
				if !ok {
					return
				}
				var line LogLine
				if err := json.Unmarshal([]byte(msg.Payload), &line); err == nil {
					select {
					case out <- line:
					case <-ctx.Done():
						return
					}
				}
			}
		}
	}()
	return out, nil
}

// Close 实现 Streamer.Close。
//
// Redis 实现无需显式关闭订阅（由订阅者 ctx 控制），这里发布一条结束标记，
// 便于订阅者感知任务已结束。序号 -1 约定为结束标记。
func (r *RedisStreamer) Close(ctx context.Context, taskID string) error {
	body, _ := json.Marshal(LogLine{Seq: -1, Content: ""})
	return r.client.Publish(ctx, r.pubChannel(taskID), body).Err()
}
