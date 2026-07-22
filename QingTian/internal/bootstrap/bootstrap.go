// Package bootstrap 负责根据配置连接真实的外部中间件并组装依赖。
//
// 该包集中处理 MongoDB、Redis、NSQ、MinIO 的连接创建，供各服务 main 复用，
// 避免连接逻辑散落。若中间件不可用，连接会在此返回错误。
package bootstrap

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	"qingtian/internal/cache"
	"qingtian/internal/config"
	"qingtian/internal/logstream"
	"qingtian/internal/mq"
	"qingtian/internal/repository"
	"qingtian/internal/storage"
)

// Infra 聚合所有已连接的基础设施客户端。
type Infra struct {
	Mongo     *mongo.Client
	Store     *repository.MongoStore
	Cache     *cache.RedisCache
	Producer  *mq.NSQProducer
	Consumer  *mq.NSQConsumer
	Object    *storage.MinIOStore
	LogStream *logstream.RedisStreamer
}

// Connect 根据配置连接所有外部中间件。
//
// 参数：
//   - ctx: 上下文，用于连接超时控制。
//   - cfg: 平台配置。
//
// 返回值：
//   - *Infra: 已连接的基础设施集合。
//   - error:  任一中间件连接失败时返回。
//
// 注意事项：
//   - 调用方在使用完毕后应调用 Infra.Close 释放资源。
func Connect(ctx context.Context, cfg *config.Config) (*Infra, error) {
	// MongoDB: 重试几次等待 MongoDB 就绪（Docker 启动顺序问题）
	var mclient *mongo.Client
	var err error
	const maxRetries = 10
	for range maxRetries {
		mctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		mclient, err = mongo.Connect(mctx, options.Client().ApplyURI(cfg.Mongo.URI))
		if err == nil {
			err = mclient.Ping(mctx, nil)
			cancel()
			if err == nil {
				break
			}
		}
		cancel()
		// 连接失败，等待一秒重试
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
		}
	}
	if err != nil {
		return nil, err
	}
	store := repository.NewMongoStore(mclient.Database(cfg.Mongo.Database))

	// Redis: 重试几次等待 Redis 就绪
	var rcache *cache.RedisCache
	for range maxRetries {
		rcache, err = cache.NewRedisCache(cfg.Redis)
		if err == nil {
			err = rcache.Ping(ctx)
			if err == nil {
				break
			}
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
		}
	}
	if err != nil {
		return nil, err
	}

	// NSQ
	var producer *mq.NSQProducer
	for i := range maxRetries {
		producer, err = mq.NewNSQProducer(cfg.NSQ)
		if err == nil {
			break
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
		}
		if i == maxRetries-1 {
			return nil, err
		}
	}
	consumer := mq.NewNSQConsumer(cfg.NSQ)

	// MinIO: 重试几次等待 MinIO 就绪
	var object *storage.MinIOStore
	for i := range maxRetries {
		object, err = storage.NewMinIOStore(cfg.MinIO)
		if err == nil {
			break
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
		}
		if i == maxRetries-1 {
			return nil, err
		}
	}

	// 日志流（复用 Redis 连接配置，独立客户端）。
	logStream := logstream.NewRedisStreamer(cfg.Redis, 24*time.Hour)

	return &Infra{
		Mongo:     mclient,
		Store:     store,
		Cache:     rcache,
		Producer:  producer,
		Consumer:  consumer,
		Object:    object,
		LogStream: logStream,
	}, nil
}

// Close 释放所有基础设施连接。
func (i *Infra) Close(ctx context.Context) {
	if i.Producer != nil {
		_ = i.Producer.Close()
	}
	if i.Consumer != nil {
		_ = i.Consumer.Close()
	}
	if i.Cache != nil {
		_ = i.Cache.Close()
	}
	if i.Mongo != nil {
		_ = i.Mongo.Disconnect(ctx)
	}
}
