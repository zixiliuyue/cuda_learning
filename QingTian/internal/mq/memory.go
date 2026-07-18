// 本文件提供 Producer / Consumer 的内存实现（进程内消息总线）。
//
// 内存实现用于单元测试与本地单机演示：生产者发布的消息会同步投递给
// 已订阅对应主题的消费者回调。它不具备持久化与跨进程能力。
package mq

import (
	"context"
	"sync"
)

// MemoryBus 是一个进程内消息总线，同时充当 Producer 与 Consumer。
//
// 多个业务组件可共享同一个 MemoryBus 实例，从而在同一进程内完成
// 发布/订阅通信，便于端到端逻辑测试。
type MemoryBus struct {
	mu   sync.RWMutex
	subs map[string][]Handler // topic -> handler 列表
}

// NewMemoryBus 创建一个内存消息总线。
func NewMemoryBus() *MemoryBus {
	return &MemoryBus{subs: make(map[string][]Handler)}
}

// Publish 实现 Producer.Publish。
//
// 会同步调用该主题下所有已注册的处理回调。为避免持锁期间执行用户回调
// 造成死锁，这里先复制回调列表再释放锁后调用。
func (b *MemoryBus) Publish(ctx context.Context, topic string, body []byte) error {
	b.mu.RLock()
	handlers := make([]Handler, len(b.subs[topic]))
	copy(handlers, b.subs[topic])
	b.mu.RUnlock()

	for _, h := range handlers {
		// 拷贝消息体，避免不同消费者相互影响底层切片。
		cp := make([]byte, len(body))
		copy(cp, body)
		if err := h(ctx, cp); err != nil {
			return err
		}
	}
	return nil
}

// Subscribe 实现 Consumer.Subscribe。
//
// 注意：内存实现忽略 channel 语义（不做负载均衡），同一主题的所有回调都会被调用。
func (b *MemoryBus) Subscribe(topic, _ string, handler Handler) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.subs[topic] = append(b.subs[topic], handler)
	return nil
}

// Close 实现 Producer/Consumer.Close，内存实现无资源需释放。
func (b *MemoryBus) Close() error { return nil }
