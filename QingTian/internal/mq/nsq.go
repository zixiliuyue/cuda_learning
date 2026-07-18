// 本文件提供基于 NSQ 的 Producer / Consumer 实现。
//
// 生产环境使用该实现完成任务下发与状态回传：
//   - Producer 通过 nsqd 直连发布消息。
//   - Consumer 通过 nsqlookupd 发现 nsqd 并消费消息。
package mq

import (
	"context"

	"qingtian/internal/config"

	nsq "github.com/nsqio/go-nsq"
)

// NSQProducer 是 Producer 的 NSQ 实现。
type NSQProducer struct {
	producer *nsq.Producer
}

// NewNSQProducer 根据配置创建 NSQ 生产者。
//
// 参数：
//   - cfg: NSQ 配置，使用其中的 NSQDAddr 建立连接。
//
// 返回值：
//   - *NSQProducer: 生产者实例。
//   - error:        创建失败时返回。
func NewNSQProducer(cfg config.NSQConfig) (*NSQProducer, error) {
	p, err := nsq.NewProducer(cfg.NSQDAddr, nsq.NewConfig())
	if err != nil {
		return nil, err
	}
	return &NSQProducer{producer: p}, nil
}

// Publish 实现 Producer.Publish。
//
// 注意：NSQ 的 Publish 为同步阻塞发送；ctx 在此实现中不参与取消，仅保持接口一致。
func (p *NSQProducer) Publish(_ context.Context, topic string, body []byte) error {
	return p.producer.Publish(topic, body)
}

// Close 实现 Producer.Close。
func (p *NSQProducer) Close() error {
	p.producer.Stop()
	return nil
}

// NSQConsumer 是 Consumer 的 NSQ 实现。
//
// 每次 Subscribe 会创建一个独立的 nsq.Consumer 并连接到 nsqlookupd，
// 所有创建的消费者在 Close 时统一停止。
type NSQConsumer struct {
	lookupdAddr string
	maxInFlight int
	consumers   []*nsq.Consumer
}

// NewNSQConsumer 根据配置创建 NSQ 消费者管理器。
func NewNSQConsumer(cfg config.NSQConfig) *NSQConsumer {
	return &NSQConsumer{
		lookupdAddr: cfg.LookupdAddr,
		maxInFlight: cfg.MaxInFlight,
	}
}

// nsqHandlerAdapter 将平台的 Handler 适配为 nsq.Handler 接口。
type nsqHandlerAdapter struct {
	handler Handler
}

// HandleMessage 实现 nsq.Handler。
//
// 返回 error 会触发 NSQ 的自动重新投递（requeue），从而实现失败重试。
func (a *nsqHandlerAdapter) HandleMessage(m *nsq.Message) error {
	return a.handler(context.Background(), m.Body)
}

// Subscribe 实现 Consumer.Subscribe。
func (c *NSQConsumer) Subscribe(topic, channel string, handler Handler) error {
	cfg := nsq.NewConfig()
	if c.maxInFlight > 0 {
		cfg.MaxInFlight = c.maxInFlight
	}
	consumer, err := nsq.NewConsumer(topic, channel, cfg)
	if err != nil {
		return err
	}
	consumer.AddHandler(&nsqHandlerAdapter{handler: handler})
	if err := consumer.ConnectToNSQLookupd(c.lookupdAddr); err != nil {
		return err
	}
	c.consumers = append(c.consumers, consumer)
	return nil
}

// Close 实现 Consumer.Close，停止所有消费者。
func (c *NSQConsumer) Close() error {
	for _, consumer := range c.consumers {
		consumer.Stop()
	}
	return nil
}
