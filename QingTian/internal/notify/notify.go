// Package notify 定义任务通知的抽象与实现。
//
// 任务通知让平台从"用户轮询"转为"主动触达"，提升日活粘性：
// 任务进入终态（成功/失败）时，通过飞书/企业微信群机器人 Webhook 推送消息。
//
// 通过接口抽象，业务不直接依赖具体渠道：单元测试使用内存实现，
// 生产环境使用基于 HTTP 的飞书/企微 Webhook 实现。
package notify

import "context"

// Message 表示一条待发送的通知。
//
// 字段：
//   - Title:   通知标题。
//   - Content: 通知正文（纯文本）。
type Message struct {
	Title   string
	Content string
}

// Notifier 定义通知发送能力。
type Notifier interface {
	// Notify 发送一条通知。
	Notify(ctx context.Context, msg Message) error
}

// MemoryNotifier 是 Notifier 的内存实现，用于单元测试。
//
// 它把发送的消息记录在内存切片中，供断言校验。
type MemoryNotifier struct {
	Sent []Message
}

// NewMemoryNotifier 创建内存通知器。
func NewMemoryNotifier() *MemoryNotifier {
	return &MemoryNotifier{}
}

// Notify 实现 Notifier.Notify，仅记录消息。
func (n *MemoryNotifier) Notify(_ context.Context, msg Message) error {
	n.Sent = append(n.Sent, msg)
	return nil
}
