// Package logstream 定义任务实时日志流的抽象与实现。
//
// 实时日志是消除"提交后黑盒等待"体验硬伤的关键能力：Agent 边执行边把
// stdout/stderr 逐行推送到日志流，前端通过 SSE/WebSocket 实时拉取；
// 任务结束后日志再归档到对象存储（MinIO）长期留存。
//
// 通过接口抽象，业务不直接依赖 Redis：单元测试使用内存实现，
// 生产环境使用基于 Redis List + Pub/Sub 的实现。
package logstream

import (
	"context"
	"sync"
)

// LogLine 表示一条日志。
//
// 字段：
//   - Seq:     序号，从 1 开始单调递增，供前端断点续读与去重。
//   - Content: 日志文本内容（单行）。
type LogLine struct {
	Seq     int64  `json:"seq"`
	Content string `json:"content"`
}

// Streamer 定义任务日志的追加、读取与订阅能力。
type Streamer interface {
	// Append 向指定任务的日志流追加一行，返回该行分配到的序号。
	Append(ctx context.Context, taskID, content string) (int64, error)
	// ReadFrom 读取指定任务从 afterSeq（不含）之后的所有日志。
	//
	// afterSeq 传 0 表示从头读取。用于前端首次加载或断线重连补齐。
	ReadFrom(ctx context.Context, taskID string, afterSeq int64) ([]LogLine, error)
	// Subscribe 订阅指定任务的后续新增日志，返回只读通道。
	//
	// 通道在 ctx 取消或调用 Close 时关闭。用于 SSE/WebSocket 实时推送。
	Subscribe(ctx context.Context, taskID string) (<-chan LogLine, error)
	// Close 关闭指定任务的日志流并通知所有订阅者（任务结束时调用）。
	Close(ctx context.Context, taskID string) error
}

// MemoryStreamer 是 Streamer 的内存实现，用于单元测试与本地演示。
//
// 每个任务维护一份日志切片与一组订阅者通道，追加时同步分发给订阅者。
type MemoryStreamer struct {
	mu     sync.Mutex
	logs   map[string][]LogLine        // taskID -> 日志行
	subs   map[string][]chan LogLine   // taskID -> 订阅者通道列表
	closed map[string]bool             // taskID -> 是否已关闭
}

// NewMemoryStreamer 创建内存日志流。
func NewMemoryStreamer() *MemoryStreamer {
	return &MemoryStreamer{
		logs:   make(map[string][]LogLine),
		subs:   make(map[string][]chan LogLine),
		closed: make(map[string]bool),
	}
}

// Append 实现 Streamer.Append。
func (m *MemoryStreamer) Append(_ context.Context, taskID, content string) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	seq := int64(len(m.logs[taskID])) + 1
	line := LogLine{Seq: seq, Content: content}
	m.logs[taskID] = append(m.logs[taskID], line)
	// 同步分发给订阅者，通道满时跳过以避免阻塞追加。
	for _, ch := range m.subs[taskID] {
		select {
		case ch <- line:
		default:
		}
	}
	return seq, nil
}

// ReadFrom 实现 Streamer.ReadFrom。
func (m *MemoryStreamer) ReadFrom(_ context.Context, taskID string, afterSeq int64) ([]LogLine, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	all := m.logs[taskID]
	if afterSeq <= 0 {
		out := make([]LogLine, len(all))
		copy(out, all)
		return out, nil
	}
	var out []LogLine
	for _, l := range all {
		if l.Seq > afterSeq {
			out = append(out, l)
		}
	}
	return out, nil
}

// Subscribe 实现 Streamer.Subscribe。
func (m *MemoryStreamer) Subscribe(ctx context.Context, taskID string) (<-chan LogLine, error) {
	m.mu.Lock()
	ch := make(chan LogLine, 256)
	if m.closed[taskID] {
		// 任务已结束，直接返回已关闭的空通道。
		m.mu.Unlock()
		close(ch)
		return ch, nil
	}
	m.subs[taskID] = append(m.subs[taskID], ch)
	m.mu.Unlock()

	// ctx 取消时移除订阅并关闭通道。
	go func() {
		<-ctx.Done()
		m.removeSub(taskID, ch)
	}()
	return ch, nil
}

// removeSub 从订阅列表移除并关闭指定通道（幂等）。
func (m *MemoryStreamer) removeSub(taskID string, target chan LogLine) {
	m.mu.Lock()
	defer m.mu.Unlock()
	subs := m.subs[taskID]
	for i, ch := range subs {
		if ch == target {
			m.subs[taskID] = append(subs[:i], subs[i+1:]...)
			close(ch)
			return
		}
	}
}

// Close 实现 Streamer.Close。
func (m *MemoryStreamer) Close(_ context.Context, taskID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed[taskID] = true
	for _, ch := range m.subs[taskID] {
		close(ch)
	}
	m.subs[taskID] = nil
	return nil
}
