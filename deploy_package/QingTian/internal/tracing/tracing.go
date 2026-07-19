// Package tracing 提供零依赖的轻量分布式追踪能力。
//
// 生产级方案通常使用 OpenTelemetry，但会引入较重的第三方依赖。为在不新增依赖的
// 前提下获得"提交 -> 调度 -> 执行"全链路的可观测性，本包用标准库实现最小可用的
// Trace/Span 模型：
//   - 每条请求分配一个 TraceID，贯穿多个处理阶段；
//   - 每个处理阶段是一个 Span，记录名称、耗时与属性；
//   - Span 通过 context 传播，子 Span 自动继承父 Span 的 TraceID。
//
// 该实现聚焦"结构化记录 + 上下文传播"，导出器接口可对接日志或后续替换为 OTLP。
package tracing

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"sync"
	"time"
)

// Span 表示一次带时间边界的操作记录。
//
// 字段：
//   - TraceID:  所属链路 ID，同一请求的所有 Span 共享。
//   - SpanID:   本 Span 的唯一 ID。
//   - ParentID: 父 Span ID，根 Span 为空。
//   - Name:     操作名称（如 "task.submit"）。
//   - Start/End: 起止时间。
//   - Attrs:    结构化属性（如 task_id、node_id）。
type Span struct {
	TraceID  string
	SpanID   string
	ParentID string
	Name     string
	Start    time.Time
	End      time.Time
	Attrs    map[string]string

	tracer   *Tracer
	finished bool
	mu       sync.Mutex
}

// Duration 返回 Span 的耗时；未结束时返回从开始到当前的时长。
func (s *Span) Duration() time.Duration {
	if s.End.IsZero() {
		return time.Since(s.Start)
	}
	return s.End.Sub(s.Start)
}

// SetAttr 为 Span 追加一个结构化属性，返回自身便于链式调用。
func (s *Span) SetAttr(key, value string) *Span {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Attrs == nil {
		s.Attrs = make(map[string]string)
	}
	s.Attrs[key] = value
	return s
}

// End 结束 Span，记录结束时间并交由导出器处理（幂等）。
func (s *Span) Finish() {
	s.mu.Lock()
	if s.finished {
		s.mu.Unlock()
		return
	}
	s.finished = true
	s.End = time.Now()
	s.mu.Unlock()
	if s.tracer != nil && s.tracer.exporter != nil {
		s.tracer.exporter.Export(s)
	}
}

// Exporter 定义 Span 导出能力，便于对接日志、内存收集或后续 OTLP。
type Exporter interface {
	Export(span *Span)
}

// Tracer 是 Span 的工厂，持有导出器。
type Tracer struct {
	exporter Exporter
}

// NewTracer 创建 Tracer；exporter 为 nil 时 Span 结束不产生副作用。
func NewTracer(exporter Exporter) *Tracer {
	return &Tracer{exporter: exporter}
}

// spanContextKey 是 context 中存放当前 Span 的键类型，避免键冲突。
type spanContextKey struct{}

// Start 基于 context 开启一个新 Span 并返回派生 context。
//
// 若父 context 中已存在 Span，则新 Span 继承其 TraceID 并将其设为 ParentID；
// 否则新建一条 Trace（生成新的 TraceID），作为根 Span。
//
// 参数：
//   - ctx:  上下文，可能已携带父 Span。
//   - name: 操作名称。
//
// 返回值：
//   - context.Context: 携带新 Span 的派生上下文。
//   - *Span:           新建的 Span，调用方应在操作结束时调用 Finish。
func (t *Tracer) Start(ctx context.Context, name string) (context.Context, *Span) {
	span := &Span{
		SpanID: randomID(8),
		Name:   name,
		Start:  time.Now(),
		tracer: t,
	}
	if parent, ok := SpanFromContext(ctx); ok {
		span.TraceID = parent.TraceID
		span.ParentID = parent.SpanID
	} else {
		span.TraceID = randomID(16)
	}
	return context.WithValue(ctx, spanContextKey{}, span), span
}

// SpanFromContext 从 context 中取出当前 Span。
//
// 返回值：
//   - *Span: 当前 Span。
//   - bool:  是否存在。
func SpanFromContext(ctx context.Context) (*Span, bool) {
	span, ok := ctx.Value(spanContextKey{}).(*Span)
	return span, ok
}

// TraceIDFromContext 返回当前链路的 TraceID，不存在时返回空串。
func TraceIDFromContext(ctx context.Context) string {
	if span, ok := SpanFromContext(ctx); ok {
		return span.TraceID
	}
	return ""
}

// randomID 生成 n 字节的随机十六进制 ID；随机源不可用时退化为时间戳。
func randomID(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return hex.EncodeToString([]byte(time.Now().Format("150405.000000")))
	}
	return hex.EncodeToString(b)
}
