// 本文件提供 tracing.Exporter 的两个实现：
//   - SlogExporter:   将结束的 Span 以结构化日志输出，接入现有日志体系。
//   - MemoryExporter: 在内存中收集 Span，供单元测试断言链路结构。
package tracing

import (
	"log/slog"
	"sync"
)

// SlogExporter 通过 slog 输出 Span，便于在现有日志中观察链路耗时。
type SlogExporter struct {
	logger *slog.Logger
}

// NewSlogExporter 创建基于 slog 的导出器；logger 为 nil 时使用默认 logger。
func NewSlogExporter(logger *slog.Logger) *SlogExporter {
	if logger == nil {
		logger = slog.Default()
	}
	return &SlogExporter{logger: logger}
}

// Export 实现 Exporter.Export，以结构化字段记录一次 Span。
func (e *SlogExporter) Export(span *Span) {
	attrs := []any{
		"trace_id", span.TraceID,
		"span_id", span.SpanID,
		"parent_id", span.ParentID,
		"name", span.Name,
		"duration_ms", span.Duration().Milliseconds(),
	}
	for k, v := range span.Attrs {
		attrs = append(attrs, k, v)
	}
	e.logger.Info("trace span", attrs...)
}

// MemoryExporter 在内存中收集全部 Span，主要用于测试。
type MemoryExporter struct {
	mu    sync.Mutex
	spans []*Span
}

// NewMemoryExporter 创建内存导出器。
func NewMemoryExporter() *MemoryExporter {
	return &MemoryExporter{}
}

// Export 实现 Exporter.Export，追加保存 Span。
func (e *MemoryExporter) Export(span *Span) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.spans = append(e.spans, span)
}

// Spans 返回已收集 Span 的快照副本。
func (e *MemoryExporter) Spans() []*Span {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]*Span, len(e.spans))
	copy(out, e.spans)
	return out
}
