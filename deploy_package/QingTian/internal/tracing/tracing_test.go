package tracing

import (
	"context"
	"testing"
)

// TestSpanInheritsTraceID 验证子 Span 继承父 Span 的 TraceID 并正确记录父子关系。
func TestSpanInheritsTraceID(t *testing.T) {
	exporter := NewMemoryExporter()
	tracer := NewTracer(exporter)

	ctx, root := tracer.Start(context.Background(), "root")
	if root.TraceID == "" || root.SpanID == "" {
		t.Fatal("根 Span 应分配 TraceID 与 SpanID")
	}
	if root.ParentID != "" {
		t.Errorf("根 Span 不应有父 Span，实际 %s", root.ParentID)
	}

	_, child := tracer.Start(ctx, "child")
	if child.TraceID != root.TraceID {
		t.Errorf("子 Span 应继承 TraceID：期望 %s，实际 %s", root.TraceID, child.TraceID)
	}
	if child.ParentID != root.SpanID {
		t.Errorf("子 Span 的 ParentID 应为根 SpanID：期望 %s，实际 %s", root.SpanID, child.ParentID)
	}

	child.Finish()
	root.Finish()

	spans := exporter.Spans()
	if len(spans) != 2 {
		t.Fatalf("应导出 2 个 Span，实际 %d", len(spans))
	}
}

// TestSpanFinishIsIdempotent 验证重复 Finish 不会重复导出。
func TestSpanFinishIsIdempotent(t *testing.T) {
	exporter := NewMemoryExporter()
	tracer := NewTracer(exporter)

	_, span := tracer.Start(context.Background(), "op")
	span.Finish()
	span.Finish() // 第二次应被忽略

	if len(exporter.Spans()) != 1 {
		t.Errorf("重复 Finish 应仅导出一次，实际 %d", len(exporter.Spans()))
	}
}

// TestTraceIDFromContext 验证可从 context 读取当前链路 ID。
func TestTraceIDFromContext(t *testing.T) {
	tracer := NewTracer(nil)
	ctx, span := tracer.Start(context.Background(), "op")
	if got := TraceIDFromContext(ctx); got != span.TraceID {
		t.Errorf("从 context 读取的 TraceID 应为 %s，实际 %s", span.TraceID, got)
	}
	if got := TraceIDFromContext(context.Background()); got != "" {
		t.Errorf("无 Span 的 context 应返回空 TraceID，实际 %s", got)
	}
}

// TestNilExporterSafe 验证未配置导出器时 Finish 不 panic。
func TestNilExporterSafe(t *testing.T) {
	tracer := NewTracer(nil)
	_, span := tracer.Start(context.Background(), "op")
	span.SetAttr("k", "v")
	span.Finish() // 不应 panic
}
