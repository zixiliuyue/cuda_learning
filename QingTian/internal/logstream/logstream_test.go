package logstream

import (
	"context"
	"testing"
	"time"
)

// TestMemoryStreamerAppendAndRead 验证追加与从头/断点读取。
func TestMemoryStreamerAppendAndRead(t *testing.T) {
	s := NewMemoryStreamer()
	ctx := context.Background()

	seq1, _ := s.Append(ctx, "task1", "line-1")
	seq2, _ := s.Append(ctx, "task1", "line-2")
	if seq1 != 1 || seq2 != 2 {
		t.Fatalf("序号应从 1 递增，实际 %d,%d", seq1, seq2)
	}

	// 从头读取。
	all, _ := s.ReadFrom(ctx, "task1", 0)
	if len(all) != 2 {
		t.Fatalf("应读到 2 行，实际 %d", len(all))
	}
	// 断点续读（seq>1）。
	rest, _ := s.ReadFrom(ctx, "task1", 1)
	if len(rest) != 1 || rest[0].Content != "line-2" {
		t.Errorf("断点续读应只返回 line-2，实际 %+v", rest)
	}
}

// TestMemoryStreamerSubscribe 验证订阅者能收到后续新增日志。
func TestMemoryStreamerSubscribe(t *testing.T) {
	s := NewMemoryStreamer()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ch, _ := s.Subscribe(ctx, "task1")
	_, _ = s.Append(ctx, "task1", "hello")

	select {
	case line := <-ch:
		if line.Content != "hello" {
			t.Errorf("订阅应收到 hello，实际 %s", line.Content)
		}
	case <-time.After(time.Second):
		t.Fatal("订阅者未在超时内收到日志")
	}
}

// TestMemoryStreamerCloseNotifiesSubscribers 验证关闭时订阅通道被关闭。
func TestMemoryStreamerCloseNotifiesSubscribers(t *testing.T) {
	s := NewMemoryStreamer()
	ctx := context.Background()
	ch, _ := s.Subscribe(ctx, "task1")

	_ = s.Close(ctx, "task1")
	select {
	case _, ok := <-ch:
		if ok {
			t.Error("关闭后通道应被关闭")
		}
	case <-time.After(time.Second):
		t.Fatal("关闭未在超时内通知订阅者")
	}
}

// TestMemoryStreamerSubscribeAfterClose 验证任务结束后订阅返回已关闭通道。
func TestMemoryStreamerSubscribeAfterClose(t *testing.T) {
	s := NewMemoryStreamer()
	ctx := context.Background()
	_, _ = s.Append(ctx, "task1", "x")
	_ = s.Close(ctx, "task1")

	ch, _ := s.Subscribe(ctx, "task1")
	if _, ok := <-ch; ok {
		t.Error("对已结束任务的订阅应立即得到关闭通道")
	}
}
