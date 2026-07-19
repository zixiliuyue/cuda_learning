package agent

import (
	"context"
	"encoding/json"
	"log/slog"
	"testing"

	"qingtian/internal/mq"
)

// TestHandleControlKillCancelsTask 验证控制指令 kill 会取消对应任务的执行上下文。
func TestHandleControlKillCancelsTask(t *testing.T) {
	ag := New(Config{Hostname: "n1"}, DefaultMockCollector(), NewCommandExecutor(""),
		&fakeControlPlane{}, mq.NewMemoryBus(), mq.NewMemoryBus(), slog.Default())

	// 手动登记一个运行中任务的取消函数。
	ctx, cancel := context.WithCancel(context.Background())
	ag.running["task-x"] = cancel

	// 下达 kill 控制指令。
	body, _ := json.Marshal(mq.ControlMessage{TaskID: "task-x", Action: mq.ControlKill, Reason: "超时"})
	if err := ag.handleControl(context.Background(), body); err != nil {
		t.Fatalf("处理控制指令应成功: %v", err)
	}

	// 上下文应已被取消。
	select {
	case <-ctx.Done():
		// 符合预期。
	default:
		t.Error("kill 指令后任务上下文应被取消")
	}
}

// TestHandleControlUnknownTaskNoop 验证针对未知任务的 kill 指令安全无副作用。
func TestHandleControlUnknownTaskNoop(t *testing.T) {
	ag := New(Config{Hostname: "n1"}, DefaultMockCollector(), NewCommandExecutor(""),
		&fakeControlPlane{}, mq.NewMemoryBus(), mq.NewMemoryBus(), slog.Default())

	body, _ := json.Marshal(mq.ControlMessage{TaskID: "not-exist", Action: mq.ControlKill})
	if err := ag.handleControl(context.Background(), body); err != nil {
		t.Errorf("未知任务的控制指令不应报错: %v", err)
	}
}
