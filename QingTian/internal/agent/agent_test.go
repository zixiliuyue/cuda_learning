package agent

import (
	"context"
	"encoding/json"
	"log/slog"
	"testing"

	"qingtian/internal/domain"
	"qingtian/internal/mq"
)

// fakeControlPlane 是 ControlPlane 的测试桩，记录注册与心跳调用。
type fakeControlPlane struct {
	registered bool
	heartbeats int
}

func (f *fakeControlPlane) Register(_ context.Context, req RegisterInfo) (string, error) {
	f.registered = true
	return "node-" + req.Hostname, nil
}

func (f *fakeControlPlane) Heartbeat(_ context.Context, _ string, _ []domain.GPU) error {
	f.heartbeats++
	return nil
}

// TestMockCollector 验证模拟采集器返回 GPU 并更新利用率。
func TestMockCollector(t *testing.T) {
	c := DefaultMockCollector()
	gpus, err := c.Collect()
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 || gpus[0].Model != "RTX 5060 Ti" {
		t.Errorf("采集结果不符合预期: %+v", gpus)
	}
	if c.CUDAVersion() != "13.1" {
		t.Errorf("CUDA 版本应为 13.1，实际 %s", c.CUDAVersion())
	}
}

// TestCommandExecutorSuccess 验证本地命令执行成功返回退出码 0。
func TestCommandExecutorSuccess(t *testing.T) {
	e := NewCommandExecutor("")
	res := e.Execute(context.Background(), mq.DispatchMessage{Command: "echo hello"})
	if res.ExitCode != 0 || res.Err != nil {
		t.Errorf("命令应成功执行，结果: %+v", res)
	}
}

// TestCommandExecutorEmpty 验证空命令视为成功。
func TestCommandExecutorEmpty(t *testing.T) {
	e := NewCommandExecutor("")
	res := e.Execute(context.Background(), mq.DispatchMessage{Command: ""})
	if res.ExitCode != 0 {
		t.Errorf("空命令应视为成功，实际退出码 %d", res.ExitCode)
	}
}

// TestAgentRegisterAndDispatch 验证 Agent 注册、消费下发并回传状态的闭环。
func TestAgentRegisterAndDispatch(t *testing.T) {
	ctx := context.Background()
	bus := mq.NewMemoryBus()
	cp := &fakeControlPlane{}
	log := slog.Default()

	ag := New(Config{Hostname: "test-node"}, DefaultMockCollector(), NewCommandExecutor(""), cp, bus, bus, log)

	// 注册。
	if err := ag.Register(ctx); err != nil {
		t.Fatalf("注册应成功: %v", err)
	}
	if !cp.registered || ag.NodeID() != "node-test-node" {
		t.Fatalf("注册后应记录 nodeID，实际 %s", ag.NodeID())
	}

	// 控制面侧订阅状态回传，收集结果。
	var received mq.StatusMessage
	statusCount := 0
	_ = bus.Subscribe(mq.TopicTaskStatus, "ctrl", func(_ context.Context, body []byte) error {
		_ = json.Unmarshal(body, &received)
		statusCount++
		return nil
	})

	// 订阅任务下发。
	if err := ag.SubscribeDispatch(); err != nil {
		t.Fatalf("订阅下发应成功: %v", err)
	}

	// 下发一个任务。
	disp := mq.DispatchMessage{TaskID: "task-1", NodeID: ag.NodeID(), Command: "echo run"}
	body, _ := json.Marshal(disp)
	if err := bus.Publish(ctx, mq.TaskDispatchTopic(ag.NodeID()), body); err != nil {
		t.Fatalf("下发应成功: %v", err)
	}

	// 应至少收到 running 与 succeeded 两次状态回传，最终为 succeeded。
	if statusCount < 2 {
		t.Errorf("应收到至少 2 次状态回传，实际 %d", statusCount)
	}
	if received.Status != string(domain.TaskSucceeded) {
		t.Errorf("最终状态应为 succeeded，实际 %s", received.Status)
	}
}
