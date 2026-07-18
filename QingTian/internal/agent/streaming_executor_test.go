package agent

import (
	"context"
	"runtime"
	"sync"
	"testing"

	"qingtian/internal/mq"
)

// TestStreamingExecutorStreamsLines 验证流式执行器逐行回调并返回退出码。
func TestStreamingExecutorStreamsLines(t *testing.T) {
	var (
		mu    sync.Mutex
		lines []string
	)
	sink := func(taskID, line string) {
		mu.Lock()
		lines = append(lines, line)
		mu.Unlock()
	}
	exec := NewStreamingExecutor("", sink)

	// 跨平台的多行输出命令：Windows 用 powershell，其它用 sh（由执行器内部选择）。
	var cmd string
	if runtime.GOOS == "windows" {
		cmd = "Write-Output 'a'; Write-Output 'b'"
	} else {
		cmd = "printf 'a\\nb\\n'"
	}
	res := exec.Execute(context.Background(), mq.DispatchMessage{TaskID: "t1", Command: cmd})
	if res.ExitCode != 0 || res.Err != nil {
		t.Fatalf("命令应成功执行: exit=%d err=%v", res.ExitCode, res.Err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(lines) < 2 {
		t.Fatalf("应至少捕获 2 行日志，实际 %d 行: %v", len(lines), lines)
	}
}

// TestStreamingExecutorEmptyCommand 验证空命令视为空任务成功。
func TestStreamingExecutorEmptyCommand(t *testing.T) {
	exec := NewStreamingExecutor("", nil)
	res := exec.Execute(context.Background(), mq.DispatchMessage{TaskID: "t1", Command: "   "})
	if res.ExitCode != 0 {
		t.Errorf("空命令应视为成功，退出码 %d", res.ExitCode)
	}
}

// TestStreamingExecutorNonZeroExit 验证失败命令返回非零退出码与输出。
func TestStreamingExecutorNonZeroExit(t *testing.T) {
	exec := NewStreamingExecutor("", nil)
	var cmd string
	if runtime.GOOS == "windows" {
		cmd = "exit 3"
	} else {
		cmd = "exit 3"
	}
	res := exec.Execute(context.Background(), mq.DispatchMessage{TaskID: "t1", Command: cmd})
	if res.ExitCode == 0 {
		t.Error("失败命令应返回非零退出码")
	}
}
