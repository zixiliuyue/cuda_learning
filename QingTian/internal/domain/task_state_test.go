package domain

import (
	"testing"
	"time"
)

// TestCanTransition 验证任务状态机的合法与非法转移。
func TestCanTransition(t *testing.T) {
	cases := []struct {
		name string
		from TaskStatus
		to   TaskStatus
		want bool
	}{
		{"pending到scheduled合法", TaskPending, TaskScheduled, true},
		{"pending到canceled合法", TaskPending, TaskCanceled, true},
		{"pending到running非法", TaskPending, TaskRunning, false},
		{"scheduled到running合法", TaskScheduled, TaskRunning, true},
		{"running到succeeded合法", TaskRunning, TaskSucceeded, true},
		{"running到failed合法", TaskRunning, TaskFailed, true},
		{"succeeded为终态不可转移", TaskSucceeded, TaskRunning, false},
		{"failed允许转移到pending用于自动重试", TaskFailed, TaskPending, true},
		{"canceled为终态不可转移", TaskCanceled, TaskRunning, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := CanTransition(c.from, c.to); got != c.want {
				t.Errorf("CanTransition(%s, %s) = %v, 期望 %v", c.from, c.to, got, c.want)
			}
		})
	}
}

// TestTransitionUpdatesTimestamps 验证转移时时间戳被正确维护。
func TestTransitionUpdatesTimestamps(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	task := &Task{Status: TaskScheduled}

	// scheduled -> running 应记录 StartedAt。
	if err := task.Transition(TaskRunning, now); err != nil {
		t.Fatalf("合法转移不应报错: %v", err)
	}
	if task.StartedAt == nil || !task.StartedAt.Equal(now) {
		t.Errorf("进入 running 应记录 StartedAt")
	}

	// running -> succeeded 应记录 FinishedAt。
	finish := now.Add(time.Hour)
	if err := task.Transition(TaskSucceeded, finish); err != nil {
		t.Fatalf("合法转移不应报错: %v", err)
	}
	if task.FinishedAt == nil || !task.FinishedAt.Equal(finish) {
		t.Errorf("进入 succeeded 应记录 FinishedAt")
	}
}

// TestTransitionIllegalReturnsError 验证非法转移返回错误且不修改状态。
func TestTransitionIllegalReturnsError(t *testing.T) {
	task := &Task{Status: TaskPending}
	if err := task.Transition(TaskRunning, time.Now()); err == nil {
		t.Fatal("pending->running 应返回错误")
	}
	if task.Status != TaskPending {
		t.Errorf("非法转移后状态不应改变，当前=%s", task.Status)
	}
}
