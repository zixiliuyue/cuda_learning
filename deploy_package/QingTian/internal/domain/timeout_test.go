package domain

import (
	"testing"
	"time"
)

// TestTaskIsTimedOut 验证运行超时判定的各种边界。
func TestTaskIsTimedOut(t *testing.T) {
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	started := now.Add(-2 * time.Hour)

	cases := []struct {
		name string
		task Task
		want bool
	}{
		{
			name: "运行中且超时",
			task: Task{Status: TaskRunning, TimeoutSeconds: 60, StartedAt: &started},
			want: true,
		},
		{
			name: "运行中未超时",
			task: Task{Status: TaskRunning, TimeoutSeconds: 36000, StartedAt: &started},
			want: false,
		},
		{
			name: "未设置超时",
			task: Task{Status: TaskRunning, TimeoutSeconds: 0, StartedAt: &started},
			want: false,
		},
		{
			name: "未开始运行",
			task: Task{Status: TaskScheduled, TimeoutSeconds: 60, StartedAt: nil},
			want: false,
		},
		{
			name: "已结束不再判定",
			task: Task{Status: TaskSucceeded, TimeoutSeconds: 60, StartedAt: &started},
			want: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.task.IsTimedOut(now); got != c.want {
				t.Errorf("IsTimedOut = %v，期望 %v", got, c.want)
			}
		})
	}
}

// TestTaskIsActive 验证活动状态判定。
func TestTaskIsActive(t *testing.T) {
	active := []TaskStatus{TaskScheduled, TaskRunning}
	inactive := []TaskStatus{TaskPending, TaskSucceeded, TaskFailed, TaskCanceled}

	for _, s := range active {
		if !(&Task{Status: s}).IsActive() {
			t.Errorf("状态 %s 应为活动状态", s)
		}
	}
	for _, s := range inactive {
		if (&Task{Status: s}).IsActive() {
			t.Errorf("状态 %s 不应为活动状态", s)
		}
	}
}
