// 本文件实现任务状态机，集中管理任务状态的合法转移规则。
//
// 将状态转移规则收敛到领域层，可避免各处直接修改 Task.Status 导致状态错乱，
// 是保证任务生命周期一致性的关键。
package domain

import (
	"qingtian/pkg/apperr"
	"time"
)

// allowedTransitions 定义每个状态允许转移到的目标状态集合。
//
// 设计说明：
//   - pending 可进入 scheduled（被调度）或 canceled（提交后立即取消）。
//   - scheduled 可进入 running（Agent 拉起）、failed（拉起失败）或 canceled。
//   - running 可进入终态 succeeded/failed，或被 canceled。
//   - succeeded/canceled 均为终态，不允许再转移。
//   - failed 允许转移回 pending 用于自动重试（还未耗尽重试次数时）。
var allowedTransitions = map[TaskStatus]map[TaskStatus]bool{
	TaskPending: {
		TaskScheduled: true,
		TaskCanceled:  true,
	},
	TaskScheduled: {
		TaskRunning:  true,
		TaskFailed:   true,
		TaskCanceled: true,
	},
	TaskRunning: {
		TaskSucceeded: true,
		TaskFailed:    true,
		TaskCanceled:  true,
	},
	TaskSucceeded: {},
	TaskFailed: {
		TaskPending: true, // 允许自动重试：失败 → 待调度
	},
	TaskCanceled: {},
}

// CanTransition 判断任务能否从 from 状态转移到 to 状态。
//
// 参数：
//   - from: 当前状态。
//   - to:   目标状态。
//
// 返回值：
//   - bool: true 表示转移合法。
func CanTransition(from, to TaskStatus) bool {
	targets, ok := allowedTransitions[from]
	if !ok {
		return false
	}
	return targets[to]
}

// Transition 尝试将任务转移到目标状态。
//
// 该方法会校验转移合法性，并在合法时更新 Status，同时维护相关时间戳：
//   - 进入 running 时记录 StartedAt。
//   - 进入 succeeded/failed/canceled 时记录 FinishedAt。
//
// 参数：
//   - to:  目标状态。
//   - now: 当前时间（由调用方传入，便于测试注入固定时间）。
//
// 返回值：
//   - error: 转移非法时返回 CodeConflict 业务错误；合法时返回 nil。
//
// 注意事项：
//   - 该方法会直接修改接收者 t 的字段，调用方应确保持有可写引用。
func (t *Task) Transition(to TaskStatus, now time.Time) error {
	if !CanTransition(t.Status, to) {
		return apperr.Conflict(
			"任务状态非法转移: " + string(t.Status) + " -> " + string(to),
		)
	}
	t.Status = to
	t.UpdatedAt = now
	switch to {
	case TaskRunning:
		started := now
		t.StartedAt = &started
	case TaskSucceeded, TaskFailed, TaskCanceled:
		finished := now
		t.FinishedAt = &finished
	}
	return nil
}
