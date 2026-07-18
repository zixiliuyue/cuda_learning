// 本文件实现任务执行器。
//
// Executor 负责在节点上实际拉起任务进程。为跨平台可编译与可测试，
// 执行通过接口抽象：默认提供 CommandExecutor（本地进程执行），
// 生产环境可替换为容器执行器（nvidia-container-runtime）。
package agent

import (
	"context"
	"os/exec"
	"runtime"
	"strings"

	"qingtian/internal/mq"
)

// ExecResult 表示一次任务执行的结果。
type ExecResult struct {
	ExitCode int    // 进程退出码
	Output   string // 合并的标准输出/错误（截断保存）
	Err      error  // 执行过程中的错误（如无法启动）
}

// Executor 定义任务执行能力。
type Executor interface {
	// Execute 执行一个任务下发消息，返回执行结果。
	Execute(ctx context.Context, msg mq.DispatchMessage) ExecResult
}

// CommandExecutor 是基于本地命令的执行器实现。
//
// 它将下发消息中的 Command 作为 shell 命令执行，并注入 CUDA_VISIBLE_DEVICES
// 环境变量以限定可见 GPU。适用于裸机/开发环境；生产建议使用容器执行器。
type CommandExecutor struct {
	// WorkDir 是命令执行的工作目录。
	WorkDir string
}

// NewCommandExecutor 创建本地命令执行器。
func NewCommandExecutor(workDir string) *CommandExecutor {
	return &CommandExecutor{WorkDir: workDir}
}

// Execute 实现 Executor.Execute。
//
// 参数：
//   - ctx: 上下文，可用于取消执行。
//   - msg: 任务下发消息。
//
// 返回值：
//   - ExecResult: 执行结果（退出码、输出、错误）。
//
// 注意事项：
//   - Windows 下使用 powershell，其它平台使用 sh。
//   - 通过 CUDA_VISIBLE_DEVICES 限定任务可见的 GPU（软隔离）。
func (e *CommandExecutor) Execute(ctx context.Context, msg mq.DispatchMessage) ExecResult {
	if strings.TrimSpace(msg.Command) == "" {
		return ExecResult{ExitCode: 0, Output: "空命令，视为空任务成功"}
	}

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "powershell", "-NoProfile", "-Command", msg.Command)
	} else {
		cmd = exec.CommandContext(ctx, "sh", "-c", msg.Command)
	}
	if e.WorkDir != "" {
		cmd.Dir = e.WorkDir
	}
	// 注入 CUDA_VISIBLE_DEVICES，限定任务可见 GPU。
	cmd.Env = append(cmd.Environ(), "CUDA_VISIBLE_DEVICES="+strings.Join(msg.AllocatedGPUs, ","))

	out, err := cmd.CombinedOutput()
	result := ExecResult{Output: string(out)}
	if err != nil {
		result.Err = err
		if exitErr, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exitErr.ExitCode()
		} else {
			result.ExitCode = -1
		}
	}
	return result
}
