// 本文件实现流式任务执行器。
//
// 相比 CommandExecutor 一次性返回合并输出，StreamingExecutor 在进程运行过程中
// 逐行捕获 stdout/stderr，通过回调实时推送，从而支撑"实时日志流"能力；
// 同时仍保留完整输出用于失败诊断与归档。
package agent

import (
	"bufio"
	"context"
	"os/exec"
	"runtime"
	"strings"
	"sync"

	"qingtian/internal/mq"
)

// LineSink 是日志行回调，执行器每捕获一行即调用一次。
//
// 参数：
//   - taskID: 产生该行日志的任务 ID。
//   - line:   一行日志内容（不含换行符）。
type LineSink func(taskID, line string)

// StreamingExecutor 是支持逐行流式输出的执行器实现。
//
// 字段：
//   - WorkDir:   命令执行的工作目录。
//   - sink:      日志行回调；为空时退化为仅收集输出。
//   - maxBuffer: 保留在内存中的输出最大字节数，用于诊断（防止超大日志占满内存）。
type StreamingExecutor struct {
	WorkDir   string
	sink      LineSink
	maxBuffer int
}

// NewStreamingExecutor 创建流式执行器。
//
// 参数：
//   - workDir: 工作目录。
//   - sink:    日志行回调，可为 nil。
func NewStreamingExecutor(workDir string, sink LineSink) *StreamingExecutor {
	return &StreamingExecutor{WorkDir: workDir, sink: sink, maxBuffer: 64 * 1024}
}

// Execute 实现 Executor.Execute。
//
// 处理流程：启动进程 -> 并发读取 stdout/stderr 两个管道，逐行回调 sink
// 并累积到诊断缓冲 -> 等待进程结束 -> 返回退出码与输出。
func (e *StreamingExecutor) Execute(ctx context.Context, msg mq.DispatchMessage) ExecResult {
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
	cmd.Env = append(cmd.Environ(), "CUDA_VISIBLE_DEVICES="+strings.Join(msg.AllocatedGPUs, ","))

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return ExecResult{ExitCode: -1, Err: err}
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return ExecResult{ExitCode: -1, Err: err}
	}

	if err := cmd.Start(); err != nil {
		return ExecResult{ExitCode: -1, Err: err}
	}

	var (
		mu  sync.Mutex
		buf strings.Builder
		wg  sync.WaitGroup
	)
	// consume 逐行读取一个管道，回调 sink 并累积诊断缓冲。
	consume := func(scanner *bufio.Scanner) {
		defer wg.Done()
		for scanner.Scan() {
			line := scanner.Text()
			if e.sink != nil {
				e.sink(msg.TaskID, line)
			}
			mu.Lock()
			if buf.Len() < e.maxBuffer {
				buf.WriteString(line)
				buf.WriteByte('\n')
			}
			mu.Unlock()
		}
	}
	wg.Add(2)
	go consume(bufio.NewScanner(stdout))
	go consume(bufio.NewScanner(stderr))
	wg.Wait()

	err = cmd.Wait()
	result := ExecResult{Output: buf.String()}
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
