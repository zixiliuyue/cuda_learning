// 本文件实现 Agent 主运行器，串联注册、心跳、任务消费与执行、状态回传。
package agent

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"qingtian/internal/domain"
	"qingtian/internal/mq"
	"qingtian/internal/storage"
)

// ControlPlane 定义 Agent 与控制面的交互能力（注册与心跳）。
//
// 通过接口抽象，可注入真实 HTTP 客户端或测试用的内存实现。
type ControlPlane interface {
	// Register 向控制面注册本节点，返回分配的节点 ID。
	Register(ctx context.Context, req RegisterInfo) (string, error)
	// Heartbeat 上报心跳与实时 GPU 信息。
	Heartbeat(ctx context.Context, nodeID string, gpus []domain.GPU) error
}

// RegisterInfo 是节点注册信息。
type RegisterInfo struct {
	Hostname      string       `json:"hostname"`
	IP            string       `json:"ip"`
	GPUs          []domain.GPU `json:"gpus"`
	DriverVersion string       `json:"driver_version"`
	CUDAVersion   string       `json:"cuda_version"`
}

// Agent 是节点上的常驻组件。
//
// 字段：
//   - hostname:  节点主机名。
//   - collector: GPU 采集器。
//   - executor:  任务执行器。
//   - cp:        控制面客户端。
//   - producer:  消息生产者（回传状态）。
//   - consumer:  消息消费者（接收下发）。
//   - interval:  心跳间隔。
//   - log:       日志器。
//   - nodeID:    注册后分配的节点 ID。
//   - workDir:    任务工作目录根，每个任务创建子目录。
//   - store:      对象存储，用于归档训练产物。
type Agent struct {
	hostname  string
	ip        string
	collector Collector
	executor  Executor
	cp        ControlPlane
	producer  mq.Producer
	consumer  mq.Consumer
	interval  time.Duration
	log       *slog.Logger
	nodeID    string
	workDir   string
	store     storage.ObjectStore

	// running 记录正在执行任务的取消函数，键为任务 ID。
	// 收到控制面的终止指令时，据此取消对应任务的执行上下文。
	mu      sync.Mutex
	running map[string]context.CancelFunc
}

// Config 是创建 Agent 所需的配置。
type Config struct {
	Hostname          string
	IP                string
	HeartbeatInterval time.Duration
	WorkDir           string              // 任务工作目录根路径
	ObjectStore       storage.ObjectStore // 对象存储，用于归档训练产物
}

// New 创建 Agent。
//
// 参数：
//   - cfg:       Agent 配置。
//   - collector: GPU 采集器。
//   - executor:  任务执行器。
//   - cp:        控制面客户端。
//   - producer:  消息生产者。
//   - consumer:  消息消费者。
//   - log:       日志器。
func New(cfg Config, collector Collector, executor Executor, cp ControlPlane, producer mq.Producer, consumer mq.Consumer, log *slog.Logger) *Agent {
	return &Agent{
		hostname:  cfg.Hostname,
		ip:        cfg.IP,
		collector: collector,
		executor:  executor,
		cp:        cp,
		producer:  producer,
		consumer:  consumer,
		interval:  cfg.HeartbeatInterval,
		log:       log,
		workDir:   cfg.WorkDir,
		store:     cfg.ObjectStore,
		running:   make(map[string]context.CancelFunc),
	}
}

// Register 采集本机 GPU 信息并向控制面注册。
//
// 返回值：
//   - error: 采集或注册失败时返回。
func (a *Agent) Register(ctx context.Context) error {
	gpus, err := a.collector.Collect()
	if err != nil {
		return err
	}
	nodeID, err := a.cp.Register(ctx, RegisterInfo{
		Hostname:      a.hostname,
		IP:            a.ip,
		GPUs:          gpus,
		DriverVersion: a.collector.DriverVersion(),
		CUDAVersion:   a.collector.CUDAVersion(),
	})
	if err != nil {
		return err
	}
	a.nodeID = nodeID
	a.log.Info("节点注册成功", "node_id", nodeID)
	return nil
}

// SubscribeDispatch 订阅本节点的任务下发主题，注册执行回调。
//
// 返回值：
//   - error: 订阅失败时返回。
func (a *Agent) SubscribeDispatch() error {
	topic := mq.TaskDispatchTopic(a.nodeID)
	return a.consumer.Subscribe(topic, "agent", a.handleDispatch)
}

// SubscribeControl 订阅本节点的任务控制主题，处理终止等指令。
//
// 返回值：
//   - error: 订阅失败时返回。
func (a *Agent) SubscribeControl() error {
	topic := mq.TaskControlTopic(a.nodeID)
	return a.consumer.Subscribe(topic, "agent", a.handleControl)
}

// handleControl 是任务控制指令的处理回调。
//
// 目前支持 kill 动作：取消对应任务的执行上下文，从而中断正在运行的进程。
func (a *Agent) handleControl(_ context.Context, body []byte) error {
	var msg mq.ControlMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		return err
	}
	if msg.Action != mq.ControlKill {
		return nil
	}
	a.mu.Lock()
	cancel, ok := a.running[msg.TaskID]
	a.mu.Unlock()
	if ok && cancel != nil {
		a.log.Info("收到任务终止指令", "task_id", msg.TaskID, "reason", msg.Reason)
		cancel()
	}
	return nil
}

// handleDispatch 是任务下发消息的处理回调。
//
// 处理流程：解析消息 -> 登记可取消上下文 -> 回传 running -> 执行 -> 回传终态。
func (a *Agent) handleDispatch(ctx context.Context, body []byte) error {
	var msg mq.DispatchMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		return err
	}
	a.log.Info("收到任务下发", "task_id", msg.TaskID)

	// 派生可取消上下文并登记，支持控制面下达终止指令时中断执行。
	execCtx, cancel := context.WithCancel(ctx)
	a.mu.Lock()
	a.running[msg.TaskID] = cancel
	a.mu.Unlock()
	defer func() {
		a.mu.Lock()
		delete(a.running, msg.TaskID)
		a.mu.Unlock()
		cancel()
	}()

	// 回传运行中状态。
	a.reportStatus(ctx, msg.TaskID, domain.TaskRunning, 0, "任务开始执行", "")

	// 执行任务（使用可取消上下文）。
	result := a.executor.Execute(execCtx, msg)

	// 若因控制指令被取消，则不再覆盖控制面已置的终态（避免与超时/抢占状态冲突）。
	if execCtx.Err() != nil {
		a.log.Info("任务被终止，跳过终态回传", "task_id", msg.TaskID)
		return nil
	}

	// 任务执行成功且指定了产物路径，尝试打包上传产物到对象存储
	if result.ExitCode == 0 && result.Err == nil && msg.ArtifactsPath != "" && a.store != nil {
		artifactPath := filepath.Join(a.workDir, msg.TaskID, msg.ArtifactsPath)
		if _, err := os.Stat(artifactPath); !os.IsNotExist(err) {
			objectPath := fmt.Sprintf("artifacts/%s/%s.tgz", msg.TaskID, msg.TaskID)
			if err := a.packAndUploadArtifacts(ctx, artifactPath, objectPath); err != nil {
				a.log.Error("产物上传失败", "task_id", msg.TaskID, "error", err)
			} else {
				a.log.Info("产物上传成功", "task_id", msg.TaskID, "object_path", objectPath)
				// TODO: 需要回调控制面更新产物归档路径
				// 由于控制面已经标记任务成功，这里只记录日志，后续可以通过状态回调更新
			}
		} else {
			a.log.Info("产物路径不存在，跳过上传", "task_id", msg.TaskID, "path", artifactPath)
		}
	}

	// 根据退出码回传终态。
	if result.ExitCode == 0 && result.Err == nil {
		a.reportStatus(ctx, msg.TaskID, domain.TaskSucceeded, 0, "任务执行成功", result.Output)
	} else {
		reason := "任务执行失败"
		if result.Err != nil {
			reason = result.Err.Error()
		}
		a.reportStatus(ctx, msg.TaskID, domain.TaskFailed, result.ExitCode, reason, result.Output)
	}
	return nil
}

// packAndUploadArtifacts 将指定目录打包为 tgz 并上传到对象存储。
func (a *Agent) packAndUploadArtifacts(ctx context.Context, sourceDir, objectPath string) error {
	// 创建内存缓冲区写入压缩包
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)

	// 遍历源目录并添加到 tar
	err := filepath.Walk(sourceDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			header, err := tar.FileInfoHeader(info, "")
			if err != nil {
				return err
			}
			// 计算相对路径
			relPath, err := filepath.Rel(filepath.Dir(sourceDir), path)
			if err != nil {
				relPath = filepath.Base(path)
			}
			header.Name = relPath
			if err := tw.WriteHeader(header); err != nil {
				return err
			}
			f, err := os.Open(path)
			if err != nil {
				return err
			}
			defer f.Close()
			_, err = io.Copy(tw, f)
			return err
		}
		return nil
	})
	if err != nil {
		return err
	}
	// 关闭 tar 和 gzip writer
	if err := tw.Close(); err != nil {
		return err
	}
	if err := gz.Close(); err != nil {
		return err
	}

	// 上传到对象存储
	size := int64(buf.Len())
	return a.store.Put(ctx, objectPath, bytes.NewReader(buf.Bytes()), size)
}

// reportStatus 通过消息队列向控制面回传任务状态。
func (a *Agent) reportStatus(ctx context.Context, taskID string, status domain.TaskStatus, exitCode int, message, output string) {
	msg := mq.StatusMessage{
		TaskID:   taskID,
		NodeID:   a.nodeID,
		Status:   string(status),
		ExitCode: exitCode,
		Message:  message,
		Output:   output,
	}
	body, _ := json.Marshal(msg)
	if err := a.producer.Publish(ctx, mq.TopicTaskStatus, body); err != nil {
		a.log.Error("回传任务状态失败", "task_id", taskID, "error", err)
	}
}

// RunHeartbeat 启动心跳循环，周期性采集并上报，直到 ctx 取消。
//
// 参数：
//   - ctx: 上下文，取消时退出循环。
func (a *Agent) RunHeartbeat(ctx context.Context) {
	ticker := time.NewTicker(a.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			gpus, err := a.collector.Collect()
			if err != nil {
				a.log.Error("GPU 采集失败", "error", err)
				continue
			}
			if err := a.cp.Heartbeat(ctx, a.nodeID, gpus); err != nil {
				a.log.Error("心跳上报失败", "error", err)
			}
		}
	}
}

// NodeID 返回注册后分配的节点 ID（主要供测试断言）。
func (a *Agent) NodeID() string { return a.nodeID }

// LogSink 返回一个把日志行通过消息队列回传控制面的回调。
//
// 将该回调注入 StreamingExecutor，即可在任务执行过程中把 stdout/stderr
// 逐行推送到控制面日志流，支撑前端实时日志展示。
func (a *Agent) LogSink() LineSink {
	return func(taskID, line string) {
		body, _ := json.Marshal(mq.LogMessage{TaskID: taskID, Line: line})
		if err := a.producer.Publish(context.Background(), mq.TopicTaskLog, body); err != nil {
			a.log.Error("回传任务日志失败", "task_id", taskID, "error", err)
		}
	}
}
