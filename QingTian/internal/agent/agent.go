// 本文件实现 Agent 主运行器，串联注册、心跳、任务消费与执行、状态回传。
package agent

import (
	"context"
	"encoding/json"
	"log/slog"
	"time"

	"qingtian/internal/domain"
	"qingtian/internal/mq"
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
}

// Config 是创建 Agent 所需的配置。
type Config struct {
	Hostname          string
	IP                string
	HeartbeatInterval time.Duration
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

// handleDispatch 是任务下发消息的处理回调。
//
// 处理流程：解析消息 -> 回传 running -> 执行 -> 回传 succeeded/failed。
func (a *Agent) handleDispatch(ctx context.Context, body []byte) error {
	var msg mq.DispatchMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		return err
	}
	a.log.Info("收到任务下发", "task_id", msg.TaskID)

	// 回传运行中状态。
	a.reportStatus(ctx, msg.TaskID, domain.TaskRunning, 0, "任务开始执行", "")

	// 执行任务。
	result := a.executor.Execute(ctx, msg)

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
