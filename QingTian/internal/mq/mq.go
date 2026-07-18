// Package mq 定义消息队列的抽象接口与消息类型。
//
// 平台使用 NSQ 作为消息队列实现任务下发与状态回传。通过接口抽象，
// 业务代码不直接依赖 NSQ，单元测试可使用内存实现。
package mq

import "context"

// 平台使用的消息主题（topic）常量。
const (
	// TopicTaskStatus 用于 Agent 向控制面回传任务状态。
	TopicTaskStatus = "task_status"
	// TopicNodeEvent 用于节点上下线等事件广播。
	TopicNodeEvent = "node_event"
	// TopicTaskLog 用于 Agent 向控制面流式回传任务日志行。
	TopicTaskLog = "task_log"
)

// TaskDispatchTopic 返回指定节点的任务下发主题。
//
// 每个节点使用独立主题，确保任务只投递给目标节点的 Agent。
//
// 参数：
//   - nodeID: 目标节点 ID。
//
// 返回值：
//   - 形如 "task_dispatch_<nodeID>" 的主题名。
func TaskDispatchTopic(nodeID string) string {
	return "task_dispatch_" + nodeID
}

// DispatchMessage 是任务下发消息体，由控制面发送给 Agent。
type DispatchMessage struct {
	TaskID         string   `json:"task_id"`
	NodeID         string   `json:"node_id"`
	Image          string   `json:"image"`
	Command        string   `json:"command"`
	AllocatedGPUs  []string `json:"allocated_gpus"`
	DatasetPath    string   `json:"dataset_path"`
	ResultPath     string   `json:"result_path"`
	ShareMode      string   `json:"share_mode,omitempty"`      // 共享模式（exclusive/memory/mig）
	CheckpointPath string   `json:"checkpoint_path,omitempty"` // 断点续跑时恢复用的存档路径
}

// StatusMessage 是任务状态回传消息体，由 Agent 发送给控制面。
type StatusMessage struct {
	TaskID   string `json:"task_id"`
	NodeID   string `json:"node_id"`
	Status   string `json:"status"`    // 对应 domain.TaskStatus
	ExitCode int    `json:"exit_code"` // 进程退出码
	Message  string `json:"message"`   // 附加信息，如失败原因
	Output   string `json:"output"`    // 执行输出片段，供控制面做失败诊断
}

// LogMessage 是任务日志行回传消息体，由 Agent 流式发送给控制面。
type LogMessage struct {
	TaskID string `json:"task_id"`
	Line   string `json:"line"`
}

// Handler 是消息消费回调函数。
//
// 返回 error 表示处理失败，实现方可据此决定是否重新投递（requeue）。
type Handler func(ctx context.Context, body []byte) error

// Producer 定义消息生产者能力。
type Producer interface {
	// Publish 向指定主题发布一条消息。
	Publish(ctx context.Context, topic string, body []byte) error
	// Close 释放生产者资源。
	Close() error
}

// Consumer 定义消息消费者能力。
type Consumer interface {
	// Subscribe 订阅指定主题的指定消费通道，并注册处理回调。
	//
	// 参数：
	//   - topic:   主题名。
	//   - channel: 消费通道名（NSQ 中同一 channel 下的消费者负载均衡）。
	//   - handler: 消息处理回调。
	Subscribe(topic, channel string, handler Handler) error
	// Close 停止消费并释放资源。
	Close() error
}
