// 本文件实现日志服务：任务实时日志的追加、读取、订阅与归档。
//
// 日志服务是"实时日志流"能力的控制面入口，编排日志流（logstream）与
// 对象存储（归档）：Agent 消费下发时把日志写入日志流，前端经 SSE 订阅；
// 任务结束后把完整日志归档到 MinIO 长期留存。
package service

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"

	"qingtian/internal/logstream"
	"qingtian/internal/mq"
	"qingtian/internal/storage"
	"qingtian/pkg/apperr"
)

// LogService 提供任务日志的读写与归档能力。
//
// 字段：
//   - streamer: 实时日志流（内存或 Redis 实现）。
//   - store:    对象存储，用于日志归档。
type LogService struct {
	streamer logstream.Streamer
	store    storage.ObjectStore
}

// NewLogService 创建日志服务。
//
// 参数：
//   - streamer: 日志流实现。
//   - store:    对象存储实现（可为 nil，则不归档）。
func NewLogService(streamer logstream.Streamer, store storage.ObjectStore) *LogService {
	return &LogService{streamer: streamer, store: store}
}

// Append 追加一行任务日志，返回分配到的序号。
func (s *LogService) Append(ctx context.Context, taskID, line string) (int64, error) {
	return s.streamer.Append(ctx, taskID, line)
}

// HandleLog 是任务日志回传消息的消费回调。
//
// Agent 通过 TopicTaskLog 流式回传日志行，控制面据此写入日志流供前端订阅。
//
// 参数：
//   - ctx:  上下文。
//   - body: LogMessage 的 JSON 字节。
//
// 返回值：
//   - error: 解析失败时返回 CodeInvalidArgument。
func (s *LogService) HandleLog(ctx context.Context, body []byte) error {
	var msg mq.LogMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		return apperr.Wrap(apperr.CodeInvalidArgument, "日志消息解析失败", err)
	}
	_, err := s.streamer.Append(ctx, msg.TaskID, msg.Line)
	return err
}

// History 读取任务从 afterSeq（不含）之后的历史日志，用于首次加载与断点续读。
func (s *LogService) History(ctx context.Context, taskID string, afterSeq int64) ([]logstream.LogLine, error) {
	return s.streamer.ReadFrom(ctx, taskID, afterSeq)
}

// Subscribe 订阅任务后续新增日志，用于 SSE/WebSocket 实时推送。
func (s *LogService) Subscribe(ctx context.Context, taskID string) (<-chan logstream.LogLine, error) {
	return s.streamer.Subscribe(ctx, taskID)
}

// Archive 将任务的全部日志归档到对象存储，并关闭其日志流。
//
// 该方法在任务进入终态时调用：先读取全部日志拼成文本上传到 objectPath，
// 再关闭日志流通知订阅者结束。归档失败不影响日志流关闭。
//
// 参数：
//   - ctx:        上下文。
//   - taskID:     任务 ID。
//   - objectPath: 归档在对象存储中的路径（对应 Task.LogPath）。
//
// 返回值：
//   - error: 归档失败时返回；store 为 nil 时跳过归档直接关闭流。
func (s *LogService) Archive(ctx context.Context, taskID, objectPath string) error {
	defer func() { _ = s.streamer.Close(ctx, taskID) }()
	if s.store == nil {
		return nil
	}
	lines, err := s.streamer.ReadFrom(ctx, taskID, 0)
	if err != nil {
		return err
	}
	var sb strings.Builder
	for _, l := range lines {
		sb.WriteString(l.Content)
		sb.WriteByte('\n')
	}
	data := []byte(sb.String())
	return s.store.Put(ctx, objectPath, bytes.NewReader(data), int64(len(data)))
}
