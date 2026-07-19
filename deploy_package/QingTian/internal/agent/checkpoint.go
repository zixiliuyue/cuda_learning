// 本文件实现 Agent 端的 Checkpoint（断点存档）能力。
//
// 长任务（如大模型训练）一旦节点宕机，进度全丢是训练类用户的刚需痛点。
// Checkpointer 周期性地把任务工作目录的存档上传到对象存储（MinIO），
// 重调度时再由 Agent 拉取恢复，从而实现"断点续跑"、提升可靠性口碑。
package agent

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"time"

	"qingtian/internal/storage"
)

// Checkpointer 负责把任务的存档周期性上传到对象存储。
//
// 字段：
//   - store:    对象存储实现。
//   - interval: 存档周期。
type Checkpointer struct {
	store    storage.ObjectStore
	interval time.Duration
}

// NewCheckpointer 创建 Checkpointer。
//
// 参数：
//   - store:    对象存储实现。
//   - interval: 存档周期；<=0 时取 5 分钟。
func NewCheckpointer(store storage.ObjectStore, interval time.Duration) *Checkpointer {
	if interval <= 0 {
		interval = 5 * time.Minute
	}
	return &Checkpointer{store: store, interval: interval}
}

// Save 立即把本地文件存档上传到对象存储的指定路径。
//
// 参数：
//   - ctx:        上下文。
//   - localPath:  本地存档文件路径。
//   - objectPath: 对象存储中的目标路径（对应 Task.CheckpointPath）。
//
// 返回值：
//   - error: 读取本地文件或上传失败时返回。
func (c *Checkpointer) Save(ctx context.Context, localPath, objectPath string) error {
	data, err := os.ReadFile(localPath)
	if err != nil {
		return err
	}
	return c.store.Put(ctx, objectPath, bytes.NewReader(data), int64(len(data)))
}

// Restore 从对象存储拉取存档并写入本地文件，供任务恢复。
//
// 参数：
//   - ctx:        上下文。
//   - objectPath: 对象存储中的存档路径。
//   - localPath:  写入的本地文件路径。
//
// 返回值：
//   - error: 下载或写入失败时返回。
func (c *Checkpointer) Restore(ctx context.Context, objectPath, localPath string) error {
	rc, err := c.store.Get(ctx, objectPath)
	if err != nil {
		return err
	}
	defer rc.Close()
	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		return err
	}
	f, err := os.Create(localPath)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, rc)
	return err
}

// RunPeriodic 启动周期性存档循环，直到 ctx 取消。
//
// 每隔 interval 调用一次 Save，把 localPath 上传到 objectPath。
// 存档失败仅忽略当次（下次周期重试），不中断任务执行。
//
// 参数：
//   - ctx:        上下文，取消时退出循环。
//   - localPath:  本地存档文件路径。
//   - objectPath: 对象存储目标路径。
func (c *Checkpointer) RunPeriodic(ctx context.Context, localPath, objectPath string) {
	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			_ = c.Save(ctx, localPath, objectPath)
		}
	}
}
