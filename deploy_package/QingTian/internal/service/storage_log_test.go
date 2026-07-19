package service

import (
	"context"
	"strings"
	"testing"

	"qingtian/internal/logstream"
	"qingtian/internal/storage"
)

// TestStoragePresignUpload 验证生成上传预签名 URL 并按用户隔离命名空间。
func TestStoragePresignUpload(t *testing.T) {
	svc := NewStorageService(storage.NewMemoryStore(), 600)
	ctx := context.Background()

	url, path, err := svc.PresignUpload(ctx, "user-1", "train.zip")
	if err != nil {
		t.Fatalf("生成上传 URL 应成功: %v", err)
	}
	if !strings.Contains(path, "datasets/user-1/") {
		t.Errorf("对象路径应按用户隔离，实际 %s", path)
	}
	if !strings.Contains(url, "op=put") {
		t.Errorf("应为上传 URL，实际 %s", url)
	}

	// 目录穿越文件名应被清洗。
	_, path2, _ := svc.PresignUpload(ctx, "user-1", "../../etc/passwd")
	if strings.Contains(path2, "..") {
		t.Errorf("文件名应被清洗以防目录穿越，实际 %s", path2)
	}
	// 空文件名应报错。
	if _, _, err := svc.PresignUpload(ctx, "user-1", "  "); err == nil {
		t.Error("空文件名应报错")
	}
}

// TestStoragePresignDownload 验证生成下载预签名 URL。
func TestStoragePresignDownload(t *testing.T) {
	svc := NewStorageService(storage.NewMemoryStore(), 600)
	ctx := context.Background()
	url, err := svc.PresignDownload(ctx, "results/task-1/out.txt")
	if err != nil {
		t.Fatalf("生成下载 URL 应成功: %v", err)
	}
	if url == "" {
		t.Error("下载 URL 不应为空")
	}
	if _, err := svc.PresignDownload(ctx, ""); err == nil {
		t.Error("空路径应报错")
	}
}

// TestLogServiceAppendHistoryArchive 验证日志追加、读取与归档。
func TestLogServiceAppendHistoryArchive(t *testing.T) {
	streamer := logstream.NewMemoryStreamer()
	store := storage.NewMemoryStore()
	svc := NewLogService(streamer, store)
	ctx := context.Background()

	_, _ = svc.Append(ctx, "task-1", "epoch 1 loss=0.5")
	_, _ = svc.Append(ctx, "task-1", "epoch 2 loss=0.3")

	hist, _ := svc.History(ctx, "task-1", 0)
	if len(hist) != 2 {
		t.Fatalf("应有 2 行历史日志，实际 %d", len(hist))
	}

	// 归档后应能从对象存储读回完整日志。
	if err := svc.Archive(ctx, "task-1", "logs/task-1.log"); err != nil {
		t.Fatalf("归档应成功: %v", err)
	}
	rc, err := store.Get(ctx, "logs/task-1.log")
	if err != nil {
		t.Fatalf("归档对象应可读取: %v", err)
	}
	defer rc.Close()
	buf := make([]byte, 128)
	n, _ := rc.Read(buf)
	if !strings.Contains(string(buf[:n]), "epoch 2 loss=0.3") {
		t.Errorf("归档内容应包含全部日志，实际 %s", string(buf[:n]))
	}
}
