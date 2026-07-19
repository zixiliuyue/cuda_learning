package agent

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"qingtian/internal/storage"
)

// TestCheckpointSaveAndRestore 验证存档上传与恢复到本地。
func TestCheckpointSaveAndRestore(t *testing.T) {
	store := storage.NewMemoryStore()
	cp := NewCheckpointer(store, 0)
	ctx := context.Background()

	dir := t.TempDir()
	local := filepath.Join(dir, "model.ckpt")
	if err := os.WriteFile(local, []byte("weights-v1"), 0o644); err != nil {
		t.Fatal(err)
	}

	// 存档到对象存储。
	if err := cp.Save(ctx, local, "ckpt/task-1/latest"); err != nil {
		t.Fatalf("存档应成功: %v", err)
	}

	// 恢复到新的本地路径。
	restored := filepath.Join(dir, "restored", "model.ckpt")
	if err := cp.Restore(ctx, "ckpt/task-1/latest", restored); err != nil {
		t.Fatalf("恢复应成功: %v", err)
	}
	data, err := os.ReadFile(restored)
	if err != nil {
		t.Fatalf("恢复文件应可读: %v", err)
	}
	if string(data) != "weights-v1" {
		t.Errorf("恢复内容不一致，实际 %s", string(data))
	}
}

// TestHeteroCollectors 验证异构算力采集器返回正确厂商。
func TestHeteroCollectors(t *testing.T) {
	asc, _ := NewAscendMockCollector(2, 32768).Collect()
	if len(asc) != 2 || asc[0].Vendor != "ascend" {
		t.Errorf("昇腾采集器结果错误: %+v", asc)
	}
	mlu, _ := NewCambriconMockCollector(1, 24576).Collect()
	if len(mlu) != 1 || mlu[0].Vendor != "cambricon" {
		t.Errorf("寒武纪采集器结果错误: %+v", mlu)
	}
}
