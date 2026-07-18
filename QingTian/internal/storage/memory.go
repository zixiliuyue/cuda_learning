// 本文件提供 ObjectStore 的内存实现。
//
// 内存实现将对象保存在进程内存的 map 中，用于单元测试与本地演示。
package storage

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"sync"

	"qingtian/pkg/apperr"
)

// MemoryStore 是 ObjectStore 的内存实现。
type MemoryStore struct {
	mu      sync.RWMutex
	objects map[string][]byte
}

// NewMemoryStore 创建内存对象存储。
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{objects: make(map[string][]byte)}
}

// Put 实现 ObjectStore.Put。
func (s *MemoryStore) Put(_ context.Context, objectPath string, reader io.Reader, _ int64) error {
	data, err := io.ReadAll(reader)
	if err != nil {
		return apperr.Wrap(apperr.CodeInternal, "读取对象数据失败", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.objects[objectPath] = data
	return nil
}

// Get 实现 ObjectStore.Get。
func (s *MemoryStore) Get(_ context.Context, objectPath string) (io.ReadCloser, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	data, ok := s.objects[objectPath]
	if !ok {
		return nil, apperr.NotFound("对象不存在: " + objectPath)
	}
	return io.NopCloser(bytes.NewReader(data)), nil
}

// Delete 实现 ObjectStore.Delete。
func (s *MemoryStore) Delete(_ context.Context, objectPath string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.objects, objectPath)
	return nil
}

// PresignedGetURL 实现 ObjectStore.PresignedGetURL。
//
// 内存实现无法生成真实可访问的 URL，这里返回一个仅用于测试的伪 URL。
func (s *MemoryStore) PresignedGetURL(_ context.Context, objectPath string, expirySeconds int) (string, error) {
	return fmt.Sprintf("memory://%s?expiry=%d", objectPath, expirySeconds), nil
}

// PresignedPutURL 实现 ObjectStore.PresignedPutURL。
//
// 内存实现返回一个仅用于测试的伪上传 URL，附带 op=put 以区分下载。
func (s *MemoryStore) PresignedPutURL(_ context.Context, objectPath string, expirySeconds int) (string, error) {
	return fmt.Sprintf("memory://%s?expiry=%d&op=put", objectPath, expirySeconds), nil
}
