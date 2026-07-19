// 本文件实现存储服务：打通"数据进-算-果出"的数据闭环。
//
// 存储服务在对象存储（MinIO）之上提供面向业务的能力：
//   - 生成数据集上传的预签名 URL，前端可直传，避免大文件经过控制面。
//   - 生成结果/日志下载的预签名 URL，前端可直连下载。
//   - 直接读写小对象（供服务内部使用）。
package service

import (
	"context"
	"strings"

	"qingtian/internal/storage"
	"qingtian/pkg/apperr"
)

// StorageService 提供数据集/结果/日志的对象存储访问能力。
//
// 字段：
//   - store:        对象存储实现。
//   - defaultExpiry: 预签名 URL 默认有效期（秒）。
type StorageService struct {
	store         storage.ObjectStore
	defaultExpiry int
}

// NewStorageService 创建存储服务。
//
// 参数：
//   - store:         对象存储实现。
//   - defaultExpiry: 预签名 URL 默认有效期（秒）；<=0 时取 3600。
func NewStorageService(store storage.ObjectStore, defaultExpiry int) *StorageService {
	if defaultExpiry <= 0 {
		defaultExpiry = 3600
	}
	return &StorageService{store: store, defaultExpiry: defaultExpiry}
}

// PresignUpload 生成数据集上传的预签名 URL。
//
// 参数：
//   - ctx:       上下文。
//   - owner:     发起用户 ID，用于隔离命名空间，防止越权覆盖他人数据。
//   - filename:  原始文件名。
//
// 返回值：
//   - url:        预签名上传 URL。
//   - objectPath: 对象在存储桶中的路径（供任务提交时引用为 DatasetPath）。
//   - error:      文件名非法或底层存储错误时返回。
func (s *StorageService) PresignUpload(ctx context.Context, owner, filename string) (string, string, error) {
	if strings.TrimSpace(filename) == "" {
		return "", "", apperr.InvalidArgument("文件名不能为空")
	}
	objectPath := "datasets/" + owner + "/" + sanitize(filename)
	url, err := s.store.PresignedPutURL(ctx, objectPath, s.defaultExpiry)
	if err != nil {
		return "", "", err
	}
	return url, objectPath, nil
}

// PresignDownload 生成结果/日志下载的预签名 URL。
//
// 参数：
//   - ctx:        上下文。
//   - objectPath: 待下载对象路径（如任务的 ResultPath / LogPath）。
//
// 返回值：
//   - url:   预签名下载 URL。
//   - error: 路径为空或底层存储错误时返回。
func (s *StorageService) PresignDownload(ctx context.Context, objectPath string) (string, error) {
	if strings.TrimSpace(objectPath) == "" {
		return "", apperr.InvalidArgument("对象路径不能为空")
	}
	return s.store.PresignedGetURL(ctx, objectPath, s.defaultExpiry)
}

// sanitize 清洗文件名，去除路径分隔符以防目录穿越。
func sanitize(name string) string {
	name = strings.ReplaceAll(name, "\\", "/")
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return "unnamed"
	}
	return name
}
