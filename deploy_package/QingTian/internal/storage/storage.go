// Package storage 定义对象存储的抽象接口。
//
// 平台使用 MinIO 存储数据集、模型权重、任务结果与日志等大文件。
// 通过接口抽象，业务代码不直接依赖 MinIO，单元测试可使用内存实现。
package storage

import (
	"context"
	"io"
)

// ObjectStore 定义对象存储能力。
type ObjectStore interface {
	// Put 上传一个对象。
	//
	// 参数：
	//   - objectPath: 对象在存储桶内的路径（键）。
	//   - reader:     数据源。
	//   - size:       数据大小（字节）；未知时可传 -1。
	Put(ctx context.Context, objectPath string, reader io.Reader, size int64) error
	// Get 下载一个对象，返回可读流，调用方负责关闭。
	Get(ctx context.Context, objectPath string) (io.ReadCloser, error)
	// Delete 删除一个对象。
	Delete(ctx context.Context, objectPath string) error
	// PresignedGetURL 生成对象的临时下载 URL，用于前端直连下载。
	//
	// 参数：
	//   - expirySeconds: URL 有效期（秒）。
	PresignedGetURL(ctx context.Context, objectPath string, expirySeconds int) (string, error)
	// PresignedPutURL 生成对象的临时上传 URL，用于前端直传大文件，
	// 避免数据集经过控制面造成带宽与内存压力。
	//
	// 参数：
	//   - expirySeconds: URL 有效期（秒）。
	PresignedPutURL(ctx context.Context, objectPath string, expirySeconds int) (string, error)
}
