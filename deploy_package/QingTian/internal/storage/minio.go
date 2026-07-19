// 本文件提供基于 MinIO 的 ObjectStore 实现。
//
// 生产环境使用该实现存储数据集、模型、任务结果与日志等对象。
package storage

import (
	"context"
	"io"
	"net/url"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"

	"qingtian/internal/config"
)

// MinIOStore 是 ObjectStore 的 MinIO 实现。
type MinIOStore struct {
	client *minio.Client
	bucket string
}

// NewMinIOStore 根据配置创建 MinIO 客户端并确保存储桶存在。
//
// 参数：
//   - cfg: MinIO 配置。
//
// 返回值：
//   - *MinIOStore: 存储实例。
//   - error:       客户端创建或建桶失败时返回。
//
// 注意事项：
//   - 会在初始化时检查存储桶是否存在，不存在则创建。
func NewMinIOStore(cfg config.MinIOConfig) (*MinIOStore, error) {
	client, err := minio.New(cfg.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.AccessKey, cfg.SecretKey, ""),
		Secure: cfg.UseSSL,
	})
	if err != nil {
		return nil, err
	}
	ctx := context.Background()
	exists, err := client.BucketExists(ctx, cfg.Bucket)
	if err != nil {
		return nil, err
	}
	if !exists {
		if err := client.MakeBucket(ctx, cfg.Bucket, minio.MakeBucketOptions{}); err != nil {
			return nil, err
		}
	}
	return &MinIOStore{client: client, bucket: cfg.Bucket}, nil
}

// Put 实现 ObjectStore.Put。
func (s *MinIOStore) Put(ctx context.Context, objectPath string, reader io.Reader, size int64) error {
	_, err := s.client.PutObject(ctx, s.bucket, objectPath, reader, size, minio.PutObjectOptions{})
	return err
}

// Get 实现 ObjectStore.Get。
func (s *MinIOStore) Get(ctx context.Context, objectPath string) (io.ReadCloser, error) {
	obj, err := s.client.GetObject(ctx, s.bucket, objectPath, minio.GetObjectOptions{})
	if err != nil {
		return nil, err
	}
	return obj, nil
}

// Delete 实现 ObjectStore.Delete。
func (s *MinIOStore) Delete(ctx context.Context, objectPath string) error {
	return s.client.RemoveObject(ctx, s.bucket, objectPath, minio.RemoveObjectOptions{})
}

// PresignedGetURL 实现 ObjectStore.PresignedGetURL。
func (s *MinIOStore) PresignedGetURL(ctx context.Context, objectPath string, expirySeconds int) (string, error) {
	u, err := s.client.PresignedGetObject(ctx, s.bucket, objectPath,
		time.Duration(expirySeconds)*time.Second, url.Values{})
	if err != nil {
		return "", err
	}
	return u.String(), nil
}

// PresignedPutURL 实现 ObjectStore.PresignedPutURL。
func (s *MinIOStore) PresignedPutURL(ctx context.Context, objectPath string, expirySeconds int) (string, error) {
	u, err := s.client.PresignedPutObject(ctx, s.bucket, objectPath,
		time.Duration(expirySeconds)*time.Second)
	if err != nil {
		return "", err
	}
	return u.String(), nil
}
