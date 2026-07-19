// 本文件实现 Agent 与控制面交互的 HTTP 客户端。
//
// 通过调用控制面的 REST 接口完成节点注册与心跳上报。
package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"qingtian/internal/domain"
	"qingtian/pkg/apperr"
)

// HTTPControlPlane 是 ControlPlane 的 HTTP 实现。
type HTTPControlPlane struct {
	baseURL string
	client  *http.Client
}

// NewHTTPControlPlane 创建控制面 HTTP 客户端。
//
// 参数：
//   - baseURL: 控制面 API 根地址，例如 http://localhost:8080。
func NewHTTPControlPlane(baseURL string) *HTTPControlPlane {
	return &HTTPControlPlane{baseURL: baseURL, client: &http.Client{}}
}

// Register 实现 ControlPlane.Register。
func (c *HTTPControlPlane) Register(ctx context.Context, req RegisterInfo) (string, error) {
	body, _ := json.Marshal(req)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/api/v1/nodes/register", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(httpReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", apperr.Internal(fmt.Sprintf("注册失败，状态码: %d", resp.StatusCode))
	}
	// 响应结构为 {code,message,data:{...node...}}，这里解析出 data.id。
	var wrapper struct {
		Data domain.Node `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&wrapper); err != nil {
		return "", err
	}
	return wrapper.Data.ID, nil
}

// Heartbeat 实现 ControlPlane.Heartbeat。
func (c *HTTPControlPlane) Heartbeat(ctx context.Context, nodeID string, gpus []domain.GPU) error {
	payload := map[string]interface{}{"gpus": gpus}
	body, _ := json.Marshal(payload)
	url := fmt.Sprintf("%s/api/v1/nodes/%s/heartbeat", c.baseURL, nodeID)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(httpReq)
	if err != nil {
		return err
	}
	defer func() { _, _ = io.Copy(io.Discard, resp.Body); resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return apperr.Internal(fmt.Sprintf("心跳失败，状态码: %d", resp.StatusCode))
	}
	return nil
}
