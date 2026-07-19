// 本文件提供基于 HTTP Webhook 的飞书 / 企业微信通知实现。
//
// 生产环境使用群机器人 Webhook：向指定 URL POST 一段 JSON，
// 群内即收到消息。两家的报文格式不同，这里各自适配。
package notify

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"qingtian/pkg/apperr"
)

// httpDoer 抽象 HTTP 执行能力，便于单测注入桩。
type httpDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

// WebhookNotifier 是基于群机器人 Webhook 的通知实现。
//
// 字段：
//   - url:      群机器人 Webhook 地址。
//   - platform: 渠道（feishu / wecom），决定报文格式。
//   - client:   HTTP 客户端。
type WebhookNotifier struct {
	url      string
	platform string
	client   httpDoer
}

const (
	// PlatformFeishu 飞书自定义机器人。
	PlatformFeishu = "feishu"
	// PlatformWeCom 企业微信群机器人。
	PlatformWeCom = "wecom"
)

// NewWebhookNotifier 创建 Webhook 通知器。
//
// 参数：
//   - url:      Webhook 地址。
//   - platform: 渠道标识（feishu / wecom），其它值默认按飞书格式。
func NewWebhookNotifier(url, platform string) *WebhookNotifier {
	return &WebhookNotifier{
		url:      url,
		platform: platform,
		client:   &http.Client{Timeout: 5 * time.Second},
	}
}

// Notify 实现 Notifier.Notify，按渠道格式 POST 到 Webhook。
func (n *WebhookNotifier) Notify(ctx context.Context, msg Message) error {
	if n.url == "" {
		return apperr.InvalidArgument("Webhook 地址为空")
	}
	body := n.buildPayload(msg)
	data, err := json.Marshal(body)
	if err != nil {
		return apperr.Wrap(apperr.CodeInternal, "通知报文序列化失败", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, n.url, bytes.NewReader(data))
	if err != nil {
		return apperr.Wrap(apperr.CodeInternal, "构建通知请求失败", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := n.client.Do(req)
	if err != nil {
		return apperr.Wrap(apperr.CodeInternal, "发送通知失败", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode >= 300 {
		return apperr.Wrap(apperr.CodeInternal,
			fmt.Sprintf("通知渠道返回异常状态码: %d", resp.StatusCode), nil)
	}
	return nil
}

// buildPayload 依据渠道构造 Webhook 报文。
func (n *WebhookNotifier) buildPayload(msg Message) map[string]interface{} {
	text := msg.Title + "\n" + msg.Content
	if n.platform == PlatformWeCom {
		// 企业微信 text 消息。
		return map[string]interface{}{
			"msgtype": "text",
			"text":    map[string]string{"content": text},
		}
	}
	// 飞书自定义机器人 text 消息（默认）。
	return map[string]interface{}{
		"msg_type": "text",
		"content":  map[string]string{"text": text},
	}
}
