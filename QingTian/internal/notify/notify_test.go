package notify

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

// stubDoer 是 httpDoer 的测试桩，记录最近一次请求并返回预设响应。
type stubDoer struct {
	lastBody   string
	statusCode int
	err        error
}

func (s *stubDoer) Do(req *http.Request) (*http.Response, error) {
	if s.err != nil {
		return nil, s.err
	}
	b, _ := io.ReadAll(req.Body)
	s.lastBody = string(b)
	code := s.statusCode
	if code == 0 {
		code = 200
	}
	return &http.Response{
		StatusCode: code,
		Body:       io.NopCloser(strings.NewReader("{}")),
	}, nil
}

// TestMemoryNotifier 验证内存通知器记录消息。
func TestMemoryNotifier(t *testing.T) {
	n := NewMemoryNotifier()
	_ = n.Notify(context.Background(), Message{Title: "t", Content: "c"})
	if len(n.Sent) != 1 || n.Sent[0].Title != "t" {
		t.Errorf("内存通知器应记录消息，实际 %+v", n.Sent)
	}
}

// TestWebhookFeishuPayload 验证飞书报文格式。
func TestWebhookFeishuPayload(t *testing.T) {
	stub := &stubDoer{}
	n := &WebhookNotifier{url: "http://example.com/hook", platform: PlatformFeishu, client: stub}
	if err := n.Notify(context.Background(), Message{Title: "任务成功", Content: "详情"}); err != nil {
		t.Fatalf("发送应成功: %v", err)
	}
	var payload map[string]interface{}
	_ = json.Unmarshal([]byte(stub.lastBody), &payload)
	if payload["msg_type"] != "text" {
		t.Errorf("飞书报文应含 msg_type=text，实际 %v", payload)
	}
}

// TestWebhookWeComPayload 验证企业微信报文格式。
func TestWebhookWeComPayload(t *testing.T) {
	stub := &stubDoer{}
	n := &WebhookNotifier{url: "http://example.com/hook", platform: PlatformWeCom, client: stub}
	_ = n.Notify(context.Background(), Message{Title: "任务失败", Content: "详情"})
	var payload map[string]interface{}
	_ = json.Unmarshal([]byte(stub.lastBody), &payload)
	if payload["msgtype"] != "text" {
		t.Errorf("企微报文应含 msgtype=text，实际 %v", payload)
	}
}

// TestWebhookNon2xxReturnsError 验证非 2xx 状态码返回错误。
func TestWebhookNon2xxReturnsError(t *testing.T) {
	stub := &stubDoer{statusCode: 500}
	n := &WebhookNotifier{url: "http://example.com/hook", platform: PlatformFeishu, client: stub}
	if err := n.Notify(context.Background(), Message{Title: "t", Content: "c"}); err == nil {
		t.Error("非 2xx 状态码应返回错误")
	}
}

// TestWebhookEmptyURL 验证空地址报错。
func TestWebhookEmptyURL(t *testing.T) {
	n := NewWebhookNotifier("", PlatformFeishu)
	if err := n.Notify(context.Background(), Message{Title: "t"}); err == nil {
		t.Error("空 Webhook 地址应报错")
	}
}
