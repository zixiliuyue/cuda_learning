// Package httpx 提供 HTTP 层统一的响应封装与错误映射。
//
// 所有 API 返回统一的 JSON 结构，方便前端统一处理成功与失败。
package httpx

import (
	"encoding/json"
	"errors"
	"net/http"

	"qingtian/pkg/apperr"
)

// Response 是所有 API 的统一响应结构。
//
// 字段：
//   - Code:    业务码，"OK" 表示成功，其余为 apperr 中定义的错误码。
//   - Message: 可读提示信息。
//   - Data:    成功时的业务数据；失败时为 nil。
type Response struct {
	Code    string      `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// WriteJSON 将任意数据以统一成功结构写入响应。
//
// 参数：
//   - w:    HTTP 响应写入器。
//   - code: HTTP 状态码（如 http.StatusOK）。
//   - data: 业务数据。
func WriteJSON(w http.ResponseWriter, code int, data interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(Response{Code: "OK", Message: "success", Data: data})
}

// WriteError 将错误映射为合适的 HTTP 状态码与统一错误结构。
//
// 参数：
//   - w:   HTTP 响应写入器。
//   - err: 错误对象。若为 *apperr.Error，会按其 Code 映射状态码；否则视为内部错误。
func WriteError(w http.ResponseWriter, err error) {
	var ae *apperr.Error
	if !errors.As(err, &ae) {
		// 非业务错误，一律作为 500 内部错误处理，避免泄露实现细节。
		ae = apperr.Internal(err.Error())
	}
	status := statusFromCode(ae.Code)
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(Response{Code: ae.Code, Message: ae.Message})
}

// statusFromCode 将业务错误码映射为 HTTP 状态码。
func statusFromCode(code string) int {
	switch code {
	case apperr.CodeInvalidArgument:
		return http.StatusBadRequest
	case apperr.CodeUnauthorized:
		return http.StatusUnauthorized
	case apperr.CodeForbidden:
		return http.StatusForbidden
	case apperr.CodeNotFound:
		return http.StatusNotFound
	case apperr.CodeConflict:
		return http.StatusConflict
	case apperr.CodeExhausted:
		return http.StatusTooManyRequests
	default:
		return http.StatusInternalServerError
	}
}
