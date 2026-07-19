// Package apperr 定义平台统一的业务错误类型。
//
// 通过携带错误码（Code）与可读消息（Message），使 HTTP 层能够将业务错误
// 映射为合适的状态码与响应体，避免在各处散落地拼装错误。
package apperr

import "fmt"

// Error 是平台统一的业务错误结构。
//
// 字段：
//   - Code:    业务错误码（见下方常量），用于程序判断与 HTTP 状态映射。
//   - Message: 面向调用方的可读错误描述。
//   - cause:   底层原始错误（可选），用于日志排查，不直接暴露给前端。
type Error struct {
	Code    string
	Message string
	cause   error
}

// 预定义的业务错误码。
const (
	CodeInvalidArgument = "INVALID_ARGUMENT" // 参数校验失败
	CodeUnauthorized    = "UNAUTHORIZED"     // 未认证或凭证无效
	CodeForbidden       = "FORBIDDEN"        // 已认证但无权限
	CodeNotFound        = "NOT_FOUND"        // 资源不存在
	CodeConflict        = "CONFLICT"         // 资源冲突（如唯一约束、状态非法转移）
	CodeExhausted       = "RESOURCE_EXHAUSTED"
	CodeInternal        = "INTERNAL" // 服务器内部错误
)

// Error 实现 error 接口，返回 "code: message" 形式的字符串。
func (e *Error) Error() string {
	if e.cause != nil {
		return fmt.Sprintf("%s: %s (%v)", e.Code, e.Message, e.cause)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// Unwrap 支持 errors.Is / errors.As 向下解包底层错误。
func (e *Error) Unwrap() error { return e.cause }

// New 创建一个业务错误。
//
// 参数：
//   - code:    业务错误码常量。
//   - message: 可读错误描述。
func New(code, message string) *Error {
	return &Error{Code: code, Message: message}
}

// Wrap 在业务错误上附加底层原始错误，便于日志追踪。
//
// 参数：
//   - code:    业务错误码常量。
//   - message: 可读错误描述。
//   - cause:   底层原始错误。
func Wrap(code, message string, cause error) *Error {
	return &Error{Code: code, Message: message, cause: cause}
}

// 下列快捷构造函数用于常见场景，减少样板代码。

func InvalidArgument(msg string) *Error { return New(CodeInvalidArgument, msg) }
func Unauthorized(msg string) *Error    { return New(CodeUnauthorized, msg) }
func Forbidden(msg string) *Error       { return New(CodeForbidden, msg) }
func NotFound(msg string) *Error        { return New(CodeNotFound, msg) }
func Conflict(msg string) *Error        { return New(CodeConflict, msg) }
func Exhausted(msg string) *Error       { return New(CodeExhausted, msg) }
func Internal(msg string) *Error        { return New(CodeInternal, msg) }
