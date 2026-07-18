// Package logger 封装平台统一的结构化日志。
//
// 基于标准库 log/slog 实现，提供带级别与 JSON 输出的日志器，
// 便于对接集中式日志系统（如 Loki/ELK）。
package logger

import (
	"log/slog"
	"os"
	"strings"
)

// New 创建一个结构化日志器。
//
// 参数：
//   - level: 日志级别字符串，支持 "debug"、"info"、"warn"、"error"（大小写不敏感）。
//     无法识别时默认为 info。
//   - service: 服务名，会作为固定字段附加到每条日志，便于区分来源。
//
// 返回值：
//   - *slog.Logger: 可直接用于记录日志的实例。
func New(level, service string) *slog.Logger {
	var lvl slog.Level
	switch strings.ToLower(level) {
	case "debug":
		lvl = slog.LevelDebug
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: lvl})
	return slog.New(handler).With("service", service)
}
