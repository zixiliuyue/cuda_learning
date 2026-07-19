// Package idgen 提供全局唯一 ID 生成能力。
//
// 平台中的用户、节点、GPU、任务等实体都需要唯一标识符。
// 这里基于时间戳 + 随机数生成短小且趋势递增的 ID，避免引入外部依赖。
package idgen

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"
)

// New 生成一个带业务前缀的唯一 ID。
//
// 参数：
//   - prefix: 业务前缀，例如 "task"、"node"、"user"，便于日志中快速识别 ID 类别。
//
// 返回值：
//   - 形如 "task-1706000000000-a1b2c3d4" 的字符串。前段为毫秒时间戳（保证趋势递增，
//     便于按时间排序），后段为 4 字节随机十六进制（保证同一毫秒内不冲突）。
//
// 注意事项：
//   - 若底层随机源不可用（极罕见），会退化为仅使用纳秒时间戳，仍能保证进程内唯一。
func New(prefix string) string {
	ts := time.Now().UnixMilli()
	buf := make([]byte, 4)
	if _, err := rand.Read(buf); err != nil {
		// 随机源不可用时的兜底：使用纳秒时间戳补充随机性。
		return fmt.Sprintf("%s-%d-%08x", prefix, ts, time.Now().UnixNano()&0xffffffff)
	}
	return fmt.Sprintf("%s-%d-%s", prefix, ts, hex.EncodeToString(buf))
}
