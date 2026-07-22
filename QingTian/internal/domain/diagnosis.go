// 本文件实现任务失败的结构化诊断与功能版本门控等纯领域规则。
//
// 将"失败原因归类"与"企业版功能开关"收敛到领域层，可让 API/前端
// 直接复用一致的判定逻辑，避免规则散落导致口径不一。
package domain

import "strings"

// Diagnosis 表示一次失败诊断结果。
//
// 字段：
//   - Category:   结构化失败归类。
//   - Summary:    面向用户的一句话原因。
//   - Suggestion: 可操作的排查/修复建议。
type Diagnosis struct {
	Category   FailureCategory `json:"category"`
	Summary    string          `json:"summary"`
	Suggestion string          `json:"suggestion"`
}

// DiagnoseFailure 根据退出码与执行输出推断结构化失败原因。
//
// 该函数不依赖任何外部服务，是可单测的纯逻辑：Agent 回传日志片段后，
// 控制面据此给出"是什么、怎么办"，将黑盒失败转化为可操作提示。
//
// 参数：
//   - exitCode: 进程退出码（0 表示正常）。
//   - output:   进程标准输出/错误的合并文本（可为片段）。
//
// 返回值：
//   - Diagnosis: 诊断结果；exitCode 为 0 时归类为 FailureNone。
func DiagnoseFailure(exitCode int, output string) Diagnosis {
	if exitCode == 0 {
		return Diagnosis{Category: FailureNone, Summary: "任务正常结束"}
	}
	lower := strings.ToLower(output)

	switch {
	case containsAny(lower, "out of memory", "cuda out of memory", "oom", "outofmemoryerror"):
		return Diagnosis{
			Category:   FailureOOM,
			Summary:    "显存不足（OOM），进程被终止",
			Suggestion: "调小 batch size、开启梯度累积或申请更大显存/更多 GPU 后重试",
		}
	case containsAny(lower, "no such image", "pull access denied", "manifest unknown", "image not found", "errimagepull"):
		return Diagnosis{
			Category:   FailureImagePull,
			Summary:    "镜像拉取失败",
			Suggestion: "检查镜像名称/标签是否正确，以及镜像仓库凭证与网络连通性",
		}
	case containsAny(lower, "command not found", "no such file or directory", "executable file not found", "permission denied"):
		return Diagnosis{
			Category:   FailureStartFailure,
			Summary:    "进程无法启动",
			Suggestion: "检查启动命令、可执行文件路径与执行权限是否正确",
		}
	default:
		return Diagnosis{
			Category:   FailureNonZeroExit,
			Summary:    "进程以非零退出码结束",
			Suggestion: "查看实时日志定位报错堆栈，确认代码与依赖是否正常",
		}
	}
}

// containsAny 判断 s 是否包含 keys 中任意一个子串。
func containsAny(s string, keys ...string) bool {
	for _, k := range keys {
		if strings.Contains(s, k) {
			return true
		}
	}
	return false
}
