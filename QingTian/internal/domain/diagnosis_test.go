package domain

import "testing"

// TestDiagnoseFailure 验证失败诊断能根据退出码与输出正确归类。
func TestDiagnoseFailure(t *testing.T) {
	cases := []struct {
		name     string
		exitCode int
		output   string
		want     FailureCategory
	}{
		{"退出码0为无失败", 0, "任何输出", FailureNone},
		{"CUDA显存不足识别为OOM", 1, "RuntimeError: CUDA out of memory. Tried to allocate", FailureOOM},
		{"通用OOM识别", 137, "Killed process, OOM", FailureOOM},
		{"镜像拉取失败识别", 1, "Error response from daemon: manifest unknown", FailureImagePull},
		{"命令未找到识别为启动失败", 127, "bash: python: command not found", FailureStartFailure},
		{"无特征非零退出归类为NonZeroExit", 2, "Traceback: ValueError", FailureNonZeroExit},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := DiagnoseFailure(c.exitCode, c.output)
			if got.Category != c.want {
				t.Errorf("DiagnoseFailure(%d) 归类=%s, 期望=%s", c.exitCode, got.Category, c.want)
			}
			if c.want != FailureNone && got.Suggestion == "" {
				t.Errorf("失败诊断应给出可操作建议")
			}
		})
	}
}

// TestFeatureEnabled 验证版本功能门控。
func TestFeatureEnabled(t *testing.T) {
	// 企业版开放全部功能。
	if !FeatureEnabled(EditionEnterprise, FeatureGPUSharing) {
		t.Error("企业版应开放 GPU 共享")
	}
	if !FeatureEnabled(EditionEnterprise, FeatureMultiTenancy) {
		t.Error("企业版应开放多租户")
	}
	// 社区版仅开放引流功能。
	if FeatureEnabled(EditionCommunity, FeatureGPUSharing) {
		t.Error("社区版不应开放 GPU 共享")
	}
	if !FeatureEnabled(EditionCommunity, FeatureMetering) {
		t.Error("社区版应开放计量看板（引流）")
	}
}

// TestResourceRequestEffectiveShareMode 验证共享模式归一。
func TestResourceRequestEffectiveShareMode(t *testing.T) {
	if got := (ResourceRequest{}).EffectiveShareMode(); got != ShareExclusive {
		t.Errorf("空共享模式应归一为 exclusive，实际 %s", got)
	}
	if got := (ResourceRequest{ShareMode: ShareMIG}).EffectiveShareMode(); got != ShareMIG {
		t.Errorf("显式 MIG 应保持，实际 %s", got)
	}
}

// TestProjectHasMember 验证项目成员判断。
func TestProjectHasMember(t *testing.T) {
	p := &Project{Members: []string{"u1", "u2"}}
	if !p.HasMember("u1") {
		t.Error("u1 应为成员")
	}
	if p.HasMember("u3") {
		t.Error("u3 不应为成员")
	}
}

// TestQuotaRemaining 验证配额剩余计算。
func TestQuotaRemaining(t *testing.T) {
	q := &Quota{MaxGPUHours: 100, UsedGPUHours: 30}
	if got := q.RemainingGPUHours(); got != 70 {
		t.Errorf("剩余卡时应为 70，实际 %v", got)
	}
	unlimited := &Quota{MaxGPUHours: 0, UsedGPUHours: 1000}
	if unlimited.RemainingGPUHours() <= 0 {
		t.Error("上限为 0（不限）时剩余应为极大值")
	}
}
