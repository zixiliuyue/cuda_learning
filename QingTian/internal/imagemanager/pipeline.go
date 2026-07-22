package imagemanager

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Pipeline represents a CI/CD pipeline.
type Pipeline struct {
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Steps       []PipelineStep  `json:"steps"`
	Trigger     TriggerConfig   `json:"trigger"`
}

// PipelineStep represents a single step in the pipeline.
type PipelineStep struct {
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	Type     string            `json:"type"` // build/train/evaluate/deploy
	Config   map[string]string `json:"config"`
	DependsOn []string         `json:"dependsOn"`
}

// TriggerConfig defines how the pipeline is triggered.
type TriggerConfig struct {
	Type       string   `json:"type"` // webhook/schedule/manual
	Branches   []string `json:"branches"`
	Schedule   string   `json:"schedule"` // cron expression
}

// PipelineRun represents an execution of a pipeline.
type PipelineRun struct {
	ID             string          `json:"id"`
	PipelineID     string          `json:"pipelineId"`
	Status         string          `json:"status"` // pending/running/success/failed
	StartTime      metav1.Time     `json:"startTime"`
	CompletionTime *metav1.Time   `json:"completionTime,omitempty"`
	Steps          []PipelineStepRun `json:"steps"`
}

// PipelineStepRun represents the execution of a single step.
type PipelineStepRun struct {
	ID             string      `json:"id"`
	Name           string      `json:"name"`
	Status         string      `json:"status"` // pending/running/success/failed
	StartTime      metav1.Time `json:"startTime"`
	CompletionTime *metav1.Time `json:"completionTime,omitempty"`
	LogsURL        string      `json:"logsUrl,omitempty"`
}
