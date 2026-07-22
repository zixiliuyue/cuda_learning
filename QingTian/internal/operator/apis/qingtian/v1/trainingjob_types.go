// Package v1 defines the API types for QingTian Kubernetes Operator.
//
// TrainingJob is the custom resource for submitting AI training jobs to QingTian.
// Users can use kubectl to create/list/delete training jobs directly.
package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Status",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:printcolumn:name="Replicas",type="integer",JSONPath=".status.replicas"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyReplicas"

// TrainingJob is the Schema for the trainingjobs API
type TrainingJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TrainingJobSpec   `json:"spec,omitempty"`
	Status TrainingJobStatus `json:"status,omitempty"`
}

// DistributedSpec defines distributed training configuration
type DistributedSpec struct {
	Mode     string `json:"mode"`     // DDP, DeepSpeed, Horovod
	Replicas int    `json:"replicas"` // Number of replicas
}

// CheckpointSpec defines checkpoint configuration
type CheckpointSpec struct {
	Enabled  bool   `json:"enabled"`
	SavePath string `json:"savePath,omitempty"`
}

// TrainingJobSpec defines the desired state of TrainingJob
type TrainingJobSpec struct {
	Image         string             `json:"image"`
	Command       []string           `json:"command"`
	Args          []string           `json:"args,omitempty"`
	GPUCount      int                `json:"gpuCount"`
	GPUMemory     string             `json:"gpuMemory,omitempty"`
	GPUModel      string             `json:"gpuModel,omitempty"`
	Distributed   *DistributedSpec   `json:"distributed,omitempty"`
	Checkpoint    *CheckpointSpec    `json:"checkpoint,omitempty"`
	Priority      string             `json:"priority,omitempty"`
	Preemptible   bool               `json:"preemptible,omitempty"`
	Env           []corev1.EnvVar   `json:"env,omitempty"`
}

// TrainingJobPhase defines the phase of the TrainingJob
type TrainingJobPhase string

const (
	TrainingJobPhasePending   TrainingJobPhase = "Pending"
	TrainingJobPhaseRunning   TrainingJobPhase = "Running"
	TrainingJobPhaseSucceeded TrainingJobPhase = "Succeeded"
	TrainingJobPhaseFailed    TrainingJobPhase = "Failed"
)

// TrainingJobStatus defines the observed state of TrainingJob
type TrainingJobStatus struct {
	Phase           TrainingJobPhase `json:"phase,omitempty"`
	TaskID          string           `json:"taskId,omitempty"`
	StartTime       *metav1.Time     `json:"startTime,omitempty"`
	CompletionTime  *metav1.Time     `json:"completionTime,omitempty"`
	Replicas        int              `json:"replicas,omitempty"`
	ReadyReplicas   int              `json:"readyReplicas,omitempty"`
	Conditions      []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true

// TrainingJobList contains a list of TrainingJob
type TrainingJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TrainingJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TrainingJob{}, &TrainingJobList{})
}
