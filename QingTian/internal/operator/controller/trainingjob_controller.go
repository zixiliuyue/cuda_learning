/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"fmt"
	"time"

	"qingtian/internal/domain"
	qingtianv1 "qingtian/internal/operator/apis/qingtian/v1"

	"github.com/go-resty/resty/v2"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// TrainingJobReconciler reconciles a TrainingJob object
type TrainingJobReconciler struct {
	client.Client
	Scheme           *runtime.Scheme
	QingTianEndpoint string // QingTian API server endpoint
	QingTianToken    string // Authentication token for QingTian API
}

// QingTianClient is a client for QingTian REST API
type QingTianClient struct {
	resty *resty.Client
}

// NewQingTianClient creates a new QingTian API client
func NewQingTianClient(endpoint, token string) *QingTianClient {
	r := resty.New()
	r.SetBaseURL(endpoint)
	if token != "" {
		r.SetAuthToken(token)
	}
	return &QingTianClient{resty: r}
}

// CreateTask creates a new task in QingTian
func (c *QingTianClient) CreateTask(ctx context.Context, task *CreateTaskRequest) (*CreateTaskResponse, error) {
	resp, err := c.resty.R().
		SetContext(ctx).
		SetBody(task).
		SetResult(&CreateTaskResponse{}).
		Post("/api/v1/tasks")
	if err != nil {
		return nil, err
	}
	if resp.IsError() {
		return nil, fmt.Errorf("create task failed: %s", resp.String())
	}
	return resp.Result().(*CreateTaskResponse), nil
}

// GetTask gets a task from QingTian
func (c *QingTianClient) GetTask(ctx context.Context, taskID string) (*GetTaskResponse, error) {
	resp, err := c.resty.R().
		SetContext(ctx).
		SetResult(&GetTaskResponse{}).
		Get(fmt.Sprintf("/api/v1/tasks/%s", taskID))
	if err != nil {
		return nil, err
	}
	if resp.IsError() {
		return nil, fmt.Errorf("get task failed: %s", resp.String())
	}
	return resp.Result().(*GetTaskResponse), nil
}

// DeleteTask deletes a task from QingTian
func (c *QingTianClient) DeleteTask(ctx context.Context, taskID string) error {
	resp, err := c.resty.R().
		SetContext(ctx).
		Delete(fmt.Sprintf("/api/v1/tasks/%s", taskID))
	if err != nil {
		return err
	}
	if resp.IsError() {
		return fmt.Errorf("delete task failed: %s", resp.String())
	}
	return nil
}

// CreateTaskRequest is the request to create a task in QingTian
type CreateTaskRequest struct {
	Name        string                 `json:"name"`
	Image       string                 `json:"image"`
	Command     []string               `json:"command"`
	Args        []string               `json:"args,omitempty"`
	Env         []domain.EnvVar       `json:"env,omitempty"`
	Resource    CreateTaskResource    `json:"resource"`
	Priority    int                    `json:"priority"`
	Preemptible bool                   `json:"preemptible"`
}

// CreateTaskResource defines resource requirements for the task
type CreateTaskResource struct {
	GPUCount  int    `json:"gpu_count"`
	MemoryMB  int64  `json:"memory_mb"`
	GPUModel  string `json:"gpu_model,omitempty"`
	GangSize  int    `json:"gang_size,omitempty"`
}

// CreateTaskResponse is the response from creating a task
type CreateTaskResponse struct {
	ID     string `json:"id"`
	Status string `json:"status"`
}

// GetTaskResponse is the response from getting a task
type GetTaskResponse struct {
	ID              string          `json:"id"`
	Name            string          `json:"name"`
	Status          domain.TaskStatus `json:"status"`
	NodeID          string          `json:"node_id,omitempty"`
	AllocatedGPUs   []string        `json:"allocated_gpus,omitempty"`
	CreatedAt       time.Time       `json:"created_at"`
	StartedAt       *time.Time      `json:"started_at,omitempty"`
	FinishedAt      *time.Time      `json:"finished_at,omitempty"`
	ExitCode        int             `json:"exit_code"`
	Message         string          `json:"message"`
	FailureCategory string          `json:"failure_category,omitempty"`
}

// SetupWithManager sets up the controller with the Manager.
func (r *TrainingJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&qingtianv1.TrainingJob{}).
		Complete(r)
}

//+kubebuilder:rbac:groups=qingtian.ai,resources=trainingjobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=qingtian.ai,resources=trainingjobs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=qingtian.ai,resources=trainingjobs/finalizers,verbs=update

// Reconcile is the main reconcile loop for TrainingJob
func (r *TrainingJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)

	// 1. Fetch the TrainingJob instance
	var job qingtianv1.TrainingJob
	if err := r.Get(ctx, req.NamespacedName, &job); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// Create QingTian client
	qtClient := NewQingTianClient(r.QingTianEndpoint, r.QingTianToken)

	// 2. Handle based on current phase
	switch job.Status.Phase {
	case "": // New job, need to create in QingTian
		return r.reconcileNewJob(ctx, &job, qtClient)
	case qingtianv1.TrainingJobPhasePending, qingtianv1.TrainingJobPhaseRunning:
		// Job is active, sync status from QingTian
		return r.reconcileActiveJob(ctx, &job, qtClient)
	case qingtianv1.TrainingJobPhaseSucceeded, qingtianv1.TrainingJobPhaseFailed:
		// Terminal state, no more reconciliation needed
		return ctrl.Result{}, nil
	}

	return ctrl.Result{}, nil
}

// reconcileNewJob creates a new job in QingTian
func (r *TrainingJobReconciler) reconcileNewJob(
	ctx context.Context,
	job *qingtianv1.TrainingJob,
	qtClient *QingTianClient,
) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	_ = log

	// Convert CR to QingTian task request
	req := r.convertToTaskRequest(job)
	resp, err := qtClient.CreateTask(ctx, req)
	if err != nil {
		log.Error(err, "Failed to create task in QingTian")
		job.Status.Phase = qingtianv1.TrainingJobPhaseFailed
		job.Status.Conditions = append(job.Status.Conditions, metav1.Condition{
			Type:    "Created",
			Status:  metav1.ConditionFalse,
			Reason:  "CreationFailed",
			Message: err.Error(),
		})
		if updateErr := r.Status().Update(ctx, job); updateErr != nil {
			log.Error(updateErr, "Failed to update status after creation failure")
		}
		return ctrl.Result{}, err
	}

	// Update status after successful creation
	job.Status.Phase = qingtianv1.TrainingJobPhasePending
	job.Status.TaskID = resp.ID
	job.Status.Replicas = 1
	if job.Spec.Distributed != nil {
		job.Status.Replicas = job.Spec.Distributed.Replicas
	}
	job.Status.ReadyReplicas = 0
	job.Status.StartTime = &metav1.Time{Time: time.Now()}
	job.Status.Conditions = append(job.Status.Conditions, metav1.Condition{
		Type:    "Created",
		Status:  metav1.ConditionTrue,
		Reason:  "CreatedSuccess",
		Message: fmt.Sprintf("Task created in QingTian with ID: %s", resp.ID),
	})

	if err := r.Status().Update(ctx, job); err != nil {
		log.Error(err, "Failed to update status after creation")
		return ctrl.Result{}, err
	}

	log.Info("Created TrainingJob in QingTian", "taskID", resp.ID)

	// Continue reconciling until terminal state
	return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
}

// reconcileActiveJob syncs status from QingTian to CR
func (r *TrainingJobReconciler) reconcileActiveJob(
	ctx context.Context,
	job *qingtianv1.TrainingJob,
	qtClient *QingTianClient,
) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	if job.Status.TaskID == "" {
		log.Error(nil, "TaskID is empty in active job")
		return ctrl.Result{}, nil
	}

	// Get current status from QingTian
	qtTask, err := qtClient.GetTask(ctx, job.Status.TaskID)
	if err != nil {
		if errors.IsNotFound(err) {
			// Task not found in QingTian, mark as failed
			job.Status.Phase = qingtianv1.TrainingJobPhaseFailed
			job.Status.Conditions = append(job.Status.Conditions, metav1.Condition{
				Type:    "Synced",
				Status:  metav1.ConditionFalse,
				Reason:  "TaskNotFound",
				Message: "Task not found in QingTian",
			})
			if updateErr := r.Status().Update(ctx, job); updateErr != nil {
				log.Error(updateErr, "Failed to update status")
			}
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Sync status from QingTian to CR
	r.syncStatus(job, qtTask)

	if err := r.Status().Update(ctx, job); err != nil {
		log.Error(err, "Failed to update status after sync")
		return ctrl.Result{}, err
	}

	// If not terminal, continue checking
	if !isTerminal(job.Status.Phase) {
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	}

	log.Info("TrainingJob reached terminal state", "phase", job.Status.Phase)
	return ctrl.Result{}, nil
}

// convertToTaskRequest converts a TrainingJob CR to QingTian API request
func (r *TrainingJobReconciler) convertToTaskRequest(job *qingtianv1.TrainingJob) *CreateTaskRequest {
	req := &CreateTaskRequest{
		Name:        job.Name,
		Image:       job.Spec.Image,
		Command:     job.Spec.Command,
		Args:        job.Spec.Args,
		Preemptible: job.Spec.Preemptible,
	}

	// Convert env
	req.Env = make([]domain.EnvVar, len(job.Spec.Env))
	for i, env := range job.Spec.Env {
		req.Env[i] = domain.EnvVar{
			Name:  env.Name,
			Value: env.Value,
		}
	}

	// Priority mapping: high -> 100, medium -> 50, low -> 10
	priority := 50
	switch job.Spec.Priority {
	case "high":
		priority = 100
	case "low":
		priority = 10
	}
	req.Priority = priority

	// Resource
	req.Resource = CreateTaskResource{
		GPUCount: job.Spec.GPUCount,
		GPUModel: job.Spec.GPUModel,
	}
	// Parse GPU memory from string like "16Gi" to MB
	if job.Spec.GPUMemory != "" {
		// Simplified parsing - in real code this would handle Gi/Mi properly
		memMB := parseMemoryToMB(job.Spec.GPUMemory)
		req.Resource.MemoryMB = memMB
	}
	// Gang size from distributed replicas
	if job.Spec.Distributed != nil {
		req.Resource.GangSize = job.Spec.Distributed.Replicas
	}

	return req
}

// syncStatus syncs status from QingTian task to TrainingJob CR
func (r *TrainingJobReconciler) syncStatus(job *qingtianv1.TrainingJob, qtTask *GetTaskResponse) {
	// Convert QingTian task status to TrainingJob phase
	job.Status.Phase = convertStatus(qtTask.Status)
	job.Status.ReadyReplicas = 0
	if job.Status.Phase == qingtianv1.TrainingJobPhaseRunning {
		job.Status.ReadyReplicas = 1
	}

	// Update timestamps
	if qtTask.StartedAt != nil {
		job.Status.StartTime = &metav1.Time{Time: *qtTask.StartedAt}
	}
	if qtTask.FinishedAt != nil {
		job.Status.CompletionTime = &metav1.Time{Time: *qtTask.FinishedAt}
	}

	// Add condition for completion
	if isTerminal(job.Status.Phase) {
		condition := metav1.Condition{
			Type:    "Completed",
			Status:  metav1.ConditionTrue,
			Reason:  string(job.Status.Phase),
			Message: qtTask.Message,
		}
		// Remove any existing completed condition
		var newConditions []metav1.Condition
		for _, c := range job.Status.Conditions {
			if c.Type != "Completed" {
				newConditions = append(newConditions, c)
			}
		}
		job.Status.Conditions = append(newConditions, condition)
	}
}

// isTerminal checks if the phase is terminal
func isTerminal(phase qingtianv1.TrainingJobPhase) bool {
	return phase == qingtianv1.TrainingJobPhaseSucceeded || phase == qingtianv1.TrainingJobPhaseFailed
}

// convertStatus converts domain.TaskStatus to TrainingJobPhase
func convertStatus(status domain.TaskStatus) qingtianv1.TrainingJobPhase {
	switch status {
	case domain.TaskPending:
		return qingtianv1.TrainingJobPhasePending
	case domain.TaskScheduled, domain.TaskRunning:
		return qingtianv1.TrainingJobPhaseRunning
	case domain.TaskSucceeded:
		return qingtianv1.TrainingJobPhaseSucceeded
	case domain.TaskFailed, domain.TaskCanceled:
		return qingtianv1.TrainingJobPhaseFailed
	default:
		return qingtianv1.TrainingJobPhasePending
	}
}

// parseMemoryToMem parses memory string like "16Gi" to MB
func parseMemoryToMB(mem string) int64 {
	// This is a simplified parser - for production use proper unit parsing
	// Example: "16Gi" -> 16 * 1024 = 16384 MB
	var value int64
	var unit string
	fmt.Sscanf(mem, "%d%s", &value, &unit)
	switch unit {
	case "Gi", "G":
		return value * 1024
	case "Mi", "M":
		return value
	default:
		return value * 1024 // default to GB
	}
}
