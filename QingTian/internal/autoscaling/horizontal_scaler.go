// Package autoscaling implements elastic auto-scaling based on workload metrics.
//
// HorizontalScaler makes scaling decisions based on queue length and GPU utilization,
// and triggers node group scaling through cloud provider APIs.
package autoscaling

import (
	"context"
	"fmt"
	"time"
)

// CloudProviderClient defines the interface for cloud provider API.
type CloudProviderClient interface {
	// GetCurrentNodeCount returns the current number of worker nodes.
	GetCurrentNodeCount(ctx context.Context) (int, error)
	// ScaleUp increases the number of nodes by delta.
	ScaleUp(ctx context.Context, delta int) error
	// ScaleDown decreases the number of nodes by delta.
	ScaleDown(ctx context.Context, delta int) error
}

// MetricsClient defines the interface for getting QingTian metrics.
type MetricsClient interface {
	// GetSchedulerQueueLength returns the current pending queue length.
	GetSchedulerQueueLength(ctx context.Context) (int, error)
	// GetAverageGPUUtilization returns the average GPU utilization across all nodes.
	GetAverageGPUUtilization(ctx context.Context) (float64, error)
}

// AutoScalingConfig configures the horizontal scaler.
type AutoScalingConfig struct {
	// MinNodes is the minimum number of nodes.
	MinNodes int
	// MaxNodes is the maximum number of nodes.
	MaxNodes int
	// QueueLengthTarget is the target queue length that triggers scaling.
	QueueLengthTarget int
	// UtilizationTarget is the target GPU utilization (percentage).
	UtilizationTarget float64
	// ScaleUpStep is the number of nodes to add on each scale up.
	ScaleUpStep int
	// ScaleDownStep is the number of nodes to remove on each scale down.
	ScaleDownStep int
	// ScaleDownDelayMinutes is the delay after last scale before scaling down.
	ScaleDownDelayMinutes int
}

// ScalingDecision represents a scaling decision.
type ScalingDecision struct {
	Action      string `json:"action"` // scale_up / scale_down / none
	TargetNodes int    `json:"targetNodes"`
	Reason      string `json:"reason"`
}

// HorizontalScaler implements horizontal auto-scaling.
type HorizontalScaler struct {
	cloudClient CloudProviderClient
	metrics     MetricsClient
	config      AutoScalingConfig
	lastScaleDown time.Time
}

// NewHorizontalScaler creates a new HorizontalScaler.
func NewHorizontalScaler(
	cloudClient CloudProviderClient,
	metrics MetricsClient,
	config AutoScalingConfig,
) *HorizontalScaler {
	// Set sensible defaults
	if config.ScaleUpStep <= 0 {
		config.ScaleUpStep = 2
	}
	if config.ScaleDownStep <= 0 {
		config.ScaleDownStep = 1
	}
	if config.ScaleDownDelayMinutes <= 0 {
		config.ScaleDownDelayMinutes = 10
	}
	return &HorizontalScaler{
		cloudClient: cloudClient,
		metrics:     metrics,
		config:      config,
	}
}

// MakeDecision makes a scaling decision based on current metrics.
func (s *HorizontalScaler) MakeDecision(ctx context.Context) (*ScalingDecision, error) {
	// Get current metrics
	queueLen, err := s.metrics.GetSchedulerQueueLength(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get queue length: %w", err)
	}

	utilAvg, err := s.metrics.GetAverageGPUUtilization(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get GPU utilization: %w", err)
	}

	currentNodes, err := s.cloudClient.GetCurrentNodeCount(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get current node count: %w", err)
	}

	// Decision logic
	switch {
	case queueLen > s.config.QueueLengthTarget && currentNodes < s.config.MaxNodes:
		// Need to scale up
		targetNodes := currentNodes + s.config.ScaleUpStep
		if targetNodes > s.config.MaxNodes {
			targetNodes = s.config.MaxNodes
		}
		return &ScalingDecision{
			Action:      "scale_up",
			TargetNodes: targetNodes,
			Reason:      fmt.Sprintf("queue length %d > target %d", queueLen, s.config.QueueLengthTarget),
		}, nil

	case utilAvg < s.config.UtilizationTarget && currentNodes > s.config.MinNodes && s.canScaleDown():
		// Need to scale down
		targetNodes := currentNodes - s.config.ScaleDownStep
		if targetNodes < s.config.MinNodes {
			targetNodes = s.config.MinNodes
		}
		s.lastScaleDown = time.Now()
		return &ScalingDecision{
			Action:      "scale_down",
			TargetNodes: targetNodes,
			Reason:      fmt.Sprintf("utilization %.1f%% < target %.1f%%", utilAvg, s.config.UtilizationTarget),
		}, nil

	default:
		// No action needed
		return &ScalingDecision{
			Action:      "none",
			TargetNodes: currentNodes,
			Reason:      "no action needed",
		}, nil
	}
}

// Execute executes the scaling decision.
func (s *HorizontalScaler) Execute(ctx context.Context, decision *ScalingDecision) error {
	currentNodes, err := s.cloudClient.GetCurrentNodeCount(ctx)
	if err != nil {
		return err
	}

	delta := decision.TargetNodes - currentNodes
	switch {
	case delta > 0:
		return s.cloudClient.ScaleUp(ctx, delta)
	case delta < 0:
		return s.cloudClient.ScaleDown(ctx, -delta)
	default:
		return nil
	}
}

// canScaleDown checks if we can scale down now (cooldown period).
func (s *HorizontalScaler) canScaleDown() bool {
	if s.lastScaleDown.IsZero() {
		return true
	}
	coolDown := time.Duration(s.config.ScaleDownDelayMinutes) * time.Minute
	return time.Since(s.lastScaleDown) > coolDown
}
