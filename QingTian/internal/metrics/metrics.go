// Package metrics implements Prometheus metrics for QingTian.
//
// This package defines all metrics exposed by QingTian:
//  - Infrastructure layer: GPU utilization, memory, temperature, power
//  - Scheduler layer: queue length, scheduling latency, wait time
//  - Task layer: duration, GPU hours, failure rate, retries
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Infrastructure layer - GPU metrics
var (
	// NodeGPUUtilizationPercent GPU utilization percentage per GPU
	NodeGPUUtilizationPercent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "qingtian_node_gpu_utilization_percent",
			Help: "GPU utilization percentage on node",
		},
		[]string{"node", "gpu_uuid", "gpu_index", "model"},
	)

	// NodeGPUMemoryUsedBytes GPU memory used in bytes
	NodeGPUMemoryUsedBytes = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "qingtian_node_gpu_memory_used_bytes",
			Help: "GPU memory used in bytes",
		},
		[]string{"node", "gpu_uuid", "gpu_index"},
	)

	// NodeGPUTemperatureCelsius GPU temperature in Celsius
	NodeGPUTemperatureCelsius = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "qingtian_node_gpu_temperature_celsius",
			Help: "GPU temperature in Celsius",
		},
		[]string{"node", "gpu_uuid", "gpu_index"},
	)

	// NodeGPUPowerDrawWatts GPU power draw in watts
	NodeGPUPowerDrawWatts = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "qingtian_node_gpu_power_draw_watts",
			Help: "GPU power draw in watts",
		},
		[]string{"node", "gpu_uuid", "gpu_index"},
	)
)

// Scheduler layer metrics
var (
	// SchedulerQueueLength Current number of tasks waiting in scheduler queue
	SchedulerQueueLength = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "qingtian_scheduler_queue_length",
			Help: "Current number of tasks waiting in scheduler queue",
		},
	)

	// SchedulingLatencySeconds Time from task submission to scheduling, in seconds
	SchedulingLatencySeconds = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "qingtian_scheduling_latency_seconds",
			Help:    "Time from task submission to scheduling",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 10),
		},
	)

	// SchedulerAverageWaitTimeSeconds Average wait time in queue, in seconds
	SchedulerAverageWaitTimeSeconds = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "qingtian_scheduler_average_wait_time_seconds",
			Help: "Average wait time of pending tasks in queue",
		},
	)

	// SchedulingSuccessTotal Total number of successful scheduling operations
	SchedulingSuccessTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "qingtian_scheduling_success_total",
			Help: "Total number of successful scheduling operations",
		},
	)

	// SchedulingFailedTotal Total number of failed scheduling operations
	SchedulingFailedTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "qingtian_scheduling_failed_total",
			Help: "Total number of failed scheduling operations",
		},
	)
)

// Task layer metrics
var (
	// TaskExecutionDurationSeconds Task execution duration distribution
	TaskExecutionDurationSeconds = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "qingtian_task_execution_duration_seconds",
			Help:    "Task execution duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"status", "gpu_count"},
	)

	// TaskGPUHoursTotal Total GPU hours consumed by tasks
	TaskGPUHoursTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "qingtian_task_gpu_hours_total",
			Help: "Total GPU hours consumed by tasks",
		},
		[]string{"tenant", "project"},
	)

	// TaskFailuresTotal Total number of failed tasks by failure reason
	TaskFailuresTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "qingtian_task_failures_total",
			Help: "Total number of failed tasks by failure reason",
		},
		[]string{"reason"},
	)

	// TaskRetryCount Total number of task retries
	TaskRetryCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "qingtian_task_retry_count_total",
			Help: "Total number of task retries",
		},
		[]string{"task_id"},
	)

	// ActiveTasks Number of currently active tasks
	ActiveTasks = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "qingtian_active_tasks",
			Help: "Number of currently active tasks by status",
		},
		[]string{"status"},
	)
)

// Task failure reason constants for consistent labeling
const (
	FailureReasonOOM          = "oom"
	FailureReasonImagePull    = "image_pull"
	FailureReasonNonZeroExit  = "non_zero_exit"
	FailureReasonNodeLost     = "node_lost"
	FailureReasonCanceled     = "canceled"
	FailureReasonPreempted    = "preempted"
	FailureReasonTimeout      = "timeout"
	FailureReasonUnknown      = "unknown"
)
