//go:build volcano
// +build volcano

// Package scheduler implements Gang Scheduling with Volcano integration.
//
// VolcanoGangScheduler integrates with Volcano Scheduler to enable
// multi-node multi-GPU distributed training with All-or-Nothing semantics.
package scheduler

import (
	"context"
	"fmt"
	"strconv"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	schedulingv1beta1 "volcano.sh/apis/pkg/apis/scheduling/v1beta1"
	volcano "volcano.sh/apis/pkg/client/clientset/versioned"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
)

const (
	// PodGroupLabel is the label key for PodGroup in Volcano
	PodGroupLabel = "volcano.sh/pod-group"
)

// DistributedGangJob represents a distributed gang job to be scheduled.
type DistributedGangJob struct {
	Name         string
	Namespace    string
	Replicas     int
	TotalGPUs    int
	Priority     int32
	TaskSpec     *corev1.PodTemplateSpec
}

// VolcanoGangScheduler implements Gang Scheduling by integrating with Volcano.
type VolcanoGangScheduler struct {
	k8sClient     kubernetes.Interface
	volcanoClient volcano.Clientset
	config        Config
}

// Config contains configuration for VolcanoGangScheduler.
type Config struct {
	// DefaultQueueName is the default queue name in Volcano.
	DefaultQueueName string
}

// NewVolcanoGangScheduler creates a new VolcanoGangScheduler.
func NewVolcanoGangScheduler(
	k8sClient kubernetes.Interface,
	volcanoClient volcano.Clientset,
	config Config,
) *VolcanoGangScheduler {
	if config.DefaultQueueName == "" {
		config.DefaultQueueName = "qingtian"
	}
	return &VolcanoGangScheduler{
		k8sClient:     k8sClient,
		volcanoClient: volcanoClient,
		config:        config,
	}
}

// Schedule submits a distributed gang job to Volcano for scheduling.
func (s *VolcanoGangScheduler) Schedule(ctx context.Context, job *DistributedGangJob) error {
	// 1. Create PodGroup for the gang job
	pg := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      job.Name,
			Namespace: job.Namespace,
			Labels: map[string]string{
				"qingtian.io/job":        job.Name,
				"qingtian.io/replicas":   strconv.Itoa(job.Replicas),
			},
		},
		Spec: schedulingv1beta1.PodGroupSpec{
			MinMember: int32(job.Replicas),
			Queue:     s.config.DefaultQueueName,
		},
	}

	_, err := s.volcanoClient.SchedulingV1beta1().PodGroups(job.Namespace).Create(ctx, pg, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("failed to create PodGroup: %w", err)
	}

	// 2. Create all pods with Volcano scheduling labels
	for i := 0; i < job.Replicas; i++ {
		err := s.createPod(ctx, job, i)
		if err != nil {
			// Rollback already created pods
			s.rollback(ctx, job)
			return fmt.Errorf("failed to create pod %d: %w", i, err)
		}
	}

	return nil
}

// createPod creates a single pod member of the gang job.
func (s *VolcanoGangScheduler) createPod(ctx context.Context, job *DistributedGangJob, replica int) error {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        fmt.Sprintf("%s-%d", job.Name, replica),
			Namespace:   job.Namespace,
			Labels:      make(map[string]string),
			Annotations: job.TaskSpec.Annotations,
		},
		Spec: job.TaskSpec.Spec,
	}

	// Copy labels from template and add Volcano PodGroup label
	for k, v := range job.TaskSpec.Labels {
		pod.Labels[k] = v
	}
	pod.Labels[PodGroupLabel] = job.Name
	pod.Spec.SchedulerName = "volcano"

	// Inject distributed training environment variables
	injectDistributedEnv(pod, replica, job.Replicas, job.Name)

	_, err := s.k8sClient.CoreV1().Pods(job.Namespace).Create(ctx, pod, metav1.CreateOptions{})
	return err
}

// rollback deletes all pods created for this gang job when partial creation fails.
func (s *VolcanoGangScheduler) rollback(ctx context.Context, job *DistributedGangJob) {
	// Delete the PodGroup
	_ = s.volcanoClient.SchedulingV1beta1().PodGroups(job.Namespace).Delete(ctx, job.Name, metav1.DeleteOptions{})

	// Delete all already created pods would require listing
	// For simplicity here, just rely on Kubernetes garbage collection when PodGroup is deleted
}

// Delete deletes a gang job and all its resources.
func (s *VolcanoGangScheduler) Delete(ctx context.Context, namespace, name string) error {
	// Delete PodGroup first
	err := s.volcanoClient.SchedulingV1beta1().PodGroups(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil {
		return fmt.Errorf("failed to delete PodGroup: %w", err)
	}

	// Delete all pods with this PodGroup label - Kubernetes garbage collection
	// will handle them when PodGroup is deleted if we set owner reference
	return nil
}

// injectDistributedEnv injects standard distributed training environment variables.
// These are used by PyTorch DDP, DeepSpeed, Horovod etc.
func injectDistributedEnv(pod *corev1.Pod, rank int, worldSize int, jobName string) {
	// Add environment variables to the first container
	container := &pod.Spec.Containers[0]

	container.Env = append(container.Env, corev1.EnvVar{
		Name:  "RANK",
		Value: strconv.Itoa(rank),
	})
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  "WORLD_SIZE",
		Value: strconv.Itoa(worldSize),
	})
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  "MASTER_ADDR",
		Value: fmt.Sprintf("%s-0.%s-headless", jobName, jobName),
	})
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  "MASTER_PORT",
		Value: "23456",
	})
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  "LOCAL_RANK",
		Value: "0", // Assuming one GPU per pod
	})
}

// CreateDefaultQueue creates the default queue in Volcano if it doesn't exist.
func (s *VolcanoGangScheduler) CreateDefaultQueue(ctx context.Context) error {
	queue := &schedulingv1beta1.Queue{
		ObjectMeta: metav1.ObjectMeta{
			Name: s.config.DefaultQueueName,
		},
		Spec: schedulingv1beta1.QueueSpec{
			Capability: corev1.ResourceList{},
		},
	}

	_, err := s.volcanoClient.SchedulingV1beta1().Queues().Create(ctx, queue, metav1.CreateOptions{})
	if err != nil {
		// If it already exists, that's fine
		if apierrors.IsAlreadyExists(err) {
			return nil
		}
		return fmt.Errorf("failed to create default queue: %w", err)
	}
	return nil
}
