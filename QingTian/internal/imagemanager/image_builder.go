package imagemanager

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// BuildRequest represents a request to build an image.
type BuildRequest struct {
	GitURL     string `json:"gitUrl"`
	GitRef     string `json:"gitRef"`
	Dockerfile string `json:"dockerfile"`
	Context    string `json:"context"`
	ImageName  string `json:"imageName"`
	Tag        string `json:"tag"`
	Namespace  string `json:"namespace"`
}

// BuildInfo contains information about an ongoing build.
type BuildInfo struct {
	BuildID string `json:"buildId"`
	Status  string `json:"status"` // building, completed, failed
}

// ImageBuilder builds container images using Kaniko in Kubernetes.
type ImageBuilder struct {
	k8sClient      kubernetes.Interface
	harborClient   *HarborClient
	buildNamespace string
	config         BuilderConfig
}

// BuilderConfig contains configuration for the image builder.
type BuilderConfig struct {
	// KanikoImage is the Kaniko executor image to use.
	KanikoImage string
	// DockerConfigSecret is the name of the secret containing docker config.
	DockerConfigSecret string
	// Cache enabled/disabled
	CacheEnabled bool
}

// NewImageBuilder creates a new ImageBuilder.
func NewImageBuilder(
	k8sClient kubernetes.Interface,
	harborClient *HarborClient,
	buildNamespace string,
	config BuilderConfig,
) *ImageBuilder {
	if config.KanikoImage == "" {
		config.KanikoImage = "gcr.io/kaniko-project/executor:latest"
	}
	return &ImageBuilder{
		k8sClient:      k8sClient,
		harborClient:   harborClient,
		buildNamespace: buildNamespace,
		config:         config,
	}
}

// StartBuild starts a new image build using Kaniko.
func (b *ImageBuilder) StartBuild(ctx context.Context, req *BuildRequest) (*BuildInfo, error) {
	// Create Kaniko build pod
	pod := b.createKanikoPod(req)
	pod, err := b.k8sClient.CoreV1().Pods(req.Namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create build pod: %w", err)
	}

	// Start a goroutine to wait for completion and trigger scan
	go b.waitBuildCompletion(ctx, req, pod.Name)

	return &BuildInfo{
		BuildID: pod.Name,
		Status:  "building",
	}, nil
}

// waitBuildCompletion waits for the build to complete and triggers vulnerability scan.
func (b *ImageBuilder) waitBuildCompletion(ctx context.Context, req *BuildRequest, podName string) {
	// In production, you'd watch pod status changes using informers
	// For this implementation, after build completes, call afterBuild
	// TODO: Implement proper waiting mechanism with exponential backoff
}

// afterBuild is called after build completes successfully to trigger scanning.
func (b *ImageBuilder) afterBuild(ctx context.Context, req *BuildRequest) error {
	// After build completes successfully, trigger vulnerability scan
	// Parse project and repo from image name
	imageParts := strings.Split(req.ImageName, "/")
	project := ""
	repo := ""
	if len(imageParts) >= 2 {
		project = imageParts[len(imageParts)-2]
		repo = imageParts[len(imageParts)-1]
	} else {
		project = "default"
		repo = imageParts[0]
	}

	if b.harborClient != nil {
		overview, err := b.harborClient.ScanImage(ctx, project, repo, req.Tag)
		if err != nil {
			return err
		}

		// Block if there are critical vulnerabilities
		if overview.HasCriticalVulnerabilities() {
			return fmt.Errorf("image has %d critical vulnerabilities, blocking deployment", overview.Overview.Summary.Critical)
		}
	}

	// Trigger next pipeline step if any
	return b.triggerNextStep(req)
}

// createKanikoPod creates the Kaniko build pod specification.
func (b *ImageBuilder) createKanikoPod(req *BuildRequest) *corev1.Pod {
	fullImage := fmt.Sprintf("%s:%s", req.ImageName, req.Tag)

	args := []string{
		fmt.Sprintf("--dockerfile=%s", req.Dockerfile),
		fmt.Sprintf("--context=%s", b.getContextURL(req.GitURL, req.GitRef)),
		fmt.Sprintf("--destination=%s", fullImage),
	}

	if b.config.CacheEnabled {
		args = append(args, "--cache=true")
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "qingtian-builder-",
			Labels: map[string]string{
				"qingtian.io/build": "image",
				"qingtian.io/image": req.ImageName,
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{
				{
					Name:  "kaniko",
					Image: b.config.KanikoImage,
					Args:  args,
					VolumeMounts: []corev1.VolumeMount{
						{
							Name:      "docker-config",
							MountPath: "/kaniko/.docker",
						},
					},
				},
			},
			Volumes: []corev1.Volume{
				{
					Name: "docker-config",
					VolumeSource: corev1.VolumeSource{
						Secret: &corev1.SecretVolumeSource{
							SecretName: b.config.DockerConfigSecret,
						},
					},
				},
			},
		},
	}

	return pod
}

// getContextURL constructs the Git context URL for Kaniko.
func (b *ImageBuilder) getContextURL(gitURL, gitRef string) string {
	if gitRef == "" {
		return gitURL
	}
	// Kaniko supports git URL#ref format
	return fmt.Sprintf("%s#%s", gitURL, gitRef)
}

// triggerNextStep triggers the next step in the pipeline.
func (b *ImageBuilder) triggerNextStep(req *BuildRequest) error {
	// TODO: Implement pipeline trigger to next step (e.g. training)
	return nil
}

// GetBuildLog gets the logs from an ongoing build.
func (b *ImageBuilder) GetBuildLog(ctx context.Context, namespace, podName string) (string, error) {
	// Get pod logs from Kubernetes
	req := b.k8sClient.CoreV1().Pods(namespace).GetLogs(podName, &corev1.PodLogOptions{
		Container: "kaniko",
		Follow:    false,
	})

	logs, err := req.Do(ctx).Raw()
	if err != nil {
		return "", fmt.Errorf("failed to get build logs: %w", err)
	}

	return string(logs), nil
}
