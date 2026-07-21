package controller

import (
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	qingtianv1 "qingtian/internal/operator/apis/qingtian/v1"
)

// ensureApiserver ensures the apiserver Deployment exists
func (r *QingTianClusterReconciler) ensureApiserver(
	ctx context.Context,
	cluster *qingtianv1.QingTianCluster,
) error {
	name := getApiserverName(cluster)
	var existing appsv1.Deployment
	err := r.Get(ctx, client.ObjectKey{Namespace: cluster.Namespace, Name: name}, &existing)
	if err == nil {
		// Deployment already exists - update if needed
		return nil
	}
	if !errors.IsNotFound(err) {
		return err
	}

	replicas := cluster.Spec.Replicas
	if replicas == 0 {
		replicas = 1
	}

	image := cluster.Spec.Image
	if image == "" {
		image = fmt.Sprintf("qingtian/apiserver:%s", cluster.Spec.Version)
	}

	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cluster.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(cluster, qingtianv1.GroupVersion.WithKind("QingTianCluster")),
			},
			Labels: map[string]string{
				"app":        "qingtian",
				"component":  "apiserver",
				"cluster":    cluster.Name,
				"version":    cluster.Spec.Version,
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app":       "qingtian",
					"component": "apiserver",
					"cluster":   cluster.Name,
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app":       "qingtian",
						"component": "apiserver",
						"cluster":   cluster.Name,
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "apiserver",
							Image: image,
							Args:  []string{"-config", "/config/config.yaml"},
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: 8080,
									Protocol:      corev1.ProtocolTCP,
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "config",
									MountPath: "/config",
									ReadOnly:  true,
								},
							},
							Resources: cluster.Spec.Resources,
							LivenessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path: "/healthz",
										Port:   intstr.FromString("http"),
									},
								},
								InitialDelaySeconds: 10,
								PeriodSeconds:       30,
							},
							ReadinessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path: "/healthz",
										Port:   intstr.FromString("http"),
									},
								},
								InitialDelaySeconds: 5,
								PeriodSeconds:       10,
							},
						},
					},
					},
					Volumes: []corev1.Volume{
						{
							Name: "config",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: getConfigMapName(cluster),
									},
								},
							},
						},
					},
				},
		},
	},
	}

	if err := r.Create(ctx, dep); err != nil {
		return err
	}
	return nil
}

// ensureApiserverService ensures the apiserver Service exists
func (r *QingTianClusterReconciler) ensureApiserverService(
	ctx context.Context,
	cluster *qingtianv1.QingTianCluster,
) error {
	name := getServiceName(cluster)
	var existing corev1.Service
	err := r.Get(ctx, client.ObjectKey{Namespace: cluster.Namespace, Name: name}, &existing)
	if err == nil {
		return nil
	}
	if !errors.IsNotFound(err) {
		return err
	}

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cluster.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(cluster, qingtianv1.GroupVersion.WithKind("QingTianCluster")),
			},
			Labels: map[string]string{
				"app":       "qingtian",
				"component": "apiserver",
				"cluster":   cluster.Name,
			},
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeClusterIP,
			Ports: []corev1.ServicePort{
				{
					Name:       "http",
					Port:       80,
					TargetPort: intstr.FromString("http"),
					Protocol:   corev1.ProtocolTCP,
				},
			},
			Selector: map[string]string{
				"app":       "qingtian",
				"component": "apiserver",
				"cluster":   cluster.Name,
			},
		},
	},
	}

	if err := r.Create(ctx, svc); err != nil {
		return err
	}
	return nil
}

// ensureAgentDaemonSet ensures the agent DaemonSet exists on GPU nodes
func (r *QingTianClusterReconciler) ensureAgentDaemonSet(
	ctx context.Context,
	cluster *qingtianv1.QingTianCluster,
) error {
	name := getAgentName(cluster)
	var existing appsv1.DaemonSet
	err := r.Get(ctx, client.ObjectKey{Namespace: cluster.Namespace, Name: name}, &existing)
	if err == nil {
		return nil
	}
	if !errors.IsNotFound(err) {
		return err
	}

	ds := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cluster.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(cluster, qingtianv1.GroupVersion.WithKind("QingTianCluster")),
			},
			Labels: map[string]string{
				"app":        "qingtian",
				"component":  "agent",
				"cluster":    cluster.Name,
			},
		},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app":       "qingtian",
					"component": "agent",
					"cluster":   cluster.Name,
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app":       "qingtian",
						"component": "agent",
						"cluster":   cluster.Name,
					},
				},
				Spec: corev1.PodSpec{
					NodeSelector: cluster.Spec.NodeSelector,
					Containers: []corev1.Container{
						{
							Name:  "qingtian-agent",
							Image: "qingtian/agent:latest", // TODO: use version from config
							Env: []corev1.EnvVar{
								{
									Name:  "QT_CONTROL_PLANE_URL",
									Value: fmt.Sprintf("http://%s.%s.svc.cluster.local", getServiceName(cluster), cluster.Namespace),
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "dockersock",
									MountPath: "/var/run/docker.sock",
								},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "dockersock",
							VolumeSource: corev1.VolumeSource{
								HostPath: &corev1.HostPathVolumeSource{
									Path: "/var/run/docker.sock",
								},
							},
						},
					},
				},
			},
			},
			UpdateStrategy: appsv1.DaemonSetUpdateStrategy{
				Type: appsv1.RollingUpdateDaemonSetStrategyType,
			},
		},
	},
	}

	if err := r.Create(ctx, ds); err != nil {
		return err
	}
	return nil
}

// ensureMongoDB ensures MongoDB StatefulSet exists when internal is enabled
func (r *QingTianClusterReconciler) ensureMongoDB(
	ctx context.Context,
	cluster *qingtianv1.QingTianCluster,
) error {
	// For simplicity, we deploy a single-instance MongoDB for getting started
	// Production users should use external MongoDB or managed service
	name := getMongoDBName(cluster)
	var existing appsv1.StatefulSet
	err := r.Get(ctx, client.ObjectKey{Namespace: cluster.Namespace, Name: name}, &existing)
	if err == nil {
		return nil
	}
	if !errors.IsNotFound(err) {
		return err
	}

	replicas := int32(1)
	ss := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cluster.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(cluster, qingtianv1.GroupVersion.WithKind("QingTianCluster")),
			},
		},
		Spec: appsv1.StatefulSetSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app":     "mongodb",
					"cluster": cluster.Name,
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app":     "mongodb",
						"cluster": cluster.Name,
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:    "mongodb",
							Image:   "mongo:5",
							Command: []string{"mongod", "--bind_ip_all"},
							Ports: []corev1.ContainerPort{
								{
									Name:          "mongodb",
									ContainerPort: 27017,
									Protocol:      corev1.ProtocolTCP,
								},
							},
						},
					},
				},
			},
		},
	},
	}

	if err := r.Create(ctx, ss); err != nil {
		return err
	}

	// Create Service for MongoDB
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cluster.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(cluster, qingtianv1.GroupVersion.WithKind("QingTianCluster")),
			},
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				"app":     "mongodb",
				"cluster": cluster.Name,
			},
			Ports: []corev1.ServicePort{
				{
					Port:     27017,
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
	},
	}
	if err := r.Create(ctx, svc); err != nil {
		return err
	}

	return nil
}

// ensureRedis ensures Redis Deployment exists when internal is enabled
func (r *QingTianClusterReconciler) ensureRedis(
	ctx context.Context,
	cluster *qingtianv1.QingTianCluster,
) error {
	name := getRedisName(cluster)
	var existing appsv1.Deployment
	err := r.Get(ctx, client.ObjectKey{Namespace: cluster.Namespace, Name: name}, &existing)
	if err == nil {
		return nil
	}
	if !errors.IsNotFound(err) {
		return err
	}

	replicas := int32(1)
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cluster.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(cluster, qingtianv1.GroupVersion.WithKind("QingTianCluster")),
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app":     "redis",
					"cluster": cluster.Name,
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app":     "redis",
						"cluster": cluster.Name,
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "redis",
							Image: "redis:7-alpine",
							Ports: []corev1.ContainerPort{
								{
									Name:     "redis",
									ContainerPort: 6379,
									Protocol:      corev1.ProtocolTCP,
								},
							},
						},
					},
				},
			},
		},
	},
	}

	if err := r.Create(ctx, dep); err != nil {
		return err
	}

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cluster.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(cluster, qingtianv1.GroupVersion.WithKind("QingTianCluster")),
			},
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				"app":     "redis",
				"cluster": cluster.Name,
			},
			Ports: []corev1.ServicePort{
				{
					Port:     6379,
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
	},
	}
	if err := r.Create(ctx, svc); err != nil {
		return err
	}

	return nil
}
