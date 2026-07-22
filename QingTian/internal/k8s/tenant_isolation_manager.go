// Package k8s implements Kubernetes integration for multi-tenancy isolation.
//
// TenantIsolationManager manages Kubernetes Namespace, ResourceQuota, and NetworkPolicy
// for QingTian multi-tenancy isolation.
package k8s

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// WorkspaceSpec defines the specification for creating an isolated workspace.
type WorkspaceSpec struct {
	TenantID       string   `json:"tenantId"`
	ProjectID      string   `json:"projectId"`
	GPUQuota       int      `json:"gpuQuota"`       // Number of GPUs allowed
	MemoryQuota    string   `json:"memoryQuota"`    // Memory quota (e.g., "128Gi")
	NetworkIsolate bool     `json:"networkIsolate"` // Enable network isolation between projects
	Whitelist      []string `json:"whitelist"`      // Allowed cross-project Namespaces
}

// TenantIsolationManager manages multi-tenancy isolation on Kubernetes.
type TenantIsolationManager struct {
	k8sClient kubernetes.Interface
}

// NewTenantIsolationManager creates a new TenantIsolationManager.
func NewTenantIsolationManager(k8sClient kubernetes.Interface) *TenantIsolationManager {
	return &TenantIsolationManager{
		k8sClient: k8sClient,
	}
}

// CreateWorkspace creates an isolated workspace for a tenant project.
// It automatically creates:
//  1. Namespace with tenant/project labels
//  2. ResourceQuota for GPU and memory
//  3. NetworkPolicy if isolation is enabled
//  4. RBAC RoleBinding for tenant access
func (m *TenantIsolationManager) CreateWorkspace(ctx context.Context, spec *WorkspaceSpec) (*corev1.Namespace, error) {
	// 1. Create Namespace
	ns := m.buildNamespace(spec)
	ns, err := m.k8sClient.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create Namespace: %w", err)
	}

	// 2. Create ResourceQuota
	rq := m.buildResourceQuota(spec, ns.Name)
	_, err = m.k8sClient.CoreV1().ResourceQuotas(ns.Name).Create(ctx, rq, metav1.CreateOptions{})
	if err != nil {
		// Cleanup on failure
		_ = m.DeleteWorkspace(ctx, ns.Name)
		return nil, fmt.Errorf("failed to create ResourceQuota: %w", err)
	}

	// 3. Create NetworkPolicy if isolation is enabled
	if spec.NetworkIsolate {
		err = m.createNetworkPolicies(ctx, ns.Name, spec)
		if err != nil {
			_ = m.DeleteWorkspace(ctx, ns.Name)
			return nil, fmt.Errorf("failed to create NetworkPolicy: %w", err)
		}
	}

	// 4. Create RBAC RoleBinding
	err = m.createRBAC(ctx, ns.Name, spec.TenantID)
	if err != nil {
		_ = m.DeleteWorkspace(ctx, ns.Name)
		return nil, fmt.Errorf("failed to create RBAC: %w", err)
	}

	return ns, nil
}

// DeleteWorkspace deletes an isolated workspace and all associated resources.
// Kubernetes garbage collection will handle cascading deletion via OwnerReference.
func (m *TenantIsolationManager) DeleteWorkspace(ctx context.Context, namespace string) error {
	deletePolicy := metav1.DeletePropagationForeground
	err := m.k8sClient.CoreV1().Namespaces().Delete(ctx, namespace, metav1.DeleteOptions{
		PropagationPolicy: &deletePolicy,
	})
	if err != nil {
		return fmt.Errorf("failed to delete Namespace: %w", err)
	}
	return nil
}

// buildNamespace builds the Namespace with proper labels.
func (m *TenantIsolationManager) buildNamespace(spec *WorkspaceSpec) *corev1.Namespace {
	return &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("%s-%s-", spec.TenantID, spec.ProjectID),
			Labels: map[string]string{
				"qingtian.io/tenant":   spec.TenantID,
				"qingtian.io/project":  spec.ProjectID,
				"qingtian.io/isolation": "enabled",
			},
		},
	}
}

// buildResourceQuota builds the ResourceQuota for GPU and memory.
func (m *TenantIsolationManager) buildResourceQuota(spec *WorkspaceSpec, namespace string) *corev1.ResourceQuota {
	hard := corev1.ResourceList{
		// Request GPU quota
		corev1.ResourceName("requests.nvidia.com/gpu"): resourceQuantity(spec.GPUQuota),
		corev1.ResourceName("limits.nvidia.com/gpu"):    resourceQuantity(spec.GPUQuota),
	}

	// Parse memory quota if provided
	if spec.MemoryQuota != "" {
		memQuantity, err := parseResourceQuantity(spec.MemoryQuota)
		if err == nil {
			hard[corev1.ResourceMemory] = memQuantity
		}
	}

	return &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "qingtian-resource-quota",
			Namespace: namespace,
		},
		Spec: corev1.ResourceQuotaSpec{
			Hard: hard,
		},
	}
}

// createNetworkPolicies creates NetworkPolicy for network isolation.
// When isolation is enabled:
//  - By default deny all ingress and egress
//  - Allow communication within the same project
//  - Allow communication to whitelisted projects
func (m *TenantIsolationManager) createNetworkPolicies(ctx context.Context, namespace string, spec *WorkspaceSpec) error {
	// 1. Default deny all policy
	defaultDeny := &networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "default-deny-all",
			Namespace: namespace,
		},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{},
			PolicyTypes: []networkingv1.PolicyType{
				networkingv1.PolicyTypeIngress,
				networkingv1.PolicyTypeEgress,
			},
		},
	}
	_, err := m.k8sClient.NetworkingV1().NetworkPolicies(namespace).Create(ctx, defaultDeny, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	// 2. Allow same project communication
	allowSameProject := &networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "allow-same-project",
			Namespace: namespace,
		},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{},
			Ingress: []networkingv1.NetworkPolicyIngressRule{
				{
					From: []networkingv1.NetworkPolicyPeer{
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"qingtian.io/project": spec.ProjectID,
								},
							},
						},
					},
				},
			},
			Egress: []networkingv1.NetworkPolicyEgressRule{
				{
					To: []networkingv1.NetworkPolicyPeer{
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"qingtian.io/project": spec.ProjectID,
								},
							},
						},
					},
				},
			},
			PolicyTypes: []networkingv1.PolicyType{
				networkingv1.PolicyTypeIngress,
				networkingv1.PolicyTypeEgress,
			},
		},
	}
	_, err = m.k8sClient.NetworkingV1().NetworkPolicies(namespace).Create(ctx, allowSameProject, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	// 3. Allow whitelisted cross-project communication
	for _, proj := range spec.Whitelist {
		allowWhitelist := &networkingv1.NetworkPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("allow-whitelist-%s", proj),
				Namespace: namespace,
			},
			Spec: networkingv1.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Ingress: []networkingv1.NetworkPolicyIngressRule{
					{
						From: []networkingv1.NetworkPolicyPeer{
							{
								NamespaceSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{
										"qingtian.io/project": proj,
									},
								},
							},
						},
					},
				},
				Egress: []networkingv1.NetworkPolicyEgressRule{
					{
						To: []networkingv1.NetworkPolicyPeer{
							{
								NamespaceSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{
										"qingtian.io/project": proj,
									},
								},
							},
						},
					},
				},
				PolicyTypes: []networkingv1.PolicyType{
					networkingv1.PolicyTypeIngress,
					networkingv1.PolicyTypeEgress,
				},
			},
		}
		_, err = m.k8sClient.NetworkingV1().NetworkPolicies(namespace).Create(ctx, allowWhitelist, metav1.CreateOptions{})
		if err != nil {
			return err
		}
	}

	// 4. Allow DNS access to Kubernetes DNS
	allowDNS := &networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "allow-dns",
			Namespace: namespace,
		},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{},
			Egress: []networkingv1.NetworkPolicyEgressRule{
				{
					To: []networkingv1.NetworkPolicyPeer{
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"k8s.io/kubernetes-system": "kube-dns",
								},
							},
						},
					},
				},
			},
			PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeEgress},
		},
	}
	_, err = m.k8sClient.NetworkingV1().NetworkPolicies(namespace).Create(ctx, allowDNS, metav1.CreateOptions{})
	// Ignore error if it doesn't match - DNS might be elsewhere
	_ = err

	return nil
}

// createRBAC creates RBAC RoleBinding for tenant access.
func (m *TenantIsolationManager) createRBAC(ctx context.Context, namespace string, tenantID string) error {
	// Create RoleBinding that binds the default edit role to the tenant user group
	rb := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("tenant-%s-access", tenantID),
			Namespace: namespace,
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "Role",
			Name:     "edit",
		},
		Subjects: []rbacv1.Subject{
			{
				Kind: rbacv1.UserKind,
				Name:  tenantID,
			},
		},
	}
	_, err := m.k8sClient.RbacV1().RoleBindings(namespace).Create(ctx, rb, metav1.CreateOptions{})
	return err
}

// UpdateQuota updates the resource quota for an existing workspace.
func (m *TenantIsolationManager) UpdateQuota(ctx context.Context, namespace string, gpuQuota int, memoryQuota string) error {
	rq, err := m.k8sClient.CoreV1().ResourceQuotas(namespace).Get(ctx, "qingtian-resource-quota", metav1.GetOptions{})
	if err != nil {
		return err
	}

	rq.Spec.Hard[corev1.ResourceName("requests.nvidia.com/gpu")] = resourceQuantity(gpuQuota)
	rq.Spec.Hard[corev1.ResourceName("limits.nvidia.com/gpu")] = resourceQuantity(gpuQuota)

	if memoryQuota != "" {
		memQuantity, err := parseResourceQuantity(memoryQuota)
		if err == nil {
			rq.Spec.Hard[corev1.ResourceMemory] = memQuantity
		}
	}

	_, err = m.k8sClient.CoreV1().ResourceQuotas(namespace).Update(ctx, rq, metav1.UpdateOptions{})
	return err
}
