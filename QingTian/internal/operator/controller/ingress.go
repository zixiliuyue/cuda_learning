package controller

import (
	"context"
	"k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"

	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	qingtianv1 "qingtian/internal/operator/apis/qingtian/v1"
)

// ensureIngress ensures the Ingress exists when enabled
func (r *QingTianClusterReconciler) ensureIngress(
	ctx context.Context,
	cluster *qingtianv1.QingTianCluster,
) error {
	name := getApiserverName(cluster)
	var existing networkingv1.Ingress
	err := r.Get(ctx, client.ObjectKey{Namespace: cluster.Namespace, Name: name}, &existing)
	if err == nil {
		// Ingress already exists
		return nil
	}
	if !errors.IsNotFound(err) {
		return err
	}

	ing := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cluster.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(cluster, qingtianv1.GroupVersion.WithKind("QingTianCluster")),
			},
			Annotations: cluster.Spec.Ingress.Annotations,
		},
		Spec: networkingv1.IngressSpec{
			IngressClassName: cluster.Spec.Ingress.IngressClassName,
			Rules: []networkingv1.IngressRule{
				{
					Host: cluster.Spec.Ingress.Host,
					IngressRuleValue: networkingv1.IngressRuleValue{
						HTTP: &networkingv1.HTTPIngressRuleValue{
							Paths: []networkingv1.HTTPIngressPath{
								{
									Path: "/",
									PathType: func() *networkingv1.PathType {
										pt := networkingv1.PathTypePrefix
										return &pt
									}(),
									Backend: networkingv1.IngressBackend{
										Service: &networkingv1.IngressServiceBackend{
											Name: getServiceName(cluster),
											Port: networkingv1.ServiceBackendPort{
												Number: 80,
											},
										},
									},
								},
							},
						},
					},
				},
			},
			TLS: make([]networkingv1.IngressTLS, len(cluster.Spec.Ingress.TLS)),
		},
	}

	// Convert TLS config
	for i, tls := range cluster.Spec.Ingress.TLS {
		ing.Spec.TLS[i] = networkingv1.IngressTLS{
			Hosts:      tls.Hosts,
			SecretName: tls.SecretName,
		}
	}

	if err := r.Create(ctx, ing); err != nil {
		return err
	}
	return nil
}
