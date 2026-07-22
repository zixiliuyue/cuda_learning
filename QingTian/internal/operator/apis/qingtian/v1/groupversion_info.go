// Package v1 contains API Schema definitions for the qingtian v1 API group
// +kubebuilder:object:generate=true
// +groupName=qingtian.ai
package v1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/scheme"
)

var (
	// GroupVersion is group version used to register these objects
	GroupVersion = schema.GroupVersion{Group: "qingtian.ai", Version: "v1"}

	// SchemeBuilder is used to add go types to the GroupVersionScheme
	SchemeBuilder = scheme.Builder{GroupVersion: GroupVersion}

	// AddToScheme adds the types in this group-version to the given scheme.
	AddToScheme = SchemeBuilder.AddToScheme
)
