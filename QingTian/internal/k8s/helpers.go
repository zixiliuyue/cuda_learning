package k8s

import (
	"fmt"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
)

// resourceQuantity converts an int to resource.Quantity.
func resourceQuantity(n int) resource.Quantity {
	return *resource.NewQuantity(int64(n), resource.DecimalSI)
}

// parseResourceQuantity parses a resource string like "128Gi" to resource.Quantity.
func parseResourceQuantity(s string) (resource.Quantity, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return resource.Quantity{}, fmt.Errorf("empty resource string")
	}

	// Try to parse directly
	q, err := resource.ParseQuantity(s)
	if err == nil {
		return q, nil
	}

	// Fallback: parse number followed by unit manually
	var numStr string
	var unit string
	for i, c := range s {
		if (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' {
			numStr = s[:i+1]
		} else {
			unit = s[i:]
			break
		}
	}

	if numStr == "" {
		return resource.Quantity{}, fmt.Errorf("invalid resource format: %s", s)
	}

	num, err := strconv.ParseInt(numStr, 10, 64)
	if err != nil {
		return resource.Quantity{}, err
	}

	// Convert common units
	switch strings.ToLower(unit) {
	case "g", "gb", "gi":
		num = num * 1024 * 1024 * 1024
	case "m", "mb", "mi":
		num = num * 1024 * 1024
	case "k", "kb", "ki":
		num = num * 1024
	}

	return *resource.NewQuantity(num, resource.DecimalSI), nil
}
