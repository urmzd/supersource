// Package sstesting holds the frozen Go test helpers (course/DESIGN.md D35).
package sstesting

import "math"

// Close reports |a-b| <= atol + rtol*|b|.
func Close(a, b, rtol, atol float64) bool {
	return math.Abs(a-b) <= atol+rtol*math.Abs(b)
}
