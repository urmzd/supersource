// Package demo is the fixture Go package (dur.90, dur.91).
package demo

import "math"

// Sum adds xs left to right, skipping NaN values.
func Sum(xs []float64) float64 {
	// SOLUTION-BEGIN dur.90
	s := 0.0
	for _, x := range xs {
		if math.IsNaN(x) {
			continue
		}
		s += x
	}
	return s
	// SOLUTION-END
}
