package demo

import "errors"

// ErrEmpty is returned by Mean for an empty slice.
var ErrEmpty = errors.New("demo: mean of an empty slice")

// Mean is Sum(xs) / len(xs).
func Mean(xs []float64) (float64, error) {
	// SOLUTION-BEGIN dur.91
	if len(xs) == 0 {
		return 0, ErrEmpty
	}
	return Sum(xs) / float64(len(xs)), nil
	// SOLUTION-END
}
