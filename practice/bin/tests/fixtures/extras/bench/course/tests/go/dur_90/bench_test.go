package dur_90

import (
	"testing"

	"tinyllm/ds/demo"
)

func BenchmarkSum1k(b *testing.B) {
	xs := make([]float64, 1000)
	for i := range xs {
		xs[i] = float64(i)
	}
	for i := 0; i < b.N; i++ {
		demo.Sum(xs)
	}
}
