package demo_test

import (
	"math"
	"testing"

	"tinyllm/ds/demo"
)

func TestSumMine(t *testing.T) {
	if got := demo.Sum([]float64{1, 2, 3}); got != 6 {
		t.Fatalf("Sum = %v, want 6", got)
	}
}

func TestSkipsNaNMine(t *testing.T) {
	if got := demo.Sum([]float64{1, math.NaN(), 2}); got != 3 {
		t.Fatalf("Sum = %v, want 3", got)
	}
}
