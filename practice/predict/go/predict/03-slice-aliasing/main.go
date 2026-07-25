// 03-slice-aliasing: a slice is a view (pointer, len, cap) onto an array.
// append writes into that array when there is spare capacity, and allocates a
// fresh one when there is not. Whether the caller sees your write depends on
// arithmetic you probably did not do.
package main

import "fmt"

func main() {
	a := []int{1, 2, 3, 4, 5}
	b := a[1:3]
	fmt.Println("1.", len(b), cap(b))

	b = append(b, 99)
	fmt.Println("2.", a)
	fmt.Println("3.", b)

	// Full slice expression caps the view, so append must copy.
	c := a[1:3:3]
	c = append(c, 100)
	fmt.Println("4.", a)
	fmt.Println("5.", c)

	// Growth reallocates: the alias is broken from here on.
	d := make([]int, 0, 2)
	d = append(d, 1, 2)
	e := d[:1]
	d = append(d, 3)
	e[0] = -1
	fmt.Println("6.", d, e)

	// copy is bounded by the shorter side and never grows the destination.
	dst := make([]int, 2)
	n := copy(dst, []int{7, 8, 9})
	fmt.Println("7.", n, dst)

	// Passing a slice by value still shares the backing array.
	f := []int{1, 2, 3}
	mutate(f)
	grow(f)
	fmt.Println("8.", f)
}

func mutate(xs []int) { xs[0] = 42 }

func grow(xs []int) { xs = append(xs, 4); xs[0] = 1000 }
