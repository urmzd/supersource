// 01-loop-var-capture: what does a closure created inside a loop actually
// close over? Go 1.22 changed the answer, so this one is worth re-predicting
// even if you were sure about it in 2021.
package main

import "fmt"

func main() {
	// (1) three-clause loop, variable declared in the loop header.
	var fns []func() int
	for i := 0; i < 3; i++ {
		fns = append(fns, func() int { return i })
	}
	fmt.Print("1.")
	for _, f := range fns {
		fmt.Printf(" %d", f())
	}
	fmt.Println()

	// (2) same loop, but the variable is declared outside the header.
	var gns []func() int
	j := 0
	for ; j < 3; j++ {
		gns = append(gns, func() int { return j })
	}
	fmt.Print("2.")
	for _, g := range gns {
		fmt.Printf(" %d", g())
	}
	fmt.Println()

	// (3) range loop capturing both the index and the element.
	xs := []string{"a", "b", "c"}
	var hns []func() string
	for i, x := range xs {
		hns = append(hns, func() string { return fmt.Sprintf("%d%s", i, x) })
	}
	fmt.Print("3.")
	for _, h := range hns {
		fmt.Printf(" %s", h())
	}
	fmt.Println()

	// (4) the loop variable is reused as a pointer target.
	var ptrs []*int
	for i := range 3 {
		ptrs = append(ptrs, &i)
	}
	fmt.Print("4.")
	for _, p := range ptrs {
		fmt.Printf(" %d", *p)
	}
	fmt.Println()
}
