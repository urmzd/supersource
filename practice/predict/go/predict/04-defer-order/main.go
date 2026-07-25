// 04-defer-order: defer runs last-in-first-out, its arguments are evaluated at
// the defer statement (not at the call), and whether it can change the returned
// value depends entirely on whether the return value has a name.
package main

import "fmt"

func namedReturn() (n int) {
	defer func() { n *= 2 }()
	n = 5
	return n
}

func plainReturn() int {
	n := 5
	defer func() { n *= 2 }()
	return n
}

func argsEvaluatedWhen() {
	i := 1
	defer fmt.Println("  deferred saw i =", i)
	i = 100
	fmt.Println("  body sees i =", i)
}

func main() {
	fmt.Println("1.", namedReturn())
	fmt.Println("2.", plainReturn())

	fmt.Println("3.")
	argsEvaluatedWhen()

	// Everything below is deferred, so it all fires after "6." prints.
	defer fmt.Println()
	for i := range 3 {
		defer fmt.Print(" ", i)
	}
	defer fmt.Print("4.")

	// A deferred closure reads the variable at call time, not at defer time.
	msg := "before"
	defer func() { fmt.Println("5.", msg) }()
	msg = "after"

	fmt.Println("6. main body done")
}
