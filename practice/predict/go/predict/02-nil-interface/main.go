// 02-nil-interface: an interface value is a (type, value) pair. It is nil only
// when *both* halves are nil, which is why the most common Go bug in this file
// looks completely correct.
package main

import "fmt"

type opErr struct{ code int }

func (e *opErr) Error() string { return fmt.Sprintf("op failed: %d", e.code) }

// Returns the concrete pointer type. Looks fine. Is not fine.
func mayFail(fail bool) error {
	var e *opErr
	if fail {
		e = &opErr{code: 7}
	}
	return e
}

// Returns an untyped nil on the success path.
func mayFailFixed(fail bool) error {
	if fail {
		return &opErr{code: 7}
	}
	return nil
}

func main() {
	fmt.Println("1.", mayFail(false) == nil)
	fmt.Println("2.", mayFailFixed(false) == nil)

	err := mayFail(false)
	fmt.Printf("3. %T %v\n", err, err)

	var anything any
	fmt.Println("4.", anything == nil)

	var p *opErr
	anything = p
	fmt.Println("5.", anything == nil, p == nil)

	// A nil map reads fine and writes fatally; a nil slice appends fine.
	var m map[string]int
	var s []int
	fmt.Println("6.", m["missing"], len(m), s == nil, len(append(s, 1)))

	// Method call on a nil receiver is legal as long as the method allows it.
	var e2 *opErr
	defer func() {
		fmt.Println("7. recovered:", recover() != nil)
	}()
	fmt.Println("7a.", e2.Error())
}
