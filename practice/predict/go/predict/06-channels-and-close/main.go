// 06-channels-and-close: receiving from a closed channel is not an error and
// never blocks, which is the whole basis of fan-out shutdown. Everything here
// is ordered on purpose so the output is deterministic.
package main

import (
	"fmt"
	"sync"
)

func main() {
	ch := make(chan int, 3)
	ch <- 1
	ch <- 2
	fmt.Println("1.", len(ch), cap(ch))

	close(ch)

	// Sends are gone but buffered values remain, in order.
	v1, ok1 := <-ch
	v2, ok2 := <-ch
	v3, ok3 := <-ch
	fmt.Println("2.", v1, ok1)
	fmt.Println("3.", v2, ok2)
	fmt.Println("4.", v3, ok3)

	// range over a closed channel drains and stops.
	ch2 := make(chan string, 2)
	ch2 <- "a"
	ch2 <- "b"
	close(ch2)
	fmt.Print("5.")
	for s := range ch2 {
		fmt.Print(" ", s)
	}
	fmt.Println()

	// select with a default never blocks.
	empty := make(chan int)
	select {
	case v := <-empty:
		fmt.Println("6. received", v)
	default:
		fmt.Println("6. nothing ready")
	}

	// A nil channel blocks forever, so its select case is never chosen.
	var nilCh chan int
	select {
	case <-nilCh:
		fmt.Println("7. impossible")
	default:
		fmt.Println("7. nil channel is never ready")
	}

	// One worker, one result channel: ordering is forced by the WaitGroup.
	var wg sync.WaitGroup
	results := make(chan int, 3)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range 3 {
			results <- i * i
		}
	}()
	wg.Wait()
	close(results)
	sum := 0
	for r := range results {
		sum += r
	}
	fmt.Println("8.", sum)
}
