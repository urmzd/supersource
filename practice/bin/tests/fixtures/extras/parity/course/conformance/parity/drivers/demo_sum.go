// Parity driver: demo.Sum (dur.90).
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"

	"tinyllm/ds/demo"
)

func main() {
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var c struct{ Xs []float64 }
		if err := json.Unmarshal(sc.Bytes(), &c); err != nil {
			os.Exit(1)
		}
		fmt.Println(demo.Sum(c.Xs))
	}
}
