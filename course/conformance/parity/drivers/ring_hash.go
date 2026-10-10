// Parity driver for `ring.hash`, implementation `go` (ds.09): reads one JSON
// case per stdin line ({vnodes, nodes, remove, keys, bounded}) and prints one
// JSON object per line, {"owners": [...]}, the node tinyllm/ds/ring assigns
// to each key, the same shape as course/oracle/ds.09/ring_golden.py.
// With "bounded": {"c": c}, keys are placed in order with GetBounded and each
// placement adds 1 to its node's load.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"tinyllm/ds/ring"
)

type input struct {
	Vnodes  int      `json:"vnodes"`
	Nodes   []string `json:"nodes"`
	Remove  []string `json:"remove"`
	Keys    []string `json:"keys"`
	Bounded *struct {
		C float64 `json:"c"`
	} `json:"bounded"`
}

func main() {
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var in input
		if err := json.Unmarshal([]byte(line), &in); err != nil {
			fmt.Fprintln(os.Stderr, "ring_hash.go: bad case:", err)
			os.Exit(2)
		}
		r := ring.New(in.Vnodes, nil)
		for _, n := range in.Nodes {
			r.Add(n)
		}
		for _, n := range in.Remove {
			r.Remove(n)
		}
		owners := make([]string, 0, len(in.Keys))
		load := map[string]int{}
		for _, k := range in.Keys {
			var n string
			if in.Bounded != nil {
				n, _ = r.GetBounded([]byte(k), func(x string) int { return load[x] }, in.Bounded.C)
				load[n]++
			} else {
				n, _ = r.Get([]byte(k))
			}
			owners = append(owners, n)
		}
		b, _ := json.Marshal(map[string][]string{"owners": owners})
		w.Write(b)
		w.WriteByte('\n')
		w.Flush()
	}
}
