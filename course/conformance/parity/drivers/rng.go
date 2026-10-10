// Parity driver for `rng`, implementation `go` (load.01): reads one JSON case
// per stdin line ({seed, seq, n_u32, n_uniform, n_normal}) and prints one
// JSON object per line, {"u32": [...], "uniform": [...], "normal": [...]},
// each list drawn from a fresh tinyllm/ds/rng PCG32(seed, seq), the same
// shape as drivers/rng.py.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"tinyllm/ds/rng"
)

type input struct {
	Seed     uint64  `json:"seed"`
	Seq      *uint64 `json:"seq"`
	NU32     int     `json:"n_u32"`
	NUniform int     `json:"n_uniform"`
	NNormal  int     `json:"n_normal"`
}

type output struct {
	U32     []uint32  `json:"u32"`
	Uniform []float64 `json:"uniform"`
	Normal  []float64 `json:"normal"`
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
			fmt.Fprintln(os.Stderr, "rng.go: bad case:", err)
			os.Exit(2)
		}
		seq := rng.DefaultSeq
		if in.Seq != nil {
			seq = *in.Seq
		}
		out := output{U32: []uint32{}, Uniform: []float64{}, Normal: []float64{}}
		g := rng.New(in.Seed, seq)
		for i := 0; i < in.NU32; i++ {
			out.U32 = append(out.U32, g.Uint32())
		}
		g = rng.New(in.Seed, seq)
		for i := 0; i < in.NUniform; i++ {
			out.Uniform = append(out.Uniform, g.Float64())
		}
		g = rng.New(in.Seed, seq)
		for i := 0; i < in.NNormal; i++ {
			out.Normal = append(out.Normal, g.Normal())
		}
		b, _ := json.Marshal(out)
		w.Write(b)
		w.WriteByte('\n')
		w.Flush()
	}
}
