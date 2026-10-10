// Course tests for load.02: the regression gate (go/loadgen/compare).
//
// The gate fails a head run only when it is worse than the base by more
// than the budget AND a one-sided permutation test says the difference is
// unlikely to be noise. The statistical test at the end checks both halves
// of that promise over many simulated pairs of runs; every other test is
// exact. Sample data comes from the tests' own SplitMix64 generator with fixed seeds.
package load_02

import (
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	"tinyllm/ds/rng"
	"tinyllm/loadgen"
	"tinyllm/loadgen/compare"
)

// testRand is the tests' own generator (SplitMix64), so test data never
// depends on the code under test or on math/rand (DESIGN D35).
type testRand struct{ s uint64 }

func newTestRand(a, b uint64) *testRand { return &testRand{s: a*0x9E3779B97F4A7C15 ^ b} }

func (r *testRand) next() uint64 {
	r.s += 0x9E3779B97F4A7C15
	z := r.s
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// Float64 is uniform on [0, 1) with 53 bits.
func (r *testRand) Float64() float64 { return float64(r.next()>>11) / (1 << 53) }

// IntN is uniform on [0, n) (the modulo bias is negligible for test sizes).
func (r *testRand) IntN(n int) int { return int(r.next() % uint64(n)) }

// ExpFloat64 is a standard exponential by inverse CDF.
func (r *testRand) ExpFloat64() float64 { return -math.Log(1 - r.Float64()) }

// NormFloat64 is a standard normal by Box-Muller (the cosine half).
func (r *testRand) NormFloat64() float64 {
	u1, u2 := r.Float64(), r.Float64()
	return math.Sqrt(-2*math.Log(1-u1)) * math.Cos(2*math.Pi*u2)
}

func lognormal(r *testRand, n int, sigma, scale float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = scale * math.Exp(sigma*r.NormFloat64())
	}
	return out
}

func stat(t *testing.T, m string) compare.Stat {
	t.Helper()
	_, s, err := compare.ParseMetric(m)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// report builds a loadgen report whose ttft histogram holds ttftMs (one
// sample per value) and `errors` failed requests.
func report(id string, ttftMs []float64, errors int) *loadgen.Report {
	var ss []loadgen.Sample
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for _, v := range ttftMs {
		d := time.Duration(v * float64(time.Millisecond))
		ss = append(ss, loadgen.Sample{Start: t0, TTFT: d, E2E: 2 * d, Tokens: 2})
	}
	for i := 0; i < errors; i++ {
		ss = append(ss, loadgen.Sample{Start: t0, Err: fmt.Errorf("HTTP 503")})
	}
	return loadgen.BuildReport(loadgen.Config{Target: "http://x", Schedule: loadgen.Constant(4), RunID: id}, ss, time.Minute)
}

func TestCompareHandExample(t *testing.T) {
	// WHY: the chapter's worked example. Base e2e samples {10, 20, 30} ms,
	//      head {40, 50, 60} ms, metric e2e_mean, budget 5%: base 20, head
	//      50, delta (50 - 20) / 20 = 1.5. Of the C(6, 3) = 20 ways to call
	//      three of the six values "head", only the observed one is this
	//      extreme, so the exact one-sided p is 1/20 = 0.05; 20000 random
	//      relabelings estimate it within 0.005. 0.05 is not below alpha
	//      0.05: three samples are too few to call anything a regression.
	// KIND: unit
	// CATCHES: m02, m06
	// CHAPTER: load.02 section 3
	v, err := compare.Gate([]float64{10, 20, 30}, []float64{40, 50, 60}, stat(t, "e2e_mean"),
		compare.Options{Metric: "e2e_mean", MaxRegress: 0.05, Permutations: 20000, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if v.Base != 20 || v.Head != 50 || v.Delta != 1.5 {
		t.Fatalf("base %v head %v delta %v, want 20, 50, 1.5", v.Base, v.Head, v.Delta)
	}
	if math.Abs(v.PValue-0.05) > 0.005 {
		t.Fatalf("p = %v, want 1/20 = 0.05 within 0.005", v.PValue)
	}
	if v.Regressed {
		t.Fatal("p = 0.05 is not below alpha = 0.05: not a regression")
	}
}

func TestParseMetricAndBudget(t *testing.T) {
	// WHY: the CLI's --metric and --max-regress strings. "5%" and "0.05" are
	//      the same budget; a percentage taken as a fraction would allow the
	//      head to be 6x slower. Unknown metrics are errors, not a silent
	//      pass.
	// KIND: unit
	// CATCHES: s05, m05
	// CHAPTER: load.02 section 4
	for in, want := range map[string]float64{"5%": 0.05, "0.05": 0.05, "12.5%": 0.125, "0": 0} {
		got, err := compare.ParseRegress(in)
		if err != nil || math.Abs(got-want) > 1e-12 {
			t.Errorf("ParseRegress(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	for _, bad := range []string{"", "five", "-5%", "NaN"} {
		if _, err := compare.ParseRegress(bad); err == nil {
			t.Errorf("ParseRegress(%q) must fail", bad)
		}
	}
	for in, want := range map[string]string{"ttft_p95": "ttft_ms/p95", "tpot_p50": "tpot_ms/p50", "e2e_mean": "e2e_ms/mean", "itl_p99": "itl_ms/p99", "error_rate": "error_rate/mean"} {
		m, s, err := compare.ParseMetric(in)
		if err != nil || m+"/"+s.Name != want {
			t.Errorf("ParseMetric(%q) = %q, %+v, %v; want %s", in, m, s, err, want)
		}
	}
	if _, s, _ := compare.ParseMetric("ttft_p95"); s.Q != 0.95 || s.Mean {
		t.Errorf("ttft_p95 stat %+v, want Q 0.95", s)
	}
	for _, bad := range []string{"ttft", "ttft_p0", "ttft_max", "latency_p95", "ttft_p100"} {
		if _, _, err := compare.ParseMetric(bad); err == nil {
			t.Errorf("ParseMetric(%q) must fail", bad)
		}
	}
}

func TestValueIsNearestRankAndLeavesInputAlone(t *testing.T) {
	// WHY: the gate's statistic must be the same nearest-rank percentile the
	//      report prints (ceil(q * n)-th smallest), computed by selection on
	//      a copy: sorting or partitioning the caller's slice in place would
	//      scramble the base/head split of the permutation test.
	// KIND: property
	// CATCHES: s08, m01, m08
	// CHAPTER: load.02 section 2
	r := newTestRand(5, 6)
	for trial := 0; trial < 300; trial++ {
		n := 1 + r.IntN(200)
		xs := make([]float64, n)
		for i := range xs {
			xs[i] = float64(r.IntN(50)) // many ties
		}
		orig := append([]float64(nil), xs...)
		sorted := append([]float64(nil), xs...)
		sort.Float64s(sorted)
		for _, q := range []float64{0.01, 0.07, 0.5, 0.55, 0.9, 0.95, 0.99} {
			rank := int(math.Ceil(q*float64(n) - 1e-9))
			if rank < 1 {
				rank = 1
			}
			want := sorted[rank-1]
			if got := (compare.Stat{Name: "p", Q: q}).Value(xs); got != want {
				t.Fatalf("n %d q %v: Value = %v, want %v", n, q, got, want)
			}
		}
		if fmt.Sprint(xs) != fmt.Sprint(orig) {
			t.Fatal("Value modified its input")
		}
	}
	if !math.IsNaN((compare.Stat{Mean: true}).Value(nil)) {
		t.Fatal("the statistic of no samples is NaN")
	}
}

func TestSamplesExpandTheHistogram(t *testing.T) {
	// WHY: a report keeps samples as [upper_ms, count] buckets; the test
	//      needs one value per request, so each bucket becomes count copies
	//      of its upper bound. error_rate is a 0/1 sample per request.
	// KIND: unit
	// CATCHES: s06, m04
	// CHAPTER: load.02 section 4
	rep := &loadgen.Report{RunID: "r", Requests: 5, Errors: 2, Histogram: map[string][][2]float64{
		"ttft_ms": {{10.5, 2}, {20, 1}},
	}}
	got, err := compare.Samples(rep, "ttft_ms")
	if err != nil || fmt.Sprint(got) != fmt.Sprint([]float64{10.5, 10.5, 20}) {
		t.Fatalf("Samples(ttft_ms) = %v, %v; want [10.5 10.5 20]", got, err)
	}
	e, err := compare.Samples(rep, "error_rate")
	sum := 0.0
	for _, x := range e {
		sum += x
	}
	if err != nil || len(e) != 5 || sum != 2 {
		t.Fatalf("Samples(error_rate) = %v, %v; want five values summing to 2", e, err)
	}
	if _, err := compare.Samples(rep, "tpot_ms"); err == nil {
		t.Fatal("a metric the report lacks must be an error")
	}
}

func TestPermutationTestIsSeededAndOneSided(t *testing.T) {
	// WHY: a CI gate must give the same verdict twice on the same files, so
	//      the relabelings come from rng.Stream(seed, shuffle); and only a
	//      WORSE head counts: a head that got faster has p near 1, never a
	//      small two-sided p.
	// KIND: unit
	// CATCHES: s01, s07
	// CHAPTER: load.02 section 5, Pitfalls
	r := newTestRand(7, 8)
	base := lognormal(r, 200, 0.2, 100)
	faster := lognormal(r, 200, 0.2, 80)
	s := stat(t, "ttft_p50")
	p1 := compare.PermutationTest(base, faster, s, 500, rng.Stream(3, rng.PurposeShuffle))
	p2 := compare.PermutationTest(base, faster, s, 500, rng.Stream(3, rng.PurposeShuffle))
	if p1 != p2 {
		t.Fatalf("same seed, different p: %v vs %v", p1, p2)
	}
	if p1 < 0.9 {
		t.Fatalf("p = %v for a FASTER head; a one-sided test gives p near 1", p1)
	}
	slower := lognormal(r, 200, 0.2, 130)
	if p := compare.PermutationTest(base, slower, s, 500, rng.Stream(3, rng.PurposeShuffle)); p > 0.01 {
		t.Fatalf("p = %v for a head 30%% slower in the median, want < 0.01", p)
	}
}

func TestPValueIsNeverZero(t *testing.T) {
	// WHY: with base all 1 ms and head all 100 ms no relabeling is as
	//      extreme as the observed one (1 in C(100, 50)), so the count is 0;
	//      the observed labeling is itself one of the possible labelings, so
	//      p = (1 + 0) / (1 + n), never 0.
	// KIND: boundary
	// CATCHES: s02
	// CHAPTER: load.02 section 2
	base, head := make([]float64, 50), make([]float64, 50)
	for i := range base {
		base[i], head[i] = 1, 100
	}
	p := compare.PermutationTest(base, head, compare.Stat{Name: "mean", Mean: true}, 999, rng.Stream(0, rng.PurposeShuffle))
	if p != 1.0/1000 {
		t.Fatalf("p = %v, want 1/1000", p)
	}
}

func TestGateNeedsBothBudgetAndSignificance(t *testing.T) {
	// WHY: two ways to cry wolf. One request per run: head 10x slower but p
	//      about 0.5, noise, not a regression. 4000 requests 2% slower: p
	//      tiny but within the 5% budget, not a regression. Only worse beyond
	//      the budget AND significant fails the gate.
	// KIND: unit
	// CATCHES: s03, s04
	// CHAPTER: load.02 section 5, Pitfalls
	o := compare.Options{Metric: "ttft_mean", MaxRegress: 0.05, Permutations: 400, Seed: 1}
	v, _ := compare.Gate([]float64{10}, []float64{100}, stat(t, "ttft_mean"), o)
	if v.Regressed || v.Delta != 9 {
		t.Fatalf("one sample each: %+v; want delta 9, not regressed (p %v)", v, v.PValue)
	}
	r := newTestRand(9, 10)
	base := lognormal(r, 4000, 0.05, 100)
	head := lognormal(r, 4000, 0.05, 102)
	v, _ = compare.Gate(base, head, stat(t, "ttft_mean"), o)
	if v.Regressed || v.PValue >= 0.05 {
		t.Fatalf("2%% slower with 4000 samples: %+v; want significant (p < 0.05) but within budget", v)
	}
	head = lognormal(r, 4000, 0.05, 110)
	if v, _ = compare.Gate(base, head, stat(t, "ttft_mean"), o); !v.Regressed {
		t.Fatalf("10%% slower with 4000 samples: %+v; want regressed", v)
	}
}

func TestZeroBaseline(t *testing.T) {
	// WHY: a base with error_rate 0 is the common case; any significant
	//      rise from 0 is an infinite relative change, never a division by
	//      zero that reads as "no change".
	// KIND: boundary
	// CATCHES: m07
	// CHAPTER: load.02 section 4
	o := compare.Options{Metric: "error_rate", MaxRegress: 0.05, Permutations: 400, Seed: 2}
	v, err := compare.Compare(report("b", []float64{10}, 0), report("h", []float64{10}, 0), o)
	if err != nil || v.Regressed || v.Delta != 0 {
		t.Fatalf("0 errors vs 0 errors: %+v, %v; want delta 0", v, err)
	}
	ok := make([]float64, 170)
	for i := range ok {
		ok[i] = 10
	}
	v, err = compare.Compare(report("b", ok, 0), report("h", ok[:140], 30), o)
	if err != nil || !math.IsInf(v.Delta, 1) || !v.Regressed {
		t.Fatalf("0 then 30 of 170 failed: %+v, %v; want delta +Inf and regressed", v, err)
	}
}

func TestCompareReportsEndToEnd(t *testing.T) {
	// WHY: the CLI path: two loadgen reports, the TTFT histogram expanded
	//      into samples, the default alpha (0.05) and permutation count. A
	//      head 30% slower fails the gate; the base against itself passes.
	// KIND: unit
	// CATCHES: m03
	// CHAPTER: load.02 section 4
	r := newTestRand(11, 12)
	base := report("base", lognormal(r, 300, 0.2, 100), 0)
	head := report("head", lognormal(r, 300, 0.2, 130), 0)
	o := compare.Options{Metric: "ttft_p95", MaxRegress: 0.05, Seed: 4}
	v, err := compare.Compare(base, head, o)
	if err != nil || !v.Regressed {
		t.Fatalf("30%% slower: %+v, %v; want regressed", v, err)
	}
	if v, err = compare.Compare(base, base, o); err != nil || v.Regressed || v.Delta != 0 {
		t.Fatalf("base vs itself: %+v, %v; want delta 0, passed", v, err)
	}
}

func TestIdenticalRunsPassAndShiftsAreCaught(t *testing.T) {
	// WHY: the gate's two promises (DESIGN load.02, S): identical runs pass
	//      at least 95% of the time, and a +10% shift of every latency is
	//      caught at least 90% of the time. 100 simulated pairs each (400
	//      lognormal samples, sigma 0.2, metric ttft_p95, budget 5%): the
	//      bounds below are binomial at about p = 1e-3, so a correct gate
	//      fails this test about once in a thousand seeds: at most 12 false
	//      alarms (rate 5%), at least 82 catches (power 90%).
	// KIND: statistical
	// CATCHES: s03
	// CHAPTER: load.02 section 2
	r := newTestRand(13, 14)
	s := stat(t, "ttft_p95")
	alarms, caught := 0, 0
	for trial := 0; trial < 100; trial++ {
		o := compare.Options{Metric: "ttft_p95", MaxRegress: 0.05, Permutations: 200, Seed: uint64(trial)}
		base := lognormal(r, 400, 0.2, 100)
		same := lognormal(r, 400, 0.2, 100)
		if v, _ := compare.Gate(base, same, s, o); v.Regressed {
			alarms++
		}
		shifted := lognormal(r, 400, 0.2, 110)
		if v, _ := compare.Gate(base, shifted, s, o); v.Regressed {
			caught++
		}
	}
	t.Logf("false alarms %d/100, caught %d/100", alarms, caught)
	if alarms > 12 {
		t.Errorf("%d of 100 identical pairs failed the gate (want <= 12: false alarm rate 5%%)", alarms)
	}
	if caught < 82 {
		t.Errorf("%d of 100 +10%% shifts caught (want >= 82: power 90%%)", caught)
	}
}
