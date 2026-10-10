package ag_12

import (
	"context"
	"encoding/json"
	"testing"

	"tinyllm/agent/eval"
	"tinyllm/agent/eval/experiment"
	"tinyllm/workflows"
)

type outputScore struct{}

func (outputScore) Name() string { return "value" }
func (outputScore) Score(_ context.Context, o eval.Observation) (eval.Score, error) {
	var v float64
	if err := json.Unmarshal(o.Output, &v); err != nil {
		return eval.Score{}, err
	}
	return eval.Score{Value: v}, nil
}

func subjects(base, exp float64) (eval.Subject, eval.Subject) {
	return func(_ context.Context, o *eval.Observation) error { o.Output, _ = json.Marshal(base); return nil }, func(_ context.Context, o *eval.Observation) error { o.Output, _ = json.Marshal(exp); return nil }
}

func TestPairedDeltaHandExample(t *testing.T) {
	// WHY: pairing the same three cases gives deltas 1, 0, 0 and mean effect one third.
	// KIND: unit
	// CATCHES: s01
	// CHAPTER: ag.12 section 3
	b, e := subjects(0, 1)
	obs := []eval.Observation{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	// A case-dependent experiment produces a single improvement for case a.
	e = func(_ context.Context, o *eval.Observation) error {
		if o.ID == "a" {
			o.Output = []byte(`1`)
		} else {
			o.Output = []byte(`0`)
		}
		return nil
	}
	r, err := experiment.RunExperiment(context.Background(), obs, b, e, []eval.Scorer{outputScore{}}, experiment.WithBootstrap(100, .05), experiment.WithSeed(5))
	if err != nil {
		t.Fatal(err)
	}
	if got := r.Metrics["value"]; got.Delta != 1.0/3 || got.NPairs != 3 || got.CILow > got.CIHigh {
		t.Fatalf("stat=%+v", got)
	}
}

func TestFailedPairsExcluded(t *testing.T) {
	// WHY: a scorer error on either arm removes that case from the paired metric instead of becoming a zero.
	// KIND: fault
	// CATCHES: s02
	// CHAPTER: ag.12 section 2
	b, e := subjects(0, 1)
	b = func(_ context.Context, o *eval.Observation) error {
		if o.ID == "b" {
			o.Output = []byte(`not-json`)
		} else {
			o.Output = []byte(`0`)
		}
		return nil
	}
	obs := []eval.Observation{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	r, err := experiment.RunExperiment(context.Background(), obs, b, e, []eval.Scorer{outputScore{}}, experiment.WithBootstrap(50, .05), experiment.WithSeed(2))
	if err != nil {
		t.Fatal(err)
	}
	if got := r.Metrics["value"]; got.NPairs != 2 {
		t.Fatalf("pairs=%d, want 2", got.NPairs)
	}
}

func TestBootstrapPairs(t *testing.T) {
	// WHY: every bootstrap draw must retain the base and experiment scores for one case as a pair.
	// KIND: property
	// CATCHES: s01
	// CHAPTER: ag.12 section 2
	b, e := subjects(1, 1)
	obs := []eval.Observation{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	r, err := experiment.RunExperiment(context.Background(), obs, b, e, []eval.Scorer{outputScore{}}, experiment.WithBootstrap(100, .05), experiment.WithSeed(7))
	if err != nil {
		t.Fatal(err)
	}
	if s := r.Metrics["value"]; s.Delta != 0 || s.CILow != 0 || s.CIHigh != 0 || s.PValue != 1 {
		t.Fatalf("null pairs=%+v", s)
	}
}

func TestKnownEffect(t *testing.T) {
	// WHY: a consistent paired treatment effect should yield a positive interval and small permutation p-value.
	// KIND: statistical
	// CATCHES: s01
	// CHAPTER: ag.12 section 4
	b, e := subjects(0, 1)
	obs := make([]eval.Observation, 80)
	for i := range obs {
		obs[i].ID = string(rune('a'+i%26)) + string(rune('A'+i/26))
	}
	r, err := experiment.RunExperiment(context.Background(), obs, b, e, []eval.Scorer{outputScore{}}, experiment.WithBootstrap(500, .05), experiment.WithSeed(10))
	if err != nil {
		t.Fatal(err)
	}
	if s := r.Metrics["value"]; s.Delta != 1 || s.CILow != 1 || s.CIHigh != 1 || s.PValue >= .05 {
		t.Fatalf("effect=%+v", s)
	}
}

func TestNullCoverage(t *testing.T) {
	// WHY: the null treatment has paired zero effect, so its interval must cover zero for every fixed seed.
	// KIND: statistical
	// CATCHES: s01
	b, e := subjects(.5, .5)
	obs := make([]eval.Observation, 30)
	for i := range obs {
		obs[i].ID = "n" + string(rune(i+32))
	}
	for seed := uint64(0); seed < 25; seed++ {
		r, err := experiment.RunExperiment(context.Background(), obs, b, e, []eval.Scorer{outputScore{}}, experiment.WithBootstrap(100, .05), experiment.WithSeed(seed))
		if err != nil {
			t.Fatal(err)
		}
		s := r.Metrics["value"]
		if s.CILow > 0 || s.CIHigh < 0 {
			t.Fatalf("seed %d excludes null: %+v", seed, s)
		}
	}
}

func TestEvalSuiteSpecCarriesAB(t *testing.T) {
	// WHY: durable EvalSuite must pass judge and A/B settings through to the learner's eval command.
	// KIND: conformance
	// CATCHES: s04
	// CHAPTER: ag.12 section 4
	in := workflows.EvalSuiteInput{
		Name: "docsqa", Suites: []string{"helpdesk"}, Seed: 4,
		Subjects: []workflows.EvalSubject{{ID: "base"}, {ID: "exp"}},
		Scorers:  []string{"exact"}, AB: &workflows.EvalAB{Base: "base", Exp: "exp", NBoot: 1000, Alpha: .05},
		Judge: &workflows.EvalJudge{BaseURL: "http://judge/v1", Model: "small", Rubric: "helpfulness", Samples: 2, Pairwise: true},
	}
	raw, err := workflows.MarshalSpec(in, "helpdesk")
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]json.RawMessage
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"ab", "judge", "scorers"} {
		if len(got[key]) == 0 {
			t.Fatalf("eval spec lost %q: %s", key, raw)
		}
	}
}
