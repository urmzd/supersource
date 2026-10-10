package ag_11

import (
	"context"
	"fmt"
	"testing"

	"tinyllm/agent/eval"
	"tinyllm/agent/eval/judge"
)

type fakeGenerator struct {
	replies []string
	prompts []string
}

func (f *fakeGenerator) Generate(_ context.Context, p string) (string, error) {
	f.prompts = append(f.prompts, p)
	r := f.replies[0]
	f.replies = f.replies[1:]
	return r, nil
}

func TestJudgeParsesEvidence(t *testing.T) {
	// WHY: a judge score is usable only with a bounded numeric grade and evidence.
	// KIND: unit
	// CATCHES: s01
	// CHAPTER: ag.11 section 2
	g := &fakeGenerator{replies: []string{`{"score":0.8,"evidence":"answers with cited source"}`}}
	s := judge.NewJudgeScorer(g, "Use citations")
	r, err := s.Score(context.Background(), eval.Observation{Output: []byte(`"answer"`), GroundTruth: []byte(`"source"`)})
	if err != nil {
		t.Fatal(err)
	}
	if r.Value != .8 || r.Reason == "" {
		t.Fatalf("score=%+v", r)
	}
}

func TestJudgeRejectsMalformedOutput(t *testing.T) {
	// WHY: malformed judge output is a measurement failure, never score zero.
	// KIND: fault
	// CATCHES: s01
	// CHAPTER: ag.11 section 5
	for _, reply := range []string{`not json`, `{"score":1.2,"evidence":"high"}`, `{"score":0.4}`, `{"score":0.5,"evidence":"ok"} trailing`} {
		_, err := judge.NewJudgeScorer(&fakeGenerator{replies: []string{reply}}, "rubric").Score(context.Background(), eval.Observation{})
		if err == nil {
			t.Fatalf("accepted %q", reply)
		}
	}
}

func TestPairwisePositionSwap(t *testing.T) {
	// WHY: swapping candidates detects a judge that chooses the first position regardless of response quality.
	// KIND: property
	// CATCHES: s02
	// CHAPTER: ag.11 section 3
	g := &fakeGenerator{replies: []string{`{"winner":"A"}`, `{"winner":"A"}`}}
	r, err := judge.Pairwise(context.Background(), g, "Which is better?", "good", "bad")
	if err != nil {
		t.Fatal(err)
	}
	if !r.PositionSensitive || r.Winner != "tie" {
		t.Fatalf("pairwise=%+v", r)
	}
	if len(g.prompts) != 2 {
		t.Fatalf("calls=%d, want swapped pair", len(g.prompts))
	}
}

func TestSampledTolerance(t *testing.T) {
	// WHY: sampled scoring must make a deterministic subset decision for a case id.
	// KIND: unit
	// CATCHES: s03
	// CHAPTER: ag.11 section 4
	inner := constantScorer{}
	a := judge.Sampled(inner, 4, .1)
	skipped := 0
	for i := 0; i < 40; i++ {
		x := eval.Observation{ID: fmt.Sprintf("case-%02d", i)}
		r1, e1 := a.Score(context.Background(), x)
		r2, e2 := a.Score(context.Background(), x)
		if (e1 == nil) != (e2 == nil) || r1.Value != r2.Value {
			t.Fatalf("sampling changed for stable case %s", x.ID)
		}
		if e1 != nil {
			skipped++
		}
	}
	if skipped == 0 || skipped == 40 {
		t.Fatalf("sampled subset has %d skipped cases", skipped)
	}
}

func TestKappaAgainstLabels(t *testing.T) {
	// WHY: agreement must be corrected for label frequencies before judge quality is reported.
	// KIND: unit
	// CHAPTER: ag.11 section 2
	k, err := judge.Kappa([]string{"a", "b", "a", "b"}, []string{"a", "b", "b", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if k < 0 || k > 1 {
		t.Fatalf("kappa=%v, want a value in [0,1]", k)
	}
}

type constantScorer struct{}

func (constantScorer) Name() string { return "constant" }
func (constantScorer) Score(context.Context, eval.Observation) (eval.Score, error) {
	return eval.Score{Value: 1}, nil
}
