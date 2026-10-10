package ag_10

import (
	"context"
	"encoding/json"
	"math"
	"testing"
	"time"

	"tinyllm/agent/eval"
	"tinyllm/agent/eval/scorers"
)

func obs(output, truth string) eval.Observation {
	return eval.Observation{Output: json.RawMessage(output), GroundTruth: json.RawMessage(truth), Annotations: map[string]json.RawMessage{}}
}

func score(t *testing.T, s eval.Scorer, o eval.Observation) float64 {
	t.Helper()
	v, err := s.Score(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	return v.Value
}

func TestTextAndSchemaScorers(t *testing.T) {
	// WHY: normalization should ignore outer punctuation and whitespace while retaining internal punctuation.
	// KIND: unit
	// CATCHES: s01
	// CHAPTER: ag.10 section 2
	o := obs(`" Chained FNV-1a 64. "`, `"chained   fnv-1a 64"`)
	if got := score(t, scorers.Exact(), o); got != 1 {
		t.Fatalf("exact=%v, want 1", got)
	}
	re, err := scorers.Regex(`FNV-1a`)
	if err != nil {
		t.Fatal(err)
	}
	if got := score(t, re, o); got != 1 {
		t.Fatalf("regex=%v, want 1", got)
	}
	schema, err := scorers.Schema(json.RawMessage(`{"type":"object","required":["ok"],"properties":{"ok":{"type":"boolean"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := score(t, schema, obs(`{"ok":true}`, `{}`)); got != 1 {
		t.Fatalf("schema=%v, want 1", got)
	}
}

func TestTimingMetrics(t *testing.T) {
	// WHY: latency scorers use the first and last token timestamps, and missing tokens are a scorer error rather than zero latency.
	// KIND: unit
	// CATCHES: s02
	// CHAPTER: ag.10 section 2
	o := eval.Observation{Timing: eval.Timing{TokenTimes: []time.Duration{10 * time.Millisecond, 30 * time.Millisecond, 50 * time.Millisecond}}}
	if got := score(t, scorers.TTFT(), o); got != 10 {
		t.Fatalf("ttft=%v, want 10ms", got)
	}
	if got := score(t, scorers.TTLT(), o); got != 50 {
		t.Fatalf("ttlt=%v, want 50ms", got)
	}
	if got := score(t, scorers.ITL(), o); got != 20 {
		t.Fatalf("itl=%v, want 20ms", got)
	}
	if _, err := scorers.TTFT().Score(context.Background(), eval.Observation{}); err == nil {
		t.Fatal("empty stream scored as a latency")
	}
}

func TestRankingHandExample(t *testing.T) {
	// WHY: hand calculations pin reciprocal rank and graded DCG at ranks starting from one, and duplicate ids count only at first rank.
	// KIND: unit
	// CATCHES: s03
	// CHAPTER: ag.10 section 3
	o := obs(`"answer [1]"`, `{"chunks":{"a":2,"c":1}}`)
	o.Annotations["retrieved"] = json.RawMessage(`["b","a","a","c"]`)
	o.Annotations["citations"] = json.RawMessage(`[ {"n":1,"chunk_id":"a"} ]`)
	if got := score(t, scorers.HitAtK(2), o); got != 1 {
		t.Fatalf("hit@2=%v", got)
	}
	if got := score(t, scorers.MRR(), o); got != .5 {
		t.Fatalf("mrr=%v, want .5", got)
	}
	want := scorers.DCG([]float64{0, 2, 1}) / scorers.DCG([]float64{2, 1})
	if got := score(t, scorers.NDCG(3), o); math.Abs(got-want) > 1e-12 {
		t.Fatalf("ndcg=%v want %v", got, want)
	}
	if got := score(t, scorers.CitationPrecision(), o); got != 1 {
		t.Fatalf("citation precision=%v", got)
	}
	if got := score(t, scorers.CitationRecall(), o); got != .5 {
		t.Fatalf("citation recall=%v", got)
	}
}

func TestStateChangeCriteria(t *testing.T) {
	// WHY: a state grader must catch both omitted required updates and side effects outside the allowed set.
	// KIND: unit
	// CATCHES: s04
	// CHAPTER: ag.10 section 2
	before := scorers.State{"ticket/1": {"status": json.RawMessage(`"open"`), "updated_at": json.RawMessage(`1`)}}
	after := scorers.State{"ticket/1": {"status": json.RawMessage(`"closed"`), "updated_at": json.RawMessage(`2`)}}
	task := scorers.Task{Required: map[string]json.RawMessage{"ticket/1.status": json.RawMessage(`"closed"`)}, Allowed: []string{"ticket/1.status"}}
	g := scorers.GradeState(before, after, task, nil, scorers.StateConfig{})
	if !g.Passed {
		t.Fatalf("valid state change rejected: %+v", g)
	}
}
