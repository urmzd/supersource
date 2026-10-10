// Course tests for dur.11: TrainRun and EvalSuite (go/workflows/
// train_run.go, eval_suite.go), driven by the replaying simulator in
// sim_test.go. The train and eval activities are fakes that keep the
// subprocess contract's rules over an in-memory run directory: train resumes
// from the run's LATEST checkpoint, writes one checkpoint per 100 steps, and
// its loss after step k is a fixed function of k, so a resumed run must end
// with exactly the loss of an uninterrupted one.
package dur_11

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"tinyllm/workflows"
)

const handSpec = `{"name":"tiny","steps":1200,"eval_every":500,"seed":7,"model":{"vocab_size":256,"tl_arch":"bigram"}}`

// runDir is the fake runs/<name>/: the newest checkpoint and the loss at
// every checkpointed step, plus counts of the effects.
type runDir struct {
	latest    int
	loss      map[int]float64
	trained   int            // optimizer steps actually taken (re-done work included)
	evals     map[string]int // checkpoint/suite -> evaluations written
	specs     []map[string]any
	crashMid  bool // the next train activity dies halfway (its checkpoints so far stay)
	evalFails string
}

func newRunDir() *runDir { return &runDir{loss: map[int]float64{}, evals: map[string]int{}} }

// stepLoss is the deterministic loss after step k from the loss before it.
func stepLoss(prev float64, k int) float64 {
	return prev*0.999 + 0.001*math.Sin(float64(k))
}

func fakeTraining(s *sim, d *runDir) {
	s.register(workflows.ActivityTrain, func(c call) (any, error) {
		var spec map[string]any
		json.Unmarshal(c.Input, &spec)
		d.specs = append(d.specs, spec)
		until := int(spec["steps"].(float64))
		step, loss := d.latest, 4.0
		if step > 0 {
			loss = d.loss[step]
		}
		stop := until
		if d.crashMid {
			d.crashMid = false
			stop = step + (until-step)/2
		}
		for step < stop {
			step++
			d.trained++
			loss = stepLoss(loss, step)
			if step%100 == 0 || step == until {
				d.loss[step], d.latest = loss, step
			}
		}
		if stop < until {
			return nil, errors.New("killed mid-segment (exit -9)")
		}
		return map[string]any{"outputs": []string{fmt.Sprintf("runs/%s/ckpt/step-%06d", spec["name"], until)}}, nil
	})
	s.register(workflows.ActivityEval, func(c call) (any, error) {
		var spec struct {
			Suites   []string
			Subjects []workflows.EvalSubject
		}
		json.Unmarshal(c.Input, &spec)
		suite := spec.Suites[0]
		if suite == d.evalFails {
			return nil, &workflows.StepFailure{Type: "ExitCode65", Message: "unknown suite", NonRetryable: true}
		}
		d.evals[spec.Subjects[0].Model+"/"+suite]++
		return map[string]any{"outputs": []string{fmt.Sprintf("evals/%s/%s/summary.json", suite, strings.ReplaceAll(spec.Subjects[0].ID, "@", "-"))}}, nil
	})
}

func runTrain(t *testing.T, s *sim, in workflows.TrainRunInput) (workflows.TrainRunResult, error) {
	var res workflows.TrainRunResult
	err := s.runToEnd(func(rt workflows.Runtime) error {
		var err error
		res, err = workflows.TrainRun(rt, in)
		return err
	})
	return res, err
}

func uninterrupted(t *testing.T) (workflows.TrainRunResult, *runDir) {
	s, d := newSim(t, "train/tiny"), newRunDir()
	fakeTraining(s, d)
	res, err := runTrain(t, s, workflows.TrainRunInput{Spec: json.RawMessage(handSpec), EvalSuites: []string{"ppl"}})
	if err != nil {
		t.Fatal(err)
	}
	return res, d
}

func TestHandExampleSegmentsAndEvals(t *testing.T) {
	// WHY: the chapter's worked example (section 3): steps 1200 with
	//      eval_every 500 is three train segments, to 500, 1000, and 1200,
	//      each followed by one ppl evaluation of the checkpoint it wrote,
	//      with stable activity ids (train-000500, eval-000500-ppl, ...).
	// KIND: unit
	// CATCHES: s01, s02, s03
	// CHAPTER: dur.11 section 3, worked example
	if got := workflows.Segments(1200, 500); !reflect.DeepEqual(got, []int{500, 1000, 1200}) {
		t.Fatalf("Segments(1200, 500) = %v", got)
	}
	res, d := uninterrupted(t)
	var ids []string
	s := newSim(t, "train/tiny")
	fakeTraining(s, newRunDir())
	runTrain(t, s, workflows.TrainRunInput{Spec: json.RawMessage(handSpec), EvalSuites: []string{"ppl"}})
	for _, h := range s.hist {
		ids = append(ids, h.ID)
	}
	if got := strings.Join(ids, " "); got != "train-000500 eval-000500-ppl train-001000 eval-001000-ppl train-001200 eval-001200-ppl" {
		t.Fatalf("activity ids: %s", got)
	}
	if res.Checkpoint != "runs/tiny/ckpt/step-001200" || len(res.Segments) != 3 || res.Segments[1].Until != 1000 {
		t.Fatalf("result %+v", res)
	}
	if d.trained != 1200 {
		t.Fatalf("%d optimizer steps for a 1200-step run", d.trained)
	}
	for _, until := range []int{500, 1000, 1200} {
		if n := d.evals[fmt.Sprintf("runs/tiny/ckpt/step-%06d/ppl", until)]; n != 1 {
			t.Fatalf("checkpoint %d evaluated %d times", until, n)
		}
	}
}

func TestEvalIntervalUsesDurableSleepAndReplaysOnce(t *testing.T) {
	// WHY: evaluation cadence is workflow time, not a worker sleep. A crash
	//      during that wait must replay the recorded timer once and preserve
	//      the same gap before the next training segment.
	// KIND: fault
	// CATCHES: s13
	// CHAPTER: dur.11 section 2.1
	s := newSim(t, "train/timed")
	d := newRunDir()
	fakeTraining(s, d)
	in := workflows.TrainRunInput{
		Spec:       json.RawMessage(`{"name":"tiny","steps":1200,"eval_every":500,"eval_interval_seconds":30,"seed":7}`),
		EvalSuites: []string{"ppl"},
	}
	want, err := runTrain(t, s, in)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.hist) != 8 {
		t.Fatalf("history has %d steps, want 3 train, 3 eval, and 2 timer waits", len(s.hist))
	}
	for i, wantKind := range []string{"activity", "activity", "sleep", "activity", "activity", "sleep", "activity", "activity"} {
		if s.hist[i].Kind != wantKind {
			t.Fatalf("history[%d] kind = %q, want %q", i, s.hist[i].Kind, wantKind)
		}
		if wantKind == "sleep" && s.hist[i].ID != "30s" {
			t.Fatalf("history[%d] timer = %q, want 30s", i, s.hist[i].ID)
		}
	}
	evalCount := len(d.evals)
	got, err := runTrain(t, s, in)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("replay result = (%+v, %v), want (%+v, nil)", got, err, want)
	}
	if len(s.hist) != 8 || len(d.evals) != evalCount {
		t.Fatalf("replay appended history or repeated evaluations: history=%d evals=%d", len(s.hist), len(d.evals))
	}
}

func TestSegmentsMath(t *testing.T) {
	// WHY: segment boundaries decide when evaluations happen; they must end
	//      exactly at steps, never repeat or skip a boundary, and eval_every
	//      0 (or not less than steps) is one segment.
	// KIND: boundary
	// CATCHES: s01, s04
	// CHAPTER: dur.11 section 2.1
	cases := []struct {
		steps, every int
		want         []int
	}{
		{1000, 500, []int{500, 1000}},
		{1001, 500, []int{500, 1000, 1001}},
		{100, 0, []int{100}},
		{100, 100, []int{100}},
		{100, 250, []int{100}},
		{3, 1, []int{1, 2, 3}},
	}
	for _, c := range cases {
		if got := workflows.Segments(c.steps, c.every); !reflect.DeepEqual(got, c.want) {
			t.Fatalf("Segments(%d, %d) = %v, want %v", c.steps, c.every, got, c.want)
		}
	}
}

func TestEachSegmentGetsTheSpecWithItsEnd(t *testing.T) {
	// WHY: a segment's activity input is the run's own spec with steps
	//      replaced by the segment's end and nothing else changed: the model,
	//      the seed, and the data are what make the run reproducible.
	// KIND: unit
	// CATCHES: s05
	// CHAPTER: dur.11 section 4, The interface
	_, d := uninterrupted(t)
	if len(d.specs) != 3 {
		t.Fatalf("%d train activities", len(d.specs))
	}
	for i, until := range []float64{500, 1000, 1200} {
		sp := d.specs[i]
		if sp["steps"].(float64) != until || sp["seed"].(float64) != 7 || sp["name"] != "tiny" || sp["eval_every"].(float64) != 500 {
			t.Fatalf("segment %d spec %v", i, sp)
		}
		if sp["model"].(map[string]any)["tl_arch"] != "bigram" {
			t.Fatalf("segment %d lost the model config: %v", i, sp)
		}
	}
}

func TestKillLoopResumesBitwise(t *testing.T) {
	// WHY: the promise of TrainRun: kill the worker (and replay the
	//      workflow) at any point, in the middle of a segment or between
	//      steps, and the run ends with the same checkpoint and exactly the
	//      same final loss as an uninterrupted run; finished segments and
	//      evaluations are replayed, never redone.
	// KIND: fault
	// CATCHES: s02
	// CHAPTER: dur.11 section 2.2
	want, wantDir := uninterrupted(t)
	finalLoss := wantDir.loss[1200]
	for _, mode := range []string{"during", "between"} {
		for at := 0; at <= 5; at++ {
			s, d := newSim(t, "train/tiny"), newRunDir()
			fakeTraining(s, d)
			s.crashAt, s.crashMode = at, mode
			got, err := runTrain(t, s, workflows.TrainRunInput{Spec: json.RawMessage(handSpec), EvalSuites: []string{"ppl"}})
			if err != nil {
				t.Fatalf("%s@%d: %v", mode, at, err)
			}
			if !reflect.DeepEqual(got, want) || d.loss[1200] != finalLoss {
				t.Fatalf("%s@%d: final loss %v (want %v) or result differs", mode, at, d.loss[1200], finalLoss)
			}
			for k, n := range d.evals {
				if n > 1 && mode == "between" {
					t.Fatalf("%s@%d: %s evaluated %d times", mode, at, k, n)
				}
			}
		}
	}
}

func TestRetryResumesFromLatest(t *testing.T) {
	// WHY: a train attempt that dies halfway leaves checkpoints; the retry
	//      (the same activity, attempt 2) must continue from the newest one,
	//      not start the segment over: optimizer steps are the expensive
	//      part, and redoing them changes nothing but the bill.
	// KIND: fault
	// CATCHES: s06
	// CHAPTER: dur.11 section 2.2
	s, d := newSim(t, "train/tiny"), newRunDir()
	fakeTraining(s, d)
	d.crashMid = true
	res, err := runTrain(t, s, workflows.TrainRunInput{Spec: json.RawMessage(`{"name":"tiny","steps":1000,"seed":7}`)})
	if err != nil {
		t.Fatal(err)
	}
	if d.trained != 1000 || res.Checkpoint != "runs/tiny/ckpt/step-001000" {
		t.Fatalf("trained %d steps after a mid-segment death (want 1000: resume from step 500); result %+v", d.trained, res)
	}
	if s.executed["train/tiny/train-001000"] != 2 {
		t.Fatalf("the failed attempt is retried as the same activity: %v", s.executed)
	}
}

func TestCancelStopsTheRun(t *testing.T) {
	// WHY: a cancelled TrainRun ends canceled after the activity in flight
	//      (which checkpoints and exits 130, dur.09); it starts no further
	//      segment or evaluation, and deletes nothing: the checkpoints are
	//      the work done.
	// KIND: fault
	// CATCHES: s07
	// CHAPTER: dur.11 section 2.3
	s, d := newSim(t, "train/tiny"), newRunDir()
	fakeTraining(s, d)
	s.cancelAt = 2 // after the first segment and its evaluation
	_, err := runTrain(t, s, workflows.TrainRunInput{Spec: json.RawMessage(handSpec), EvalSuites: []string{"ppl"}})
	if !workflows.IsCanceled(err) {
		t.Fatalf("err %v", err)
	}
	if d.latest != 500 || len(s.order) != 2 {
		t.Fatalf("after the cancel: latest %d, activities %v", d.latest, s.order)
	}
}

func TestCorpusFirst(t *testing.T) {
	// WHY: a TrainRun given a corpus config builds it first, inside the same
	//      run (CorpusBuild, data.09), so training never starts on a missing
	//      or half-built token stream, and a failed build fails the run
	//      before any training step.
	// KIND: unit
	// CATCHES: s08
	// CHAPTER: dur.11 section 2.1
	cfg := `{"dataset":"tinystories","version":"v1","tokenizer":{"id":"bytes"}}`
	s, d := newSim(t, "train/tiny"), newRunDir()
	fakeTraining(s, d)
	stages := 0
	s.register(workflows.ActCorpusStage, func(c call) (any, error) {
		stages++
		var in workflows.StageInput
		json.Unmarshal(c.Input, &in)
		return map[string]any{"outputs": []string{"corpus/tinystories/v1/_MANIFEST.json"}}, nil
	})
	res, err := runTrain(t, s, workflows.TrainRunInput{Spec: json.RawMessage(`{"name":"tiny","steps":100,"seed":1}`), Corpus: json.RawMessage(cfg)})
	if err != nil || stages != 3 || res.Corpus == nil || res.Corpus.Manifest == "" {
		t.Fatalf("err %v, stages %d, corpus %+v", err, stages, res.Corpus)
	}
	if got := strings.Join(s.order, " "); got != "corpus.stage corpus.stage corpus.stage train" {
		t.Fatalf("order: %s", got)
	}
	s2, d2 := newSim(t, "train/tiny"), newRunDir()
	fakeTraining(s2, d2)
	s2.register(workflows.ActCorpusStage, func(c call) (any, error) {
		return nil, &workflows.StepFailure{Type: "ExitCode65", Message: "unlicensed", NonRetryable: true}
	})
	s2.register(workflows.ActCorpusCleanup, func(c call) (any, error) { return nil, nil })
	if _, err := runTrain(t, s2, workflows.TrainRunInput{Spec: json.RawMessage(`{"name":"tiny","steps":100,"seed":1}`), Corpus: json.RawMessage(cfg)}); err == nil || d2.trained != 0 {
		t.Fatalf("a failed corpus build: err %v, trained %d", err, d2.trained)
	}
}

func TestBadSpecFailsFast(t *testing.T) {
	// WHY: a spec with no name, no steps, or a negative cadence can never
	//      train; the run fails non-retryably before any activity.
	// KIND: boundary
	// CATCHES: s09
	// CHAPTER: dur.11 section 5, Pitfalls
	for _, sp := range []string{`{"steps":10}`, `{"name":"x","steps":0}`, `{"name":"x","steps":10,"eval_every":-1}`, `{"name":"x","steps":10,"eval_interval_seconds":-1}`, `[]`} {
		s, d := newSim(t, "train/x"), newRunDir()
		fakeTraining(s, d)
		_, err := runTrain(t, s, workflows.TrainRunInput{Spec: json.RawMessage(sp)})
		var nr interface{ NonRetryable() bool }
		if !errors.As(err, &nr) || !nr.NonRetryable() || len(s.order) != 0 {
			t.Fatalf("spec %s: err %v, activities %v", sp, err, s.order)
		}
	}
}

func TestEvalSuiteOneActivityPerSuite(t *testing.T) {
	// WHY: one activity per suite, in order, each with its own id: a crash
	//      during the second suite replays the first instead of rerunning it,
	//      and the first suite that fails for good fails the whole step.
	// KIND: unit
	// CATCHES: s10, s11
	// CHAPTER: dur.11 section 2.4
	in := workflows.EvalSuiteInput{Suites: []string{"ppl", "zoo", "safety"}, Subjects: []workflows.EvalSubject{{ID: "m", Model: "models/m/v1"}}, Seed: 3}
	s, d := newSim(t, "eval/m"), newRunDir()
	fakeTraining(s, d)
	s.crashAt, s.crashMode = 1, "between"
	var res workflows.EvalSuiteResult
	err := s.runToEnd(func(rt workflows.Runtime) error {
		var err error
		res, err = workflows.EvalSuite(rt, in)
		return err
	})
	if err != nil || len(res.Suites) != 3 || res.Suites[2].Suite != "safety" {
		t.Fatalf("err %v res %+v", err, res)
	}
	for _, suite := range in.Suites {
		if d.evals["models/m/v1/"+suite] != 1 || s.executed["eval/m/eval-"+suite] != 1 {
			t.Fatalf("suite %s: evaluations %v, executions %v", suite, d.evals, s.executed)
		}
	}
	spec, _ := workflows.MarshalSpec(in, "zoo")
	if string(spec) != `{"name":"evalsuite","suites":["zoo"],"subjects":[{"id":"m","model":"models/m/v1"}],"seed":3}` {
		t.Fatalf("eval spec %s", spec)
	}
	s2, d2 := newSim(t, "eval/m"), newRunDir()
	fakeTraining(s2, d2)
	d2.evalFails = "zoo"
	err = s2.runToEnd(func(rt workflows.Runtime) error { _, err := workflows.EvalSuite(rt, in); return err })
	var f *workflows.StepFailure
	if !errors.As(err, &f) || d2.evals["models/m/v1/safety"] != 0 {
		t.Fatalf("a failed suite: err %v, later suites %v", err, d2.evals)
	}
}

func TestOptionsIDs(t *testing.T) {
	// WHY: activity ids are idempotency keys: they must name the segment
	//      and the suite, so two segments or two suites never share a work
	//      directory, and a heartbeat timeout must be set so a dead trainer
	//      is noticed in a minute, not after the day-long start-to-close.
	// KIND: unit
	// CATCHES: s03, s12
	// CHAPTER: dur.11 section 4, The interface
	if o := workflows.TrainOptions(500); o.ID != "train-000500" || o.HeartbeatTimeout <= 0 || o.HeartbeatTimeout >= o.StartToClose {
		t.Fatalf("TrainOptions(500) = %+v", o)
	}
	if workflows.EvalOptions("000500", "ppl").ID != "eval-000500-ppl" || workflows.EvalOptions("", "zoo").ID != "eval-zoo" {
		t.Fatal("EvalOptions ids")
	}
}
