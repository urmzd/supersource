// Course tests for data.09: CorpusBuild (go/workflows/corpus_build.go),
// driven by the replaying simulator in sim_test.go. The stage activities are
// fakes that keep the subprocess contract's rules (DONE.json makes a rerun a
// no-op, outputs appear whole) over an in-memory disk, so a kill at any point
// can be checked against an uninterrupted build without running Python.
package data_09

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"

	"tinyllm/workflows"
)

// The chapter's worked example config, and its canonical form's sha256
// (computed by hand: keys sorted, no whitespace).
const handConfig = `{
  "version": "v1",
  "dataset": "tinystories",
  "tokenizer": {"id": "bytes"}
}`

const handSHA = "a03711ec959b2a8fb6cf51d9445bee90fcb69a103cfa033483b52f836269c0a6"

// disk is the fake /artifacts: path -> content, plus the effect log of
// every write that changed a file (the external effects of the build).
type disk struct {
	files   map[string]string
	done    map[string]bool // idempotency keys with a DONE.json
	writes  map[string]int  // path -> times its content was written
	cleaned int
}

func newDisk() *disk {
	return &disk{files: map[string]string{}, done: map[string]bool{}, writes: map[string]int{}}
}

func (d *disk) write(path, content string) {
	if d.files[path] != content {
		d.writes[path]++
	}
	d.files[path] = content
}

func (d *disk) under(prefix string) []string {
	var out []string
	for p := range d.files {
		if strings.HasPrefix(p, prefix) {
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return out
}

// fakeCorpus registers the stage and cleanup activities on s over d. A
// stage named in fail exits 65 (non-retryable) the first time it runs.
func fakeCorpus(s *sim, d *disk, fail string) {
	s.register(workflows.ActCorpusStage, func(c call) (any, error) {
		var in workflows.StageInput
		if err := json.Unmarshal(c.Input, &in); err != nil {
			return nil, err
		}
		var cfg struct{ Dataset, Version string }
		json.Unmarshal(in.Config, &cfg)
		if in.Stage == fail {
			return nil, &workflows.StepFailure{Type: "ExitCode65", Message: "source wiki is not licensed for train", NonRetryable: true}
		}
		if d.done[c.Key] {
			return d.outputs(in.Stage, cfg.Dataset, cfg.Version), nil // a rerun after success: no-op
		}
		base := fmt.Sprintf("corpus/%s/%s/", cfg.Dataset, cfg.Version)
		switch in.Stage {
		case "fetch":
			d.write("corpus/raw/books/20260101/0.jsonl.zst", "raw-books")
			d.write("corpus/LEDGER.jsonl", "ledger")
		case "shard":
			for i := 0; i < 3; i++ {
				d.write(fmt.Sprintf("%sshard-%05d-of-00003.parquet", base, i), fmt.Sprintf("rows-%d-%s", i, in.ConfigSHA256[:8]))
			}
			d.write(base+"_MANIFEST.json", "manifest-"+in.ConfigSHA256[:8])
		case "tokenize":
			d.write("tokens/bytes/"+cfg.Dataset+"/train-00000.bin", "train-ids")
			d.write("tokens/bytes/"+cfg.Dataset+"/val-00000.bin", "val-ids")
		}
		d.done[c.Key] = true
		return d.outputs(in.Stage, cfg.Dataset, cfg.Version), nil
	})
	s.register(workflows.ActCorpusCleanup, func(c call) (any, error) {
		var in workflows.CleanupInput
		json.Unmarshal(c.Input, &in)
		for _, p := range append(d.under(fmt.Sprintf("corpus/%s/%s/", in.Dataset, in.Version)), d.under(fmt.Sprintf("tokens/%s/%s/", in.TokenizerID, in.Dataset))...) {
			delete(d.files, p)
		}
		d.cleaned++
		return map[string]any{"outputs": []string{}}, nil
	})
}

func (d *disk) outputs(stage, dataset, version string) map[string]any {
	switch stage {
	case "fetch":
		return map[string]any{"outputs": []string{"corpus/raw", "corpus/LEDGER.jsonl"}}
	case "shard":
		return map[string]any{"outputs": []string{fmt.Sprintf("corpus/%s/%s/_MANIFEST.json", dataset, version)}}
	default:
		return map[string]any{"outputs": []string{"tokens/bytes/" + dataset}}
	}
}

func build(cfg string) func(rt workflows.Runtime) error {
	return func(rt workflows.Runtime) error {
		_, err := workflows.CorpusBuild(rt, workflows.CorpusBuildInput{Config: json.RawMessage(cfg)})
		return err
	}
}

func buildResult(t *testing.T, s *sim, cfg string) (workflows.CorpusBuildResult, error) {
	var res workflows.CorpusBuildResult
	err := s.runToEnd(func(rt workflows.Runtime) error {
		var err error
		res, err = workflows.CorpusBuild(rt, workflows.CorpusBuildInput{Config: json.RawMessage(cfg)})
		return err
	})
	return res, err
}

func TestHandExampleKeysAndOrder(t *testing.T) {
	// WHY: the chapter's worked example (section 3). The config's canonical
	//      form hashes to a03711ec...; the workflow id is
	//      corpus/tinystories/v1/a03711ec959b; the three stages run in order
	//      with the stage name as activity id, so the keys are
	//      <workflow id>/fetch, /shard, /tokenize; the result names the
	//      manifest and the token directory.
	// KIND: unit
	// CATCHES: s01, s02, s03
	// CHAPTER: data.09 section 3, worked example
	sha, err := workflows.ConfigSHA256(json.RawMessage(handConfig))
	if err != nil || sha != handSHA {
		t.Fatalf("ConfigSHA256 = %q, %v; want %s", sha, err, handSHA)
	}
	id, err := workflows.CorpusWorkflowID(json.RawMessage(handConfig))
	if err != nil || id != "corpus/tinystories/v1/a03711ec959b" {
		t.Fatalf("CorpusWorkflowID = %q, %v", id, err)
	}
	s, d := newSim(t, id), newDisk()
	fakeCorpus(s, d, "")
	res, err := buildResult(t, s, handConfig)
	if err != nil {
		t.Fatal(err)
	}
	keys := make([]string, 0, len(s.executed))
	for k := range s.executed {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if got := strings.Join(keys, " "); got != id+"/fetch "+id+"/shard "+id+"/tokenize" {
		t.Fatalf("idempotency keys: %s", got)
	}
	if got := strings.Join(s.order, " "); got != "corpus.stage corpus.stage corpus.stage" {
		t.Fatalf("activities: %s (no cleanup on success)", got)
	}
	var stages []string
	for _, h := range s.hist {
		stages = append(stages, h.ID)
	}
	if strings.Join(stages, " ") != "fetch shard tokenize" {
		t.Fatalf("stage order %v", stages)
	}
	if res.Manifest != "corpus/tinystories/v1/_MANIFEST.json" || !reflect.DeepEqual(res.Tokens, []string{"tokens/bytes/tinystories"}) || res.ConfigSHA256 != handSHA {
		t.Fatalf("result %+v", res)
	}
}

func TestConfigHashIsContentNotSpelling(t *testing.T) {
	// WHY: the same config written with other whitespace or key order is the
	//      same build (one workflow id, so StartWorkflow finds the run that
	//      exists); any changed value is a different build.
	// KIND: property
	// CATCHES: s03, s04
	// CHAPTER: data.09 section 2.2
	same := []string{
		`{"dataset":"tinystories","version":"v1","tokenizer":{"id":"bytes"}}`,
		"{\n\t\"tokenizer\": {\"id\": \"bytes\"},\n\t\"version\": \"v1\", \"dataset\": \"tinystories\"}",
	}
	for _, c := range same {
		if id, err := workflows.CorpusWorkflowID(json.RawMessage(c)); err != nil || id != "corpus/tinystories/v1/a03711ec959b" {
			t.Fatalf("%q: %q %v", c, id, err)
		}
	}
	other, _ := workflows.CorpusWorkflowID(json.RawMessage(`{"dataset":"tinystories","version":"v1","tokenizer":{"id":"gpt2"}}`))
	if other == "corpus/tinystories/v1/a03711ec959b" || !strings.HasPrefix(other, "corpus/tinystories/v1/") {
		t.Fatalf("a different tokenizer must give a different id: %q", other)
	}
}

func TestBadConfigFailsBeforeAnyActivity(t *testing.T) {
	// WHY: a config with no dataset, version, or tokenizer id can never
	//      succeed; it fails the workflow at once (non-retryable) instead of
	//      fetching gigabytes first.
	// KIND: boundary
	// CATCHES: s05
	// CHAPTER: data.09 section 5, Pitfalls
	for _, c := range []string{`{"dataset":"d","version":"v1"}`, `{"version":"v1","tokenizer":{"id":"b"}}`, `[1]`, `{"dataset":"d","tokenizer":{"id":"b"}}`} {
		s, d := newSim(t, "corpus/x"), newDisk()
		fakeCorpus(s, d, "")
		_, err := buildResult(t, s, c)
		var nr interface{ NonRetryable() bool }
		if err == nil || !errors.As(err, &nr) || !nr.NonRetryable() || len(s.order) != 0 {
			t.Fatalf("config %s: err %v, activities %v", c, err, s.order)
		}
	}
}

func TestKillLoopSameResult(t *testing.T) {
	// WHY: the promise of a durable build: the worker can die at any point,
	//      with or without the running stage's effects, and the final result
	//      and files equal an uninterrupted build's, each output written once
	//      (a rerun of a finished stage is a no-op through DONE.json), and no
	//      stage that finished is run again from history.
	// KIND: fault
	// CHAPTER: data.09 section 2.3
	id, _ := workflows.CorpusWorkflowID(json.RawMessage(handConfig))
	clean, cleanDisk := newSim(t, id), newDisk()
	fakeCorpus(clean, cleanDisk, "")
	want, err := buildResult(t, clean, handConfig)
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"during", "between"} {
		for at := 0; at <= 3; at++ {
			s, d := newSim(t, id), newDisk()
			fakeCorpus(s, d, "")
			s.crashAt, s.crashMode = at, mode
			got, err := buildResult(t, s, handConfig)
			if err != nil {
				t.Fatalf("%s@%d: %v", mode, at, err)
			}
			if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(d.files, cleanDisk.files) {
				t.Fatalf("%s@%d: result or files differ from the uninterrupted build", mode, at)
			}
			for p, n := range d.writes {
				if n != 1 {
					t.Fatalf("%s@%d: %s written %d times", mode, at, p, n)
				}
			}
			if d.cleaned != 0 {
				t.Fatalf("%s@%d: a crash is not a cancel: the cleanup ran", mode, at)
			}
		}
	}
}

func TestCancelDeletesPartialShards(t *testing.T) {
	// WHY: cancel mid-build runs the compensation once: the shards and
	//      tokens of this dataset and version are deleted (a half-built
	//      corpus that looks complete is worse than none), the fetched raw
	//      documents stay (a cache other builds share), and the workflow ends
	//      canceled. The compensation is registered before the shard stage,
	//      so a cancel that interrupts it cleans up too.
	// KIND: fault
	// CATCHES: s07, s08
	// CHAPTER: data.09 section 2.4
	id, _ := workflows.CorpusWorkflowID(json.RawMessage(handConfig))
	for _, at := range []int{1, 2} { // during shard, during tokenize
		s, d := newSim(t, id), newDisk()
		fakeCorpus(s, d, "")
		if at == 1 {
			d.write("corpus/tinystories/v1/shard-00000-of-00003.parquet", "half") // the interrupted stage left a file
		}
		s.cancelAt = at
		_, err := buildResult(t, s, handConfig)
		if !workflows.IsCanceled(err) {
			t.Fatalf("cancel at %d: %v", at, err)
		}
		if d.cleaned != 1 {
			t.Fatalf("cancel at %d: cleanup ran %d times", at, d.cleaned)
		}
		if left := d.under("corpus/tinystories/"); len(left) != 0 {
			t.Fatalf("cancel at %d: partial outputs left: %v", at, left)
		}
		if len(d.under("corpus/raw/")) == 0 {
			t.Fatalf("cancel at %d: the raw cache was deleted", at)
		}
	}
}

func TestFailedStageCompensates(t *testing.T) {
	// WHY: a stage that fails for good (an unlicensed source exits 65) must
	//      not leave the shards it wrote looking like a finished corpus; the
	//      workflow fails with the stage's failure, not a cancel.
	// KIND: fault
	// CATCHES: s09
	// CHAPTER: data.09 section 2.4
	s, d := newSim(t, "corpus/x"), newDisk()
	fakeCorpus(s, d, "tokenize")
	_, err := buildResult(t, s, handConfig)
	var f *workflows.StepFailure
	if !errors.As(err, &f) || f.Type != "ExitCode65" || workflows.IsCanceled(err) {
		t.Fatalf("err %v", err)
	}
	if d.cleaned != 1 || len(d.under("corpus/tinystories/")) != 0 {
		t.Fatalf("cleanup %d, left %v", d.cleaned, d.under("corpus/tinystories/"))
	}
}

func TestFetchFailureNeedsNoCleanup(t *testing.T) {
	// WHY: when fetch fails nothing of this build exists yet, so there is
	//      nothing to compensate: the cleanup activity must not run (it would
	//      delete a previous good build of the same dataset and version).
	// KIND: boundary
	// CATCHES: s10
	// CHAPTER: data.09 section 5, Pitfalls
	s, d := newSim(t, "corpus/x"), newDisk()
	d.write("corpus/tinystories/v1/_MANIFEST.json", "an earlier good build")
	fakeCorpus(s, d, "fetch")
	if _, err := buildResult(t, s, handConfig); err == nil {
		t.Fatal("fetch failed: the build must fail")
	}
	if d.cleaned != 0 || len(d.under("corpus/tinystories/v1/")) != 1 {
		t.Fatalf("a failed fetch ran the cleanup (%d) or deleted files", d.cleaned)
	}
}

func TestStageOptions(t *testing.T) {
	// WHY: the options are part of the contract with the server: the stage
	//      name as activity id (the key), a heartbeat timeout so a dead worker
	//      is noticed in minutes, not after the start-to-close of hours, and
	//      more attempts for the network-bound fetch.
	// KIND: unit
	// CATCHES: s11
	// CHAPTER: data.09 section 4, The interface
	for _, st := range workflows.CorpusStages {
		o := workflows.StageOptions(st)
		if o.ID != st || o.HeartbeatTimeout <= 0 || o.HeartbeatTimeout >= o.StartToClose || o.MaxAttempts < 2 {
			t.Fatalf("StageOptions(%s) = %+v", st, o)
		}
	}
	if workflows.StageOptions("fetch").MaxAttempts <= workflows.StageOptions("shard").MaxAttempts {
		t.Fatal("fetch retries more than the local stages")
	}
	if !reflect.DeepEqual(workflows.CorpusStages, []string{"fetch", "shard", "tokenize"}) {
		t.Fatalf("CorpusStages = %v", workflows.CorpusStages)
	}
}
