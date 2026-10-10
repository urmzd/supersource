package dur_12

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"tinyllm/workflows"
)

func TestHandExampleCanaryPromotes(t *testing.T) {
	// WHY: the chapter's worked example (section 3). The route tinystories
	//      sends everything to tinystories-10m-v1 (no backends field). After
	//      approve, the canary of weight 0.1 makes it v1 0.9 and v2 0.1; the
	//      timer bakes 600 s; the burn query at t0 + 600 s answers 0.8, under
	//      the ceiling 14.4, so the route becomes v2 1.0. Two PUTs in all, and
	//      the steps run in the order of section 2.
	// KIND: unit
	// CATCHES: s08, s11, s12, s13, s33, s39
	// CHAPTER: dur.12 section 3, worked example
	s := newScenario(t, vector("0.8"))
	s.w.signal("approve", `{"by":"alice"}`)
	var mid map[string]float64
	s.w.impls[workflows.ActQuery] = wrapAfter(s.w.impls[workflows.ActQuery], func() {
		mid = backendsOf(t, s.gw.route(t, "tinystories"))
	})
	res, err, _ := s.w.run(s.spec)
	if err != nil {
		t.Fatalf("release failed: %v", err)
	}
	if res.Outcome != workflows.OutcomePromoted {
		t.Fatalf("outcome %q (%s), want promoted", res.Outcome, res.Reason)
	}
	want := "release.preflight export eval eval release.results signal:approve routes.snapshot routes.canary promql.query routes.promote"
	if got := joined(s.w.order); got != want {
		t.Fatalf("steps ran as\n  %s\nwant\n  %s", got, want)
	}
	ids := "release-preflight release-export eval-release-quality eval-release-safety release-results routes-snapshot routes-canary promql-burn routes-promote"
	if got := joined(s.w.ids); got != ids {
		t.Fatalf("activity ids (idempotency keys)\n  %s\nwant\n  %s", got, ids)
	}
	if len(mid) != 2 || !approx(mid["tinystories-10m-v1"], 0.9) || !approx(mid["tinystories-10m-v2"], 0.1) {
		t.Fatalf("during the bake the route was %v, want v1 0.9 and v2 0.1", mid)
	}
	if b := backendsOf(t, s.gw.route(t, "tinystories")); len(b) != 1 || b["tinystories-10m-v2"] != 1 {
		t.Fatalf("after promotion the route is %v, want only tinystories-10m-v2 at 1", b)
	}
	if s.gw.puts != 2 {
		t.Fatalf("%d PUTs, want 2 (canary, promote)", s.gw.puts)
	}
	if q := s.prom.queries; len(q) != 1 || q[0]["query"] != burnQuery || q[0]["time"] != "1791547800.000" {
		t.Fatalf("burn queries %v, want one at t0 + 600 s (1791547800.000)", q)
	}
	if res.Burn == nil || *res.Burn != 0.8 || res.ModelDir != "models/tinystories-10m/v2" || res.Served != "tinystories-10m-v2" {
		t.Fatalf("result %+v", res)
	}
	if string(res.Approval) != `{"by":"alice"}` {
		t.Fatalf("approval payload %q not kept", res.Approval)
	}
	var ev struct {
		Suites   []string            `json:"suites"`
		Subjects []map[string]string `json:"subjects"`
	}
	if err := json.Unmarshal(s.w.inputs[workflows.ActivityEval], &ev); err != nil || len(ev.Suites) != 1 || len(ev.Subjects) != 1 ||
		ev.Subjects[0]["id"] != "tinystories-10m" || ev.Subjects[0]["model"] != "models/tinystories-10m/v2" {
		t.Fatalf("eval input %s: want one suite and one subject, the exported model directory", s.w.inputs[workflows.ActivityEval])
	}
}

func TestFastBurnRollsBackToTheSnapshot(t *testing.T) {
	// WHY: the canary burns the error budget 20 times too fast: the route
	//      must go back to exactly what it was before the canary (the
	//      snapshot), not to "v1 at weight 1" rebuilt by hand, and the other
	//      route must not move.
	// KIND: fault
	// CATCHES: s12, s20, s21, s37
	// CHAPTER: dur.12 section 5, Pitfalls
	s := newScenario(t, vector("20.5"))
	before := s.gw.rawRoute(t, "tinystories")
	other := s.gw.rawRoute(t, "smart")
	s.w.signal("approve", "{}")
	res, err, _ := s.w.run(s.spec)
	if err != nil {
		t.Fatalf("a rollback is a completed release, not an error: %v", err)
	}
	if res.Outcome != workflows.OutcomeRolledBack || res.Burn == nil || *res.Burn != 20.5 {
		t.Fatalf("result %+v, want rolled_back with burn 20.5", res)
	}
	if !jsonEqual(t, s.gw.rawRoute(t, "tinystories"), before) {
		t.Fatalf("route after rollback\n  %s\nwant the snapshot\n  %s", s.gw.rawRoute(t, "tinystories"), before)
	}
	if s.gw.rawRoute(t, "smart") != other {
		t.Fatal("the rollback changed another route")
	}
	if got := joined(s.w.order); !strings.HasSuffix(got, "promql.query routes.restore") {
		t.Fatalf("steps %s: want the burn query, then routes.restore", got)
	}
}

func TestBurnAtTheCeilingPromotes(t *testing.T) {
	// WHY: the ceiling is inclusive: a burn rate equal to max is within
	//      budget and promotes. A strict comparison rolls back a healthy
	//      canary.
	// KIND: boundary
	// CATCHES: s13, s17
	// CHAPTER: dur.12 section 2, the burn check
	s := newScenario(t, vector("14.4"))
	s.w.signal("approve", "{}")
	res, err, _ := s.w.run(s.spec)
	if err != nil || res.Outcome != workflows.OutcomePromoted {
		t.Fatalf("burn 14.4 with max 14.4: outcome %q err %v, want promoted", res.Outcome, err)
	}
	if got := s.w.now.Sub(t0); got != 600*time.Second {
		t.Fatalf("the release took %v on the durable clock, want the 600 s bake", got)
	}
}

func TestNoDataFailsClosed(t *testing.T) {
	// WHY: a canary that served no traffic proves nothing: an empty answer
	//      from Prometheus (or 0/0 = NaN) must roll back, never count as a
	//      burn rate of 0.
	// KIND: fault
	// CATCHES: s15, s29
	// CHAPTER: dur.12 section 5, Pitfalls
	for _, body := range []string{emptyVector, vector("NaN")} {
		s := newScenario(t, body)
		s.w.signal("approve", "{}")
		res, err, _ := s.w.run(s.spec)
		if err != nil || res.Outcome != workflows.OutcomeRolledBack || !res.BurnNoData {
			t.Fatalf("answer %s: outcome %q no_data %v err %v, want rolled_back with no data", body, res.Outcome, res.BurnNoData, err)
		}
	}
}

func TestFailedBurnQueryFailsClosed(t *testing.T) {
	// WHY: Prometheus rejects the expression (status error, 400): the check
	//      did not run, so the release must roll back rather than promote or
	//      hang.
	// KIND: fault
	// CATCHES: s16, s34
	// CHAPTER: dur.12 section 5, Pitfalls
	s := newScenario(t, `{"status":"error","errorType":"bad_data","error":"parse error: unexpected end of input"}`)
	s.prom.status = 400
	s.w.signal("approve", "{}")
	res, err, _ := s.w.run(s.spec)
	if err != nil || res.Outcome != workflows.OutcomeRolledBack {
		t.Fatalf("outcome %q err %v, want rolled_back", res.Outcome, err)
	}
	if n := s.w.execs[workflows.ActQuery]; n != 1 {
		t.Fatalf("a rejected expression was tried %d times; it is permanent, try once", n)
	}
}

func TestUnlicensedSourceFailsBeforeExport(t *testing.T) {
	// WHY: a ledger source whose allowed_uses lacks train (eval only) means
	//      the model may not ship. The gate runs before the export, fails
	//      the workflow non-retryably, and nothing else runs.
	// KIND: fault
	// CATCHES: s04, s07
	// CHAPTER: dur.12 section 5, Pitfalls
	s := newScenario(t, vector("0.1"))
	writeFile(t, s.spec.Ledger, goodLedger+strings.Replace(strings.Replace(goodLedger, `"tinystories"`, `"wiki-nc"`, 1),
		`["train","eval"]`, `["eval"]`, 1))
	_, err, _ := s.w.run(s.spec)
	var g *workflows.GateError
	if !errors.As(err, &g) || !strings.Contains(g.Reason, "wiki-nc") {
		t.Fatalf("error %v, want a GateError naming wiki-nc", err)
	}
	if s.w.execs[workflows.ActExport] != 0 || s.gw.puts != 0 {
		t.Fatal("the release exported or routed after the ledger gate failed")
	}
	if n := s.w.execs[workflows.ActPreflight]; n != 1 {
		t.Fatalf("the ledger gate ran %d times: a gate failure is non-retryable", n)
	}
}

func TestRevokedSourceFails(t *testing.T) {
	// WHY: a source revoked after training (the ops.08 data incident) must
	//      block every release of a model trained on it, even though its
	//      license allows training.
	// KIND: fault
	// CATCHES: s05, s07
	// CHAPTER: dur.12 section 2, the gates
	s := newScenario(t, vector("0.1"))
	writeFile(t, s.spec.Ledger, strings.Replace(goodLedger, `"notes":""`, `"notes":"","revoked":true`, 1))
	_, err, _ := s.w.run(s.spec)
	var g *workflows.GateError
	if !errors.As(err, &g) || !strings.Contains(g.Reason, "revoked") {
		t.Fatalf("error %v, want a GateError about the revoked source", err)
	}
}

func TestModelCardGate(t *testing.T) {
	// WHY: no model card, a card missing a template section, and a card
	//      that still holds template placeholders all fail before export.
	//      Placeholders inside HTML comments are fine, and so is a `<` that
	//      is a comparison (p < 0.05).
	// KIND: fault
	// CATCHES: s06
	// CHAPTER: dur.12 section 2, the gates
	cases := map[string]string{
		"missing":     "",
		"no section":  strings.Replace(goodCard, "## Data", "## Datasets", 1),
		"placeholder": strings.Replace(goodCard, "forge.", "<you>, as part of <system>.", 1),
	}
	for name, card := range cases {
		s := newScenario(t, vector("0.1"))
		if card == "" {
			s.spec.ModelCard = filepath.Join(s.dir, "NO_CARD.md")
		} else {
			writeFile(t, s.spec.ModelCard, card)
		}
		_, err, _ := s.w.run(s.spec)
		var g *workflows.GateError
		if !errors.As(err, &g) || s.w.execs[workflows.ActExport] != 0 {
			t.Fatalf("%s card: error %v, export ran %d times; want a GateError before export", name, err, s.w.execs[workflows.ActExport])
		}
	}
	if p := workflows.ModelCardProblems(goodCard); len(p) != 0 {
		t.Fatalf("a filled card is refused: %v", p)
	}
}

func TestMissingEvalRowFailsTheGate(t *testing.T) {
	// WHY: the safety suite did not produce its refusal row (it errored, or
	//      was not run). A gate with nothing to compare must fail: a missing
	//      safety row is not a passing one. No signal is consumed and no
	//      traffic moves.
	// KIND: fault
	// CATCHES: s01, s02
	// CHAPTER: dur.12 section 5, Pitfalls
	s := newScenario(t, vector("0.1"))
	s.rows = s.rows[:1] // quality only
	s.w.signal("approve", "{}")
	_, err, _ := s.w.run(s.spec)
	var g *workflows.GateError
	if !errors.As(err, &g) || g.Gate != "eval" {
		t.Fatalf("error %v, want the eval gate to refuse", err)
	}
	if s.gw.puts != 0 || len(s.w.signals) != 1 {
		t.Fatal("the release waited for approval or moved traffic after a failed gate")
	}
	res := workflows.CheckGates(s.spec.Gates, []workflows.EvalReport{{Suite: "safety", Rows: []workflows.EvalRow{
		{Model: "tinystories-10m", Task: "refusal", Metric: "score", Value: nil, Status: "error"}}}}, "tinystories-10m")
	if res[1].OK {
		t.Fatal("an errored safety row passed its gate")
	}
}

func TestGateDirections(t *testing.T) {
	// WHY: bpb is lower-is-better (<=), refusal is higher-is-better (>=).
	//      Values exactly at the threshold pass; just past it fail; a row of
	//      another model (the baseline in the same report) is not ours.
	// KIND: boundary
	// CATCHES: s02, s03
	// CHAPTER: dur.12 section 4, the interface
	gates := []workflows.Gate{{Suite: "quality", Task: "ts-val", Metric: "bpb", Op: "<=", Value: 1.30},
		{Suite: "safety", Task: "refusal", Metric: "score", Op: ">=", Value: 0.9}}
	rep := func(bpb, refusal float64) []workflows.EvalReport {
		return []workflows.EvalReport{
			{Suite: "quality", Rows: []workflows.EvalRow{
				{Model: "baseline", Task: "ts-val", Metric: "bpb", Value: f64(0.5), Status: "ok"},
				{Model: "tinystories-10m", Task: "ts-val", Metric: "bpb", Value: f64(bpb), Status: "ok"}}},
			{Suite: "safety", Rows: []workflows.EvalRow{
				{Model: "tinystories-10m", Task: "refusal", Metric: "score", Value: f64(refusal), Status: "ok"}}},
		}
	}
	if r := workflows.CheckGates(gates, nil, "tinystories-10m"); len(r) != 2 || r[0].OK || r[1].OK {
		t.Fatalf("gates with no reports at all: %+v, want both to fail", r)
	}
	for _, c := range []struct {
		bpb, refusal float64
		ok           [2]bool
	}{{1.30, 0.9, [2]bool{true, true}}, {1.3001, 0.95, [2]bool{false, true}}, {1.0, 0.8999, [2]bool{true, false}}} {
		got := workflows.CheckGates(gates, rep(c.bpb, c.refusal), "tinystories-10m")
		if len(got) != 2 || got[0].OK != c.ok[0] || got[1].OK != c.ok[1] {
			t.Fatalf("bpb %v refusal %v: gates %+v, want ok %v", c.bpb, c.refusal, got, c.ok)
		}
	}
}

func TestRejectStopsBeforeTraffic(t *testing.T) {
	// WHY: a human said no: the release ends as rejected and the route
	//      table is never read or written.
	// KIND: unit
	// CATCHES: s09, s11
	// CHAPTER: dur.12 section 2, approval
	s := newScenario(t, vector("0.1"))
	s.w.signal("reject", `{"by":"bob","why":"bias probe"}`)
	res, err, _ := s.w.run(s.spec)
	if err != nil || res.Outcome != workflows.OutcomeRejected || s.gw.puts != 0 || s.gw.gets != 0 {
		t.Fatalf("outcome %q err %v puts %d gets %d, want rejected with no route access", res.Outcome, err, s.gw.puts, s.gw.gets)
	}
}

func TestApprovalTimesOut(t *testing.T) {
	// WHY: nobody answered within approve_timeout_s: the release expires
	//      on the durable clock (one day later) without touching traffic,
	//      instead of holding a worker forever.
	// KIND: unit
	// CATCHES: s10, s11
	// CHAPTER: dur.12 section 2, approval
	s := newScenario(t, vector("0.1"))
	res, err, _ := s.w.run(s.spec)
	if err != nil || res.Outcome != workflows.OutcomeExpired || s.gw.puts != 0 {
		t.Fatalf("outcome %q err %v puts %d, want expired with no PUT", res.Outcome, err, s.gw.puts)
	}
	if got := s.w.now.Sub(t0); got != 24*time.Hour {
		t.Fatalf("the release expired after %v, want approve_timeout_s = 24h", got)
	}
}

func TestWorkerKilledMidCanary(t *testing.T) {
	// WHY: the worker dies right after its canary PUT, before the server
	//      recorded the activity as complete, and a second worker dies while
	//      the bake timer is pending. The canary activity runs again and must
	//      find its weight already in place (one canary PUT, not two), and the
	//      timer keeps the deadline it got when it started: the burn query runs
	//      at t0 + 2 min + 600 s, not 600 s after the second restart.
	// KIND: fault
	// CATCHES: s13, s22, s33
	// CHAPTER: dur.12 section 5, Pitfalls
	s := newScenario(t, vector("1.0"))
	s.w.signal("approve", "{}")
	s.w.downFor = 2 * time.Minute
	crashes := map[string]bool{}
	s.w.crash = func(kind, name string) bool {
		key := kind + ":" + name
		if (key == "activity:routes.canary" || key == "timer:") && !crashes[key] {
			crashes[key] = true
			return true
		}
		return false
	}
	res, err, restarts := s.w.run(s.spec)
	if err != nil || res.Outcome != workflows.OutcomePromoted {
		t.Fatalf("outcome %q err %v, want promoted after the crashes", res.Outcome, err)
	}
	if restarts != 2 || s.w.execs[workflows.ActCanary] != 2 {
		t.Fatalf("%d restarts, canary ran %d times; want 2 and 2", restarts, s.w.execs[workflows.ActCanary])
	}
	if s.gw.puts != 2 {
		t.Fatalf("%d PUTs, want 2: the retried canary must not apply its weight again", s.gw.puts)
	}
	// The timer started after the first restart (t0 + 2 min) and keeps
	// its deadline through the second crash: t0 + 2 min + 600 s.
	if q := s.prom.queries; len(q) != 1 || q[0]["time"] != "1791547920.000" {
		t.Fatalf("burn queries %v: the bake must end at t0 + 720 s (1791547920.000), not later", q)
	}
}

func TestCancelRestoresTheRoute(t *testing.T) {
	// WHY: an operator cancels the release. Before traffic moved (during the
	//      approval wait) the run just ends canceled; during the bake it must
	//      also put the snapshot back, through the saga's compensation run
	//      detached so the cancel cannot interrupt it, and never query or
	//      promote. A cancel that leaves the canary in place serves an
	//      unfinished release forever.
	// KIND: fault
	// CATCHES: s11, s12, s36, s39
	// CHAPTER: dur.12 section 2, cancellation
	s := newScenario(t, vector("0.1"))
	s.w.signal("approve", "{}")
	s.w.cancel = func(kind, _ string) bool { return kind == "signal" }
	if _, err, _ := s.w.run(s.spec); !workflows.IsCanceled(err) || s.gw.puts != 0 || s.gw.gets != 0 {
		t.Fatalf("cancel during the approval wait: err %v, %d PUTs, %d GETs; want ErrCanceled and no route access", err, s.gw.puts, s.gw.gets)
	}
	s = newScenario(t, vector("0.1"))
	before := s.gw.rawRoute(t, "tinystories")
	s.w.signal("approve", "{}")
	s.w.cancel = func(kind, _ string) bool { return kind == "timer" }
	_, err, _ := s.w.run(s.spec)
	if !workflows.IsCanceled(err) {
		t.Fatalf("error %v, want one that wraps ErrCanceled", err)
	}
	if last := s.w.ids[len(s.w.ids)-1]; last != "routes-restore" {
		t.Fatalf("the compensation ran with activity id %q, want routes-restore (a stable idempotency key)", last)
	}
	if !jsonEqual(t, s.gw.rawRoute(t, "tinystories"), before) || len(s.prom.queries) != 0 {
		t.Fatalf("after a cancel during the bake the route is %s (want the snapshot %s); %d burn queries (want 0)",
			s.gw.rawRoute(t, "tinystories"), before, len(s.prom.queries))
	}
}

func TestReplayIsDeterministic(t *testing.T) {
	// WHY: a durable workflow is replayed from its history on every
	//      restart. Replaying a finished release with no live activities at
	//      all must take the same path and give the same result; any wall
	//      clock, random choice, or map order in the workflow shows up here.
	// KIND: regression
	// CATCHES: s18, s23
	// CHAPTER: dur.12 section 2, determinism
	s := newScenario(t, vector("3.25"))
	s.w.signal("approve", "{}")
	first, err, _ := s.w.run(s.spec)
	if err != nil {
		t.Fatal(err)
	}
	for name := range s.w.impls {
		n := name
		s.w.impls[n] = func(context.Context, []byte) ([]byte, error) {
			t.Fatalf("replay executed activity %s", n)
			return nil, nil
		}
	}
	for i := 0; i < 3; i++ {
		again, err, _ := s.w.run(s.spec)
		a, _ := json.Marshal(first)
		b, _ := json.Marshal(again)
		if err != nil || string(a) != string(b) {
			t.Fatalf("replay %d gave %s (err %v), the first run %s", i, b, err, a)
		}
	}
}

func TestSpecIsValidatedFirst(t *testing.T) {
	// WHY: a canary weight of 1 is a full cutover, and a burn check without
	//      a query checks nothing. Both are refused before any activity runs.
	// KIND: boundary
	// CATCHES: s35
	// CHAPTER: dur.12 section 4, the interface
	for _, mut := range []func(*workflows.ReleaseSpec){
		func(s *workflows.ReleaseSpec) { s.CanaryWeight = 1 },
		func(s *workflows.ReleaseSpec) { s.Burn.Query = " " },
		func(s *workflows.ReleaseSpec) { s.Gates[0].Op = ">" },
	} {
		s := newScenario(t, vector("0.1"))
		mut(&s.spec)
		_, err, _ := s.w.run(s.spec)
		var g *workflows.GateError
		if !errors.As(err, &g) || g.Gate != "spec" || len(s.w.order) != 0 {
			t.Fatalf("error %v after %v, want a spec GateError before any activity", err, s.w.order)
		}
	}
}

// wrapAfter runs f after the wrapped activity's live execution.
func wrapAfter(a actFn, f func()) actFn {
	return func(ctx context.Context, in []byte) ([]byte, error) {
		out, err := a(ctx, in)
		f()
		return out, err
	}
}

func jsonEqual(t *testing.T, a, b string) bool {
	t.Helper()
	var x, y any
	if json.Unmarshal([]byte(a), &x) != nil || json.Unmarshal([]byte(b), &y) != nil {
		return false
	}
	p, _ := json.Marshal(x)
	q, _ := json.Marshal(y)
	return string(p) == string(q)
}
