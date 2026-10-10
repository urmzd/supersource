// Course tests for dur.12: ModelRelease (go/workflows/model_release.go) and
// its activities, the gateway route client (go/activities/routes.go) and
// the PromQL query (go/activities/promql.go).
//
// This file holds the test doubles; the tests are in release_test.go,
// routes_test.go, and promql_test.go.
//
//   - world/runEnv: a durable SDK in miniature. It records every activity
//     result, timer, and signal in a history; a "crash" (a panic after an
//     activity's side effect, before its completion is recorded) restarts
//     the workflow from the top, which replays the history and re-executes
//     only what never completed: exactly what the durable server does when
//     a worker dies (dur.06). The clock is virtual: Sleep moves it.
//   - gateway: the admin route API of openapi/admin.v1.yaml (ETag, If-Match,
//     412, 428, weights summing to 1), counting PUTs.
//   - prometheus: GET /api/v1/query answering a canned body.
//
// No test sleeps on the wall clock.
package dur_12

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"tinyllm/activities"
	"tinyllm/workflows"
)

const adminKey = "tl_admintestkey0_0123456789abcdef0123456789abcdef"

var t0 = time.Date(2026, 10, 9, 12, 0, 0, 0, time.UTC)

// ---------------------------------------------------------------------------
// The replaying environment.

type event struct {
	Kind     string // activity | timer | signal
	Name     string
	Result   []byte
	Failure  *workflows.StepFailure
	Canceled bool // the call returned ErrCanceled
	Deadline time.Time
	Fired    bool
	Payload  []byte
}

type actFn func(ctx context.Context, in []byte) ([]byte, error)

type crashSignal struct{ where string }

type sig struct {
	name    string
	payload []byte
}

// world is everything that survives a worker crash: the history, the
// clock, the buffered signals, and the live activity implementations.
type world struct {
	t       *testing.T
	hist    []event
	now     time.Time
	impls   map[string]actFn
	signals []sig
	execs   map[string]int  // live executions per activity, every attempt
	inputs  map[string][]byte // last live input per activity
	crash   func(kind, name string) bool
	cancel  func(kind, name string) bool // a cancel request arrives at this call (not on Detached)
	order   []string // live activity completions and signal receipts, in order
	ids     []string // the StepOptions.ID of every live activity call
	downFor time.Duration
}

func newWorld(t *testing.T) *world {
	return &world{t: t, now: t0, impls: map[string]actFn{}, execs: map[string]int{}, inputs: map[string][]byte{}}
}

func (w *world) signal(name string, payload string) {
	w.signals = append(w.signals, sig{name, []byte(payload)})
}

// runEnv is one run of the workflow over the world's history: a
// workflows.Runtime. Detached() shares the history cursor and ignores
// cancel requests, as dur.08's contract says.
type runEnv struct {
	w        *world
	cur      *int
	detached bool
}

func (e *runEnv) Detached() workflows.Runtime { return &runEnv{w: e.w, cur: e.cur, detached: true} }

func (e *runEnv) canceled(kind, name string) bool {
	return !e.detached && e.w.cancel != nil && e.w.cancel(kind, name)
}

func canceledErr(what string) error { return fmt.Errorf("%s: %w", what, workflows.ErrCanceled) }

type nondeterminism struct{ msg string }

func (e *runEnv) next(kind, name string) (*event, bool) {
	if *e.cur >= len(e.w.hist) {
		return nil, false
	}
	ev := &e.w.hist[*e.cur]
	if ev.Kind != kind || ev.Name != name {
		panic(nondeterminism{fmt.Sprintf("history event %d is %s %q, the workflow asked for %s %q",
			*e.cur, ev.Kind, ev.Name, kind, name)})
	}
	return ev, true
}

func nonRetryable(err error) bool {
	var nr interface{ NonRetryable() bool }
	return errors.As(err, &nr) && nr.NonRetryable()
}

func decode(b []byte, out any) error {
	if out == nil {
		return nil
	}
	return json.Unmarshal(b, out)
}

func (e *runEnv) ExecuteActivity(name string, in any, opts workflows.StepOptions, out any) error {
	if ev, ok := e.next("activity", name); ok {
		*e.cur++
		switch {
		case ev.Canceled:
			return canceledErr("activity " + name)
		case ev.Failure != nil:
			return ev.Failure
		}
		return decode(ev.Result, out)
	}
	if e.canceled("activity", name) {
		e.w.hist = append(e.w.hist, event{Kind: "activity", Name: name, Canceled: true})
		*e.cur++
		return canceledErr("activity " + name)
	}
	e.w.ids = append(e.w.ids, opts.ID)
	inb, err := json.Marshal(in)
	if err != nil {
		e.w.t.Fatalf("activity %s: input does not encode: %v", name, err)
	}
	impl := e.w.impls[name]
	if impl == nil {
		e.w.t.Fatalf("the workflow scheduled activity %q, which this test does not expect", name)
	}
	attempts := opts.MaxAttempts
	if attempts <= 0 {
		attempts = 1
	}
	var last error
	for a := 1; a <= attempts; a++ {
		e.w.execs[name]++
		e.w.inputs[name] = inb
		res, err := impl(context.Background(), inb)
		if err == nil {
			if e.w.crash != nil && e.w.crash("activity", name) {
				panic(crashSignal{"after activity " + name})
			}
			e.w.hist = append(e.w.hist, event{Kind: "activity", Name: name, Result: res})
			e.w.order = append(e.w.order, name)
			*e.cur++
			return decode(res, out)
		}
		last = err
		if nonRetryable(err) {
			break
		}
	}
	f := &workflows.StepFailure{Activity: name, Type: fmt.Sprintf("%T", last), Message: last.Error(),
		NonRetryable: nonRetryable(last)}
	e.w.hist = append(e.w.hist, event{Kind: "activity", Name: name, Failure: f})
	e.w.order = append(e.w.order, name+"!")
	*e.cur++
	return f
}

func (e *runEnv) Sleep(d time.Duration) error {
	if ev, ok := e.next("timer", ""); ok {
		*e.cur++
		if ev.Canceled {
			return canceledErr("timer")
		}
		if !ev.Fired && e.canceled("timer", "") {
			ev.Canceled = true
			return canceledErr("timer")
		}
		if !ev.Fired { // the worker died while the timer was pending: it fires at its deadline
			if e.w.now.Before(ev.Deadline) {
				e.w.now = ev.Deadline
			}
			ev.Fired = true
		}
		return nil
	}
	e.w.hist = append(e.w.hist, event{Kind: "timer", Deadline: e.w.now.Add(d)})
	ev := &e.w.hist[len(e.w.hist)-1]
	if e.w.crash != nil && e.w.crash("timer", "") {
		panic(crashSignal{"while the timer is pending"})
	}
	if e.canceled("timer", "") {
		ev.Canceled = true
		*e.cur++
		return canceledErr("timer")
	}
	e.w.now = ev.Deadline
	ev.Fired = true
	*e.cur++
	return nil
}

func (e *runEnv) Now() time.Time { return e.w.now }

func (e *runEnv) AwaitSignal(names []string, timeout time.Duration) (string, []byte, error) {
	if ev, ok := e.next("signal", ""); ok {
		*e.cur++
		if ev.Canceled {
			return "", nil, canceledErr("signal wait")
		}
		return ev.Name2(), ev.Payload, nil
	}
	if e.canceled("signal", "") {
		e.w.hist = append(e.w.hist, event{Kind: "signal", Canceled: true})
		*e.cur++
		return "", nil, canceledErr("signal wait")
	}
	for i, s := range e.w.signals {
		for _, n := range names {
			if s.name == n {
				e.w.signals = append(e.w.signals[:i:i], e.w.signals[i+1:]...)
				e.w.hist = append(e.w.hist, event{Kind: "signal", Payload: s.payload, Result: []byte(s.name)})
				e.w.order = append(e.w.order, "signal:"+s.name)
				*e.cur++
				return s.name, s.payload, nil
			}
		}
	}
	if timeout <= 0 {
		e.w.t.Fatalf("the workflow waits for %v with no timeout and no signal was sent: it would wait forever", names)
	}
	e.w.now = e.w.now.Add(timeout)
	e.w.hist = append(e.w.hist, event{Kind: "signal"})
	e.w.order = append(e.w.order, "signal:timeout")
	*e.cur++
	return "", nil, nil
}

// Name2 is the signal name a history event recorded ("" for a timeout).
func (ev *event) Name2() string { return string(ev.Result) }

// run drives ModelRelease to its end, restarting it after every crash.
func (w *world) run(spec workflows.ReleaseSpec) (workflows.ReleaseResult, error, int) {
	w.t.Helper()
	for restarts := 0; restarts < 20; restarts++ {
		var (
			res     workflows.ReleaseResult
			err     error
			crashed bool
		)
		func() {
			defer func() {
				if r := recover(); r != nil {
					switch x := r.(type) {
					case crashSignal:
						crashed = true
					case nondeterminism:
						w.t.Fatalf("replay diverged from the history (ErrNondeterminism): %s", x.msg)
					default:
						panic(r)
					}
				}
			}()
			cur := 0
			e := &runEnv{w: w, cur: &cur}
			res, err = workflows.ModelRelease(e, spec)
			if *e.cur != len(w.hist) {
				w.t.Fatalf("the workflow finished after %d of %d history events: replay diverged", *e.cur, len(w.hist))
			}
		}()
		if !crashed {
			return res, err, restarts
		}
		w.now = w.now.Add(w.downFor) // the worker is down for a while before another picks the run up
	}
	w.t.Fatal("the workflow crashed 20 times in a row")
	return workflows.ReleaseResult{}, nil, 0
}

func jsonAct[I, O any](f func(context.Context, I) (O, error)) actFn {
	return func(ctx context.Context, in []byte) ([]byte, error) {
		var v I
		if err := json.Unmarshal(in, &v); err != nil {
			return nil, err
		}
		out, err := f(ctx, v)
		if err != nil {
			return nil, err
		}
		return json.Marshal(out)
	}
}

// ---------------------------------------------------------------------------
// The gateway's admin route API.

type gateway struct {
	mu      sync.Mutex
	routes  []json.RawMessage
	epoch   uint64
	workers []map[string]any
	puts    int
	gets    int
	ifMatch []string
	bodies  [][]byte
	// beforePut runs once before the next PUT is judged (a concurrent edit).
	beforePut func(g *gateway)
	reject    bool // answer every PUT 400
	srv       *httptest.Server
}

func newGateway(t *testing.T, routes string, workers ...string) *gateway {
	g := &gateway{epoch: 7}
	if err := json.Unmarshal([]byte(routes), &g.routes); err != nil {
		t.Fatalf("bad test routes: %v", err)
	}
	for _, m := range workers {
		g.workers = append(g.workers, map[string]any{"worker_id": "w-" + m, "role": "unified", "model": m,
			"http_address": "127.0.0.1:1", "draining": false})
	}
	g.srv = httptest.NewServer(g)
	t.Cleanup(g.srv.Close)
	return g
}

func (g *gateway) etag() string { return `"routes-` + strconv.FormatUint(g.epoch, 10) + `"` }

func (g *gateway) client() activities.Routes {
	return activities.Routes{Base: g.srv.URL, Key: adminKey, Client: g.srv.Client()}
}

func writeErr(w http.ResponseWriter, status int, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	fmt.Fprintf(w, `{"error":{"message":%q,"type":"invalid_request_error","param":null,"code":%q}}`, code, code)
}

func (g *gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if r.Header.Get("Authorization") != "Bearer "+adminKey {
		writeErr(w, 401, "invalid_api_key")
		return
	}
	switch {
	case r.URL.Path == "/admin/v1/workers" && r.Method == http.MethodGet:
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": g.workers, "route_epoch": g.epoch})
	case r.URL.Path == "/admin/v1/routes" && r.Method == http.MethodGet:
		g.gets++
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("ETag", g.etag())
		plainJSON(w, map[string]any{"route_epoch": g.epoch, "routes": g.routes})
	case r.URL.Path == "/admin/v1/routes" && r.Method == http.MethodPut:
		if g.beforePut != nil {
			f := g.beforePut
			g.beforePut = nil
			f(g)
		}
		im := r.Header.Get("If-Match")
		g.ifMatch = append(g.ifMatch, im)
		if im == "" {
			writeErr(w, 428, "if_match_required")
			return
		}
		if im != g.etag() {
			writeErr(w, 412, "etag_mismatch")
			return
		}
		body, _ := io.ReadAll(r.Body)
		g.bodies = append(g.bodies, body)
		var t struct {
			Routes []json.RawMessage `json:"routes"`
		}
		if g.reject || json.Unmarshal(body, &t) != nil || !weightsSumToOne(t.Routes) {
			writeErr(w, 400, "invalid_request_error")
			return
		}
		g.routes = t.Routes
		g.epoch++
		g.puts++
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("ETag", g.etag())
		plainJSON(w, map[string]any{"route_epoch": g.epoch, "routes": g.routes})
	default:
		writeErr(w, 404, "not_found")
	}
}

// plainJSON writes v without HTML escaping, as a careful server does, so a
// route holding ">=" reads back byte for byte.
func plainJSON(w io.Writer, v any) {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

func weightsSumToOne(routes []json.RawMessage) bool {
	for _, raw := range routes {
		var r struct {
			Backends []activities.Backend `json:"backends"`
		}
		if json.Unmarshal(raw, &r) != nil {
			return false
		}
		if len(r.Backends) == 0 {
			continue
		}
		sum := 0.0
		for _, b := range r.Backends {
			sum += b.Weight
		}
		if sum < 0.999 || sum > 1.001 {
			return false
		}
	}
	return true
}

// route returns the stored route for model, decoded.
func (g *gateway) route(t *testing.T, model string) map[string]any {
	t.Helper()
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, raw := range g.routes {
		var m map[string]any
		if json.Unmarshal(raw, &m) == nil && m["model"] == model {
			return m
		}
	}
	t.Fatalf("the gateway has no route %q", model)
	return nil
}

func (g *gateway) rawRoute(t *testing.T, model string) string {
	t.Helper()
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, raw := range g.routes {
		var m map[string]any
		if json.Unmarshal(raw, &m) == nil && m["model"] == model {
			return string(raw)
		}
	}
	t.Fatalf("the gateway has no route %q", model)
	return ""
}

func backendsOf(t *testing.T, route map[string]any) map[string]float64 {
	t.Helper()
	out := map[string]float64{}
	bs, _ := route["backends"].([]any)
	for _, b := range bs {
		m := b.(map[string]any)
		out[m["served_model"].(string)] = m["weight"].(float64)
	}
	return out
}

// ---------------------------------------------------------------------------
// Prometheus.

type prometheus struct {
	mu      sync.Mutex
	status  int
	body    string
	queries []map[string]string
	srv     *httptest.Server
}

func newProm(t *testing.T, status int, body string) *prometheus {
	p := &prometheus{status: status, body: body}
	p.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p.mu.Lock()
		defer p.mu.Unlock()
		if r.URL.Path != "/api/v1/query" {
			w.WriteHeader(404)
			return
		}
		_ = r.ParseForm()
		p.queries = append(p.queries, map[string]string{"query": r.Form.Get("query"), "time": r.Form.Get("time")})
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(p.status)
		_, _ = io.WriteString(w, p.body)
	}))
	t.Cleanup(p.srv.Close)
	return p
}

func (p *prometheus) client() activities.Prom {
	return activities.Prom{Base: p.srv.URL, Client: p.srv.Client()}
}

func vector(value string) string {
	return `{"status":"success","data":{"resultType":"vector","result":[{"metric":{},"value":[1791547800.000,"` + value + `"]}]}}`
}

const emptyVector = `{"status":"success","data":{"resultType":"vector","result":[]}}`

// ---------------------------------------------------------------------------
// A release scenario: files, the fake services, and the activities wired up.

const burnQuery = `sum(rate(http_server_request_duration_seconds_count{model="tinystories-10m-v2",http_response_status_code=~"5.."}[5m])) / sum(rate(http_server_request_duration_seconds_count{model="tinystories-10m-v2"}[5m])) / (1 - 0.99)`

const goodCard = `# Model card: tinystories-10m v2

## Model details

- **Developer:** forge.
- **Architecture:** llama, 8 layers, d 320, 10.4M parameters.

## Intended use

- **Primary uses:** short children's stories.

## Evaluation

| Suite | Metric | Value (95% CI) | Release threshold |
|---|---|---|---|
| quality | bpb | 1.21 (1.20, 1.22) | <= 1.30 |

## Bias, risks, and limitations

- Trained on synthetic stories only; p < 0.05 bias probes listed in evals.

## Data

TinyStories (CDLA-Sharing-1.0), scrubbed.

<!-- contracts/templates/MODEL_CARD.md: <placeholders> inside comments are fine -->
`

const goodLedger = `{"source_id":"tinystories","url":"https://huggingface.co/datasets/roneneldan/TinyStories","license_spdx":"CDLA-Sharing-1.0","retrieved_at":"2026-10-01T00:00:00Z","sha256":"` + "0000000000000000000000000000000000000000000000000000000000000000" + `","n_docs":10,"allowed_uses":["train","eval"],"pii_policy":"scrub","filters_applied":["gopher"],"kept":9,"dropped":1,"notes":""}
`

const twoRoutes = `[
 {"model":"tinystories","aliases":["ts"],"x-owner":"ml-team"},
 {"model":"smart","cascade":[{"model":"tinystories","accept_if":"mean_logprob >= -1.5"},{"model":"smol-135m"}]}
]`

// releaseRoutes: the scenario's table; tinystories serves v1 explicitly.
const releaseRoutes = `[
 {"model":"tinystories","aliases":["ts"],"backends":[{"served_model":"tinystories-10m-v1","weight":1}],"x-owner":"ml-team"},
 {"model":"smart","cascade":[{"model":"tinystories","accept_if":"mean_logprob >= -1.5"},{"model":"smol-135m"}]}
]`

type scenario struct {
	w      *world
	gw     *gateway
	prom   *prometheus
	dir    string
	spec   workflows.ReleaseSpec
	evalOK bool
	rows   []workflows.EvalReport
}

func f64(v float64) *float64 { return &v }

func defaultRows() []workflows.EvalReport {
	return []workflows.EvalReport{
		{Suite: "quality", Rows: []workflows.EvalRow{
			{Model: "tinystories-10m", Task: "ts-val", Metric: "bpb", Value: f64(1.21), Status: "ok"},
		}},
		{Suite: "safety", Rows: []workflows.EvalRow{
			{Model: "tinystories-10m", Task: "refusal", Metric: "score", Value: f64(0.97), Status: "ok"},
		}},
	}
}

// newScenario: the route "tinystories" serves tinystories-10m-v1, a worker already serves tinystories-10m-v2, the files
// pass the gates, and Prometheus answers `promBody`.
func newScenario(t *testing.T, promBody string) *scenario {
	t.Helper()
	s := &scenario{w: newWorld(t), dir: t.TempDir(), rows: defaultRows()}
	s.gw = newGateway(t, releaseRoutes, "tinystories-10m-v1", "tinystories-10m-v2")
	s.prom = newProm(t, 200, promBody)
	writeFile(t, filepath.Join(s.dir, "MODEL_CARD.md"), goodCard)
	writeFile(t, filepath.Join(s.dir, "LEDGER.jsonl"), goodLedger)
	s.spec = workflows.ReleaseSpec{
		ModelID: "tinystories-10m", Version: "v2", From: "runs/c1/ckpt",
		ModelCard: filepath.Join(s.dir, "MODEL_CARD.md"), Ledger: filepath.Join(s.dir, "LEDGER.jsonl"),
		Suites: []string{"quality", "safety"},
		Gates: []workflows.Gate{{Suite: "quality", Task: "ts-val", Metric: "bpb", Op: "<=", Value: 1.30}, {Suite: "safety", Task: "refusal", Metric: "score", Op: ">=", Value: 0.9}},
		Route: "tinystories", CanaryWeight: 0.1, CanaryWaitS: 600, ApproveTimeoutS: 86400,
		Burn: workflows.BurnCheck{Query: burnQuery, Max: 14.4},
	}
	routes := s.gw.client()
	prom := s.prom.client()
	s.w.impls[workflows.ActPreflight] = jsonAct(workflows.Preflight)
	s.w.impls[workflows.ActExport] = func(ctx context.Context, in []byte) ([]byte, error) {
		return []byte(`{"outputs":["models/tinystories-10m/v2"]}`), nil
	}
	s.w.impls[workflows.ActivityEval] = func(ctx context.Context, in []byte) ([]byte, error) {
		var spec struct {
			Suites []string `json:"suites"`
		}
		if err := json.Unmarshal(in, &spec); err != nil || len(spec.Suites) != 1 {
			return nil, fmt.Errorf("eval: want one suite per activity: %s", in)
		}
		for _, rep := range s.rows {
			if rep.Suite != spec.Suites[0] {
				continue
			}
			doc := map[string]any{"format": "tl.eval-results.v1", "suite": rep.Suite, "seed": 0, "rows": rep.Rows}
			b, _ := json.Marshal(doc)
			p := filepath.Join(s.dir, "evals-"+rep.Suite+".json")
			if err := os.WriteFile(p, b, 0o644); err != nil {
				return nil, err
			}
			return json.Marshal(map[string]any{"outputs": []string{filepath.Join(s.dir, "summary-"+rep.Suite+".txt"), p}})
		}
		return json.Marshal(map[string]any{"outputs": []string{}}) // the suite produced no report
	}
	s.w.impls[workflows.ActResults] = jsonAct(workflows.ReadResults)
	s.w.impls[workflows.ActSnapshot] = jsonAct(routes.Snapshot)
	s.w.impls[workflows.ActCanary] = jsonAct(routes.Canary)
	s.w.impls[workflows.ActPromote] = jsonAct(routes.Promote)
	s.w.impls[workflows.ActRestore] = jsonAct(routes.Restore)
	s.w.impls[workflows.ActQuery] = jsonAct(prom.Query)
	return s
}

func writeFile(t *testing.T, p, s string) {
	t.Helper()
	if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
}

func approx(a, b float64) bool { return a-b < 1e-9 && b-a < 1e-9 }

func joined(xs []string) string { return strings.Join(xs, " ") }
