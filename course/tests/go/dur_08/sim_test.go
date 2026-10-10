package dur_08

// sim is a replaying workflows.Runtime, the course's stand-in for the
// durable SDK in workflow tests (the same file serves dur.08, data.09, and
// dur.11). A workflow function runs against it; every call is appended to
// hist the first time and answered from hist on a later run, which is what
// a replay after a crash does. Faults are scripted by history position:
//
//	crashAt     the run dies when hist reaches this length: in "during" mode
//	            the activity's effect happens and the worker dies before its
//	            result is recorded (the next run executes it again); in
//	            "between" mode the result is recorded and the worker dies
//	            before the workflow sees it (the next run replays it)
//	cancelAt    a cancel request arrives when hist reaches this length
//	signals     arrive when hist reaches their position
//
// Nothing here sleeps or reads the wall clock.

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"tinyllm/workflows"
)

type call struct {
	Key     string // "<workflow id>/<activity id>"
	Attempt int    // 1-based
	Input   json.RawMessage
}

// actFunc implements one activity type for the simulator. A returned error
// that is a *workflows.StepFailure with NonRetryable fails at once; any
// other error is retried up to the activity's MaxAttempts (default 3).
type actFunc func(c call) (any, error)

type step struct {
	Kind     string // activity, sleep, signal
	Name     string
	ID       string
	Result   json.RawMessage
	Failure  *workflows.StepFailure
	Canceled bool
	Payload  []byte
	Detached bool
}

type queuedSignal struct {
	at      int
	name    string
	payload []byte
	used    bool
}

type crashed struct{}

type sim struct {
	t         *testing.T
	wfID      string
	acts      map[string]actFunc
	hist      []step
	pos       int
	seq       int
	start     time.Time
	now       time.Time
	signals   []*queuedSignal
	cancelAt  int    // -1: never
	crashAt   int    // -1: never
	crashMode string // "during" or "between"
	crashes   int
	executed  map[string]int // key -> times the activity function ran
	order     []string       // activity names in execution order (re-executions included)
}

func newSim(t *testing.T, wfID string) *sim {
	return &sim{t: t, wfID: wfID, acts: map[string]actFunc{}, cancelAt: -1, crashAt: -1, crashMode: "during",
		start: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), executed: map[string]int{}}
}

func (s *sim) register(name string, f actFunc) { s.acts[name] = f }

func (s *sim) signalAt(at int, name string, payload string) {
	s.signals = append(s.signals, &queuedSignal{at: at, name: name, payload: []byte(payload)})
}

// run executes wf from the start against the recorded history; it returns
// crashed=true when a scripted crash ended this run.
func (s *sim) run(wf func(rt workflows.Runtime) error) (err error, crash bool) {
	s.pos, s.seq, s.now = 0, 0, s.start
	for _, q := range s.signals {
		q.used = false
	}
	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(crashed); !ok {
				panic(r)
			}
			s.crashes++
			s.crashAt = -1
			err, crash = nil, true
		}
	}()
	err = wf(view{s, false})
	if s.pos < len(s.hist) {
		s.t.Fatalf("nondeterminism: the run ended after %d steps but history has %d", s.pos, len(s.hist))
	}
	return err, false
}

// runToEnd repeats run after each crash, at most 20 times.
func (s *sim) runToEnd(wf func(rt workflows.Runtime) error) error {
	for i := 0; i < 20; i++ {
		err, crash := s.run(wf)
		if !crash {
			return err
		}
	}
	s.t.Fatal("the workflow never finished in 20 runs")
	return nil
}

func (s *sim) canceled() bool { return s.cancelAt >= 0 && len(s.hist) >= s.cancelAt }

// replay returns the recorded step at pos, checking it is the same call.
func (s *sim) replay(kind, name, id string) (step, bool) {
	if s.pos >= len(s.hist) {
		return step{}, false
	}
	st := s.hist[s.pos]
	if st.Kind != kind || st.Name != name || st.ID != id {
		s.t.Fatalf("nondeterminism at step %d: history has %s %s (%s), the workflow now calls %s %s (%s)",
			s.pos, st.Kind, st.Name, st.ID, kind, name, id)
	}
	s.pos++
	return st, true
}

func (s *sim) record(st step) {
	s.hist = append(s.hist, st)
	s.pos++
}

// maybeCrash ends the run right after a step was recorded ("between").
func (s *sim) maybeCrash() {
	if s.crashMode == "between" && s.crashAt >= 0 && len(s.hist) >= s.crashAt {
		panic(crashed{})
	}
}

type view struct {
	s        *sim
	detached bool
}

func (v view) ExecuteActivity(name string, in any, opts workflows.StepOptions, out any) error {
	s := v.s
	s.seq++
	id := opts.ID
	if id == "" {
		id = fmt.Sprint(s.seq)
	}
	if st, ok := s.replay("activity", name, id); ok {
		return v.result(st, out)
	}
	if s.canceled() && !v.detached {
		s.record(step{Kind: "activity", Name: name, ID: id, Canceled: true})
		return fmt.Errorf("activity %s: %w", name, workflows.ErrCanceled)
	}
	f := s.acts[name]
	if f == nil {
		s.t.Fatalf("the workflow called activity %q, which the test did not register", name)
	}
	input, err := json.Marshal(in)
	if err != nil {
		s.t.Fatalf("activity %s input does not encode: %v", name, err)
	}
	if s.crashMode == "during" && s.crashAt >= 0 && len(s.hist) >= s.crashAt {
		s.executed[s.wfID+"/"+id]++
		s.order = append(s.order, name)
		f(call{Key: s.wfID + "/" + id, Attempt: 1, Input: input})
		panic(crashed{})
	}
	max := opts.MaxAttempts
	if max <= 0 {
		max = 3
	}
	st := step{Kind: "activity", Name: name, ID: id, Detached: v.detached}
	for attempt := 1; ; attempt++ {
		s.executed[s.wfID+"/"+id]++
		s.order = append(s.order, name)
		res, err := f(call{Key: s.wfID + "/" + id, Attempt: attempt, Input: input})
		if err == nil {
			b, merr := json.Marshal(res)
			if merr != nil {
				s.t.Fatalf("activity %s result does not encode: %v", name, merr)
			}
			st.Result = b
			break
		}
		var sf *workflows.StepFailure
		if errors.As(err, &sf) && sf.NonRetryable || attempt >= max {
			if sf == nil {
				sf = &workflows.StepFailure{Type: "Error", Message: err.Error()}
			}
			cp := *sf
			cp.Activity = name
			st.Failure = &cp
			break
		}
	}
	s.record(st)
	s.maybeCrash()
	return v.result(st, out)
}

func (v view) result(st step, out any) error {
	if st.Canceled {
		return fmt.Errorf("activity %s: %w", st.Name, workflows.ErrCanceled)
	}
	if st.Failure != nil {
		f := *st.Failure
		return &f
	}
	if out != nil && len(st.Result) > 0 {
		return json.Unmarshal(st.Result, out)
	}
	return nil
}

func (v view) Sleep(d time.Duration) error {
	s := v.s
	if st, ok := s.replay("sleep", "", fmt.Sprint(d)); ok {
		s.now = s.now.Add(d)
		if st.Canceled {
			return workflows.ErrCanceled
		}
		return nil
	}
	if s.canceled() && !v.detached {
		s.record(step{Kind: "sleep", ID: fmt.Sprint(d), Canceled: true})
		return workflows.ErrCanceled
	}
	s.now = s.now.Add(d)
	s.record(step{Kind: "sleep", ID: fmt.Sprint(d)})
	s.maybeCrash()
	return nil
}

func (v view) Now() time.Time { return v.s.now }

func (v view) AwaitSignal(names []string, timeout time.Duration) (string, []byte, error) {
	s := v.s
	key := fmt.Sprint(names, timeout)
	if st, ok := s.replay("signal", "", key); ok {
		for _, q := range s.signals {
			if !q.used && q.name == st.Name && string(q.payload) == string(st.Payload) {
				q.used = true
				break
			}
		}
		if st.Canceled {
			return "", nil, workflows.ErrCanceled
		}
		return st.Name, st.Payload, nil
	}
	want := map[string]bool{}
	for _, n := range names {
		want[n] = true
	}
	// The first matching signal in arrival order: one that arrived earlier
	// is waiting in the buffer; a later one arrives while the workflow waits.
	var got *queuedSignal
	for _, q := range s.signals {
		if !q.used && want[q.name] && (got == nil || q.at < got.at) {
			got = q
		}
	}
	if s.canceled() && !v.detached && (got == nil || got.at > len(s.hist)) {
		s.record(step{Kind: "signal", ID: key, Canceled: true})
		return "", nil, workflows.ErrCanceled
	}
	if got == nil {
		if timeout <= 0 {
			s.t.Fatalf("the workflow waits for %v forever: no such signal is scripted", names)
		}
		s.now = s.now.Add(timeout)
		s.record(step{Kind: "signal", ID: key})
		s.maybeCrash()
		return "", nil, nil
	}
	got.used = true
	s.record(step{Kind: "signal", Name: got.name, ID: key, Payload: got.payload})
	s.maybeCrash()
	return got.name, got.payload, nil
}

func (v view) Detached() workflows.Runtime { return view{v.s, true} }
