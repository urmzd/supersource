// Course tests for dur.08, the workflow half: the Runtime seam's cancel
// rule and sagas (go/workflows/runtime.go, go/workflows/saga.go), driven by
// the replaying simulator in sim_test.go.
package dur_08

import (
	"errors"
	"strings"
	"testing"
	"time"

	"tinyllm/workflows"
)

var opts = workflows.StepOptions{StartToClose: time.Minute}

// booking is the chapter's three-step saga: reserve a GPU, copy the data,
// start the job; each step registers its undo before it runs.
func booking() func(rt workflows.Runtime) error {
	return func(rt workflows.Runtime) error {
		var saga workflows.Saga
		for _, st := range []struct{ do, undo, name string }{
			{"gpu.reserve", "gpu.release", "release the GPU"},
			{"data.copy", "data.delete", "delete the copy"},
			{"job.start", "job.stop", "stop the job"},
		} {
			saga.Add(st.name, st.undo, map[string]string{"step": st.do}, workflows.StepOptions{StartToClose: time.Minute})
			if err := rt.ExecuteActivity(st.do, nil, opts, nil); err != nil {
				return saga.Fail(rt, err)
			}
		}
		return nil
	}
}

func okAct(c call) (any, error) { return "ok", nil }

func newBooking(t *testing.T, failing string) *sim {
	s := newSim(t, "book-1")
	for _, n := range []string{"gpu.reserve", "data.copy", "job.start", "gpu.release", "data.delete", "job.stop"} {
		n := n
		s.register(n, func(c call) (any, error) {
			if n == failing {
				return nil, &workflows.StepFailure{Type: "QuotaExceeded", Message: "no GPU job slots", NonRetryable: true}
			}
			return "ok", nil
		})
	}
	return s
}

func TestHandExampleSagaUnwinds(t *testing.T) {
	// WHY: the chapter's worked example (section 3). job.start fails for
	//      good after the GPU was reserved and the data copied. The saga runs
	//      the compensations newest first: stop the job (registered before
	//      the step that failed, which may have half-run), delete the copy,
	//      release the GPU, each once with a stable key, and the workflow
	//      fails with the original error.
	// KIND: unit
	// CATCHES: s14, s15
	// CHAPTER: dur.08 section 3, worked example
	s := newBooking(t, "job.start")
	err := s.runToEnd(booking())
	var f *workflows.StepFailure
	if !errors.As(err, &f) || f.Type != "QuotaExceeded" {
		t.Fatalf("the workflow must fail with job.start's failure, got %v", err)
	}
	if got := strings.Join(s.order, " "); got != "gpu.reserve data.copy job.start job.stop data.delete gpu.release" {
		t.Fatalf("execution order:\n got %s\nwant gpu.reserve data.copy job.start job.stop data.delete gpu.release", got)
	}
	for _, k := range []string{"book-1/compensate-1", "book-1/compensate-2", "book-1/compensate-3"} {
		if s.executed[k] != 1 {
			t.Fatalf("%s ran %d times; each compensation has its own stable key and runs once", k, s.executed[k])
		}
	}
}

func TestCompensateRunsOnce(t *testing.T) {
	// WHY: a workflow can reach its compensation from a failure path and
	//      from a cancel path in the same run; undoing twice deletes data a
	//      retry wrote in between. Compensate is idempotent within a run.
	// KIND: unit
	// CATCHES: s16
	// CHAPTER: dur.08 section 2.3
	s := newBooking(t, "")
	err := s.runToEnd(func(rt workflows.Runtime) error {
		var saga workflows.Saga
		saga.Add("undo", "gpu.release", nil, opts)
		if err := saga.Compensate(rt); err != nil {
			return err
		}
		return saga.Compensate(rt)
	})
	if err != nil || s.executed["book-1/compensate-1"] != 1 || len(s.order) != 1 {
		t.Fatalf("two Compensate calls: err %v, executions %v", err, s.order)
	}
}

func TestCompensationsRunAfterCancel(t *testing.T) {
	// WHY: a cancel interrupts the step in progress (ErrCanceled), and every
	//      later call on the same Runtime fails the same way. Compensations
	//      therefore run on rt.Detached(); a saga that compensates on the
	//      cancelled Runtime undoes nothing and leaves the GPU reserved.
	// KIND: fault
	// CATCHES: s17, s18
	// CHAPTER: dur.08 section 2.2
	s := newBooking(t, "")
	s.cancelAt = 2 // the cancel arrives after gpu.reserve and data.copy
	err := s.runToEnd(booking())
	if !workflows.IsCanceled(err) {
		t.Fatalf("a cancelled saga returns an error matching ErrCanceled, got %v", err)
	}
	if got := strings.Join(s.order, " "); got != "gpu.reserve data.copy job.stop data.delete gpu.release" {
		t.Fatalf("after the cancel: %s; want the three compensations, newest first, and no job.start", got)
	}
}

func TestFailedCompensationDoesNotStopTheOthers(t *testing.T) {
	// WHY: each compensation undoes a different step. When deleting the copy
	//      fails for good, the GPU must still be released, and the error must
	//      carry both the cause and the compensation that failed.
	// KIND: unit
	// CATCHES: s19, s20
	// CHAPTER: dur.08 section 5, Pitfalls
	s := newBooking(t, "job.start")
	s.register("data.delete", func(c call) (any, error) {
		return nil, &workflows.StepFailure{Type: "PermissionDenied", Message: "bucket is read-only", NonRetryable: true}
	})
	err := s.runToEnd(booking())
	if got := strings.Join(s.order, " "); !strings.HasSuffix(got, "job.stop data.delete gpu.release") {
		t.Fatalf("execution order %s: gpu.release must run after data.delete failed", got)
	}
	var f *workflows.StepFailure
	if !errors.As(err, &f) || f.Type != "QuotaExceeded" {
		t.Fatalf("the cause must stay matchable: %v", err)
	}
	if !strings.Contains(err.Error(), "delete the copy") || !strings.Contains(err.Error(), "PermissionDenied") {
		t.Fatalf("the error must name the failed compensation: %v", err)
	}
}

func TestCompensationSurvivesACrash(t *testing.T) {
	// WHY: the worker can die in the middle of compensating. On replay the
	//      compensations already recorded are not run again, the one that was
	//      running when the worker died runs again with the same key (the
	//      external system deduplicates it), and the rest run once.
	// KIND: fault
	// CATCHES: s15
	// CHAPTER: dur.08 section 2.3
	for _, mode := range []string{"during", "between"} {
		s := newBooking(t, "job.start")
		// History: gpu.reserve, data.copy, job.start (failed), job.stop; the
		// worker dies at step 4: while data.delete runs ("during"), or
		// right after job.stop was recorded ("between").
		s.crashAt, s.crashMode = 4, mode
		err := s.runToEnd(booking())
		if s.crashes != 1 {
			t.Fatalf("%s: the scripted crash did not happen", mode)
		}
		var f *workflows.StepFailure
		if !errors.As(err, &f) {
			t.Fatalf("%s: %v", mode, err)
		}
		want := map[string]int{"book-1/compensate-3": 1, "book-1/compensate-2": 1, "book-1/compensate-1": 1}
		if mode == "during" {
			want["book-1/compensate-2"] = 2 // the attempt in flight when the worker died runs again
		}
		for k, n := range want {
			if s.executed[k] != n {
				t.Fatalf("%s: %s ran %d times, want %d (all: %v)", mode, k, s.executed[k], n, s.executed)
			}
		}
	}
}

func TestExplicitIDsAreKept(t *testing.T) {
	// WHY: a workflow that names a compensation's activity id (CorpusBuild
	//      keys cleanup by dataset and version) must get exactly that key;
	//      only an empty ID is numbered.
	// KIND: boundary
	// CATCHES: s21
	// CHAPTER: dur.08 section 4, The interface
	var saga workflows.Saga
	saga.Add("a", "gpu.release", nil, workflows.StepOptions{ID: "cleanup-tinystories-v1"})
	saga.Add("b", "gpu.release", nil, workflows.StepOptions{})
	if saga.Len() != 2 {
		t.Fatalf("Len = %d", saga.Len())
	}
	s := newBooking(t, "")
	if err := s.runToEnd(func(rt workflows.Runtime) error { return saga.Compensate(rt) }); err != nil {
		t.Fatal(err)
	}
	if s.executed["book-1/cleanup-tinystories-v1"] != 1 || s.executed["book-1/compensate-2"] != 1 {
		t.Fatalf("keys %v", s.executed)
	}
}

func TestIsCanceledMatchesWrapped(t *testing.T) {
	// WHY: Runtimes wrap ErrCanceled with context ("activity shard: ...");
	//      the check must see through wrapping and never match other errors.
	// KIND: unit
	// CATCHES: s22
	// CHAPTER: dur.08 section 2.2
	wrapped := errors.Join(errors.New("compensation x failed"), errors.New("activity shard: "+workflows.ErrCanceled.Error()))
	if workflows.IsCanceled(wrapped) {
		t.Fatal("a message that only looks like ErrCanceled is not a cancel")
	}
	if !workflows.IsCanceled(errors.Join(errors.New("x"), workflows.ErrCanceled)) {
		t.Fatal("a joined ErrCanceled is a cancel")
	}
	if workflows.IsCanceled(nil) || workflows.IsCanceled(errors.New("canceled")) {
		t.Fatal("nil and other errors are not cancels")
	}
}
