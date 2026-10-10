//go:build primer && coursefaults

// Course tests for the craft.21 kata (primers/craft.21/taskq.go): the
// queue and worker checked under the course's fault kit.
//
// The `primer && coursefaults` build tags keep this file out of the
// coursetests module and out of your own test runs: craft.21's check copies
// it next to your kata and the fault kit in a scratch module named craft21
// and runs it with those tags, separately from your resilience tests, which
// it grades by planted faults.
package craft21_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"craft21"
	"craft21/faults"
)

var t0 = time.Unix(1760000000, 0)

const ttl = 30 * time.Second

func open(t *testing.T, fs *faults.CrashFS, clk *faults.Clock) *craft21.Queue {
	t.Helper()
	q, err := craft21.Open(fs, clk)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return q
}

func echo(ctx context.Context, p string) (string, error) { return "done:" + p, nil }

func worker(q *craft21.Queue, id string, clk *faults.Clock, fx *faults.Effects) *craft21.Worker {
	return &craft21.Worker{Q: q, ID: id, TTL: ttl, Clock: clk, Work: echo, Apply: fx.Apply}
}

func TestHandExampleCrashBeforeComplete(t *testing.T) {
	// WHY: the chapter's worked example (section 3). w1 leases t1 (token 1,
	//      deadline t0+30s), applies its effect, and the machine dies before
	//      Complete. After the restart nothing is lost that was acknowledged
	//      (both enqueues and the lease replay), t1 is leased again only once
	//      its lease expired (token 2, attempt 2), its effect is delivered twice
	//      and applied once, and w1's late Complete with token 1 is fenced.
	// KIND: fault
	// CHAPTER: craft.21 section 3, worked example
	fs, clk, fx := faults.NewCrashFS(), faults.NewClock(t0), faults.NewEffects()
	q := open(t, fs, clk)
	if err := q.Enqueue("t1", "a"); err != nil {
		t.Fatal(err)
	}
	if err := q.Enqueue("t2", "b"); err != nil {
		t.Fatal(err)
	}
	l1, err := q.Lease("w1", ttl)
	if err != nil || l1.Task != "t1" || l1.Token != 1 || l1.Attempt != 1 || !l1.Deadline.Equal(t0.Add(ttl)) {
		t.Fatalf("first lease: %+v, %v", l1, err)
	}
	want := `{"op":"enq","id":"t1","payload":"a"}` + "\n" +
		`{"op":"enq","id":"t2","payload":"b"}` + "\n" +
		`{"op":"lease","id":"t1","worker":"w1","token":1,"deadline":1760000030000000000}` + "\n"
	if got := string(fs.Durable(craft21.LogName)); got != want {
		t.Fatalf("durable log after the lease:\n got %q\nwant %q", got, want)
	}
	if err := fx.Apply(l1.Task, "done:a"); err != nil {
		t.Fatal(err)
	}
	fs.Crash()

	q = open(t, fs, clk)
	clk.Advance(ttl + time.Second)
	w2 := worker(q, "w2", clk, fx)
	for {
		ok, err := w2.RunOne(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
	}
	if r, ok := q.Result("t1"); !ok || r != "done:a" {
		t.Fatalf("t1 after the restart: %q, %v", r, ok)
	}
	if fx.Deliveries("t1") != 2 {
		t.Fatalf("t1 delivered %d times, want 2 (once before the crash, once after)", fx.Deliveries("t1"))
	}
	fx.AssertExactlyOnce(t, []string{"t1", "t2"})
	if err := q.Complete(l1, "late"); !errors.Is(err, craft21.ErrLeaseLost) {
		t.Fatalf("w1's late Complete with token 1: %v, want ErrLeaseLost", err)
	}
}

func TestEnqueueIsIdempotent(t *testing.T) {
	// WHY: a producer that retries Enqueue after a timeout must not create a
	//      second task, nor reset a task that is running or done.
	// KIND: unit
	// CATCHES: s05
	// CHAPTER: craft.21 section 2.1
	fs, clk, fx := faults.NewCrashFS(), faults.NewClock(t0), faults.NewEffects()
	q := open(t, fs, clk)
	q.Enqueue("t1", "a")
	q.Enqueue("t1", "other")
	if _, err := worker(q, "w", clk, fx).RunOne(context.Background()); err != nil {
		t.Fatal(err)
	}
	q.Enqueue("t1", "again")
	if ok, _ := worker(q, "w", clk, fx).RunOne(context.Background()); ok {
		t.Fatal("re-enqueueing a done task made it runnable again")
	}
	if v, _ := fx.Value("t1"); v != "done:a" || fx.Deliveries("t1") != 1 {
		t.Fatalf("t1: value %q, deliveries %d", v, fx.Deliveries("t1"))
	}
}

func TestAcknowledgedSurvivesCrash(t *testing.T) {
	// WHY: a call that returned nil is a promise. After a crash (unsynced
	//      bytes lost) every acknowledged enqueue, lease, and completion must
	//      replay; a queue that skips Sync keeps them only in the page cache.
	// KIND: fault
	// CATCHES: s12
	// CHAPTER: craft.21 section 2.2
	fs, clk, fx := faults.NewCrashFS(), faults.NewClock(t0), faults.NewEffects()
	q := open(t, fs, clk)
	for _, id := range []string{"a", "b", "c"} {
		if err := q.Enqueue(id, id); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := worker(q, "w", clk, fx).RunOne(context.Background()); err != nil {
		t.Fatal(err)
	}
	fs.Crash()
	q = open(t, fs, clk)
	if _, ok := q.Result("a"); !ok {
		t.Fatal("the completion of a was acknowledged and lost in the crash")
	}
	if got := strings.Join(q.Pending(), ","); got != "b,c" {
		t.Fatalf("pending after the crash: %q, want b,c", got)
	}
}

func TestTornTailIsIgnored(t *testing.T) {
	// WHY: the crash can land in the middle of a write: the log ends in half
	//      a record. Open must keep everything before it, ignore the torn
	//      fragment (its call never returned), and seal it, so the next
	//      acknowledged record is not glued to it and lost on the replay after
	//      the next crash.
	// KIND: fault
	// CATCHES: s04, s09
	// CHAPTER: craft.21 section 2.2
	fs, clk := faults.NewCrashFS(), faults.NewClock(t0)
	q := open(t, fs, clk)
	q.Enqueue("a", "1")
	// A record written and never synced, then torn by the crash.
	f, _ := fs.Append(craft21.LogName)
	f.Write([]byte(`{"op":"enq","id":"b","payload":"2"}` + "\n"))
	fs.CrashTorn(10)
	q = open(t, fs, clk)
	if got := strings.Join(q.Pending(), ","); got != "a" {
		t.Fatalf("pending after a torn write: %q, want a", got)
	}
	if err := q.Enqueue("c", "3"); err != nil {
		t.Fatalf("the queue must keep working after a torn tail: %v", err)
	}
	fs.Crash()
	q = open(t, fs, clk)
	if got := strings.Join(q.Pending(), ","); got != "a,c" {
		t.Fatalf("c was acknowledged after the torn crash and lost in the next one: pending %q, want a,c", got)
	}
}

func TestFailedWriteChangesNothing(t *testing.T) {
	// WHY: a write that failed (disk full) must leave the in-memory state
	//      as it was: a queue that updates memory first serves a lease or a
	//      completion that is not on disk and vanishes on restart.
	// KIND: fault
	// CATCHES: s07
	// CHAPTER: craft.21 section 2.2
	fs, clk := faults.NewCrashFS(), faults.NewClock(t0)
	q := open(t, fs, clk)
	q.Enqueue("a", "1")
	l, err := q.Lease("w", ttl)
	if err != nil {
		t.Fatal(err)
	}
	fs.FailWrites(errors.New("ENOSPC"))
	if err := q.Complete(l, "r"); err == nil {
		t.Fatal("Complete must fail when its record cannot be written")
	}
	if _, ok := q.Result("a"); ok {
		t.Fatal("a failed Complete still marked the task done in memory")
	}
	if err := q.Enqueue("b", "2"); err == nil {
		t.Fatal("Enqueue must fail when its record cannot be written")
	}
	if strings.Join(q.Pending(), ",") != "a" {
		t.Fatalf("a failed Enqueue added a task: %v", q.Pending())
	}
}

func TestLeaseIsExclusiveUntilExpiry(t *testing.T) {
	// WHY: while a lease is live nobody else gets the task; once it expires
	//      (the holder is dead or partitioned) the task is redelivered with a
	//      new, higher token and attempt + 1.
	// KIND: boundary
	// CATCHES: s02, s03
	// CHAPTER: craft.21 section 2.3
	fs, clk := faults.NewCrashFS(), faults.NewClock(t0)
	q := open(t, fs, clk)
	q.Enqueue("a", "1")
	l1, _ := q.Lease("w1", ttl)
	if _, err := q.Lease("w2", ttl); !errors.Is(err, craft21.ErrNoTask) {
		t.Fatalf("a live lease was handed out again: %v", err)
	}
	clk.Advance(ttl - time.Nanosecond)
	if _, err := q.Lease("w2", ttl); !errors.Is(err, craft21.ErrNoTask) {
		t.Fatalf("one nanosecond before the deadline: %v", err)
	}
	clk.Advance(time.Nanosecond)
	l2, err := q.Lease("w2", ttl)
	if err != nil || l2.Token <= l1.Token || l2.Attempt != 2 {
		t.Fatalf("at the deadline the task is redelivered with a higher token: %+v, %v", l2, err)
	}
}

func TestStaleTokenIsFenced(t *testing.T) {
	// WHY: the worker whose lease expired may still be running (a GC pause,
	//      a partition). Its Renew and Complete carry an old token and must
	//      fail, or a zombie overwrites the result of the worker that owns
	//      the task now.
	// KIND: fault
	// CATCHES: s01, s06
	// CHAPTER: craft.21 section 2.3
	fs, clk := faults.NewCrashFS(), faults.NewClock(t0)
	q := open(t, fs, clk)
	q.Enqueue("a", "1")
	old, _ := q.Lease("w1", ttl)
	clk.Advance(ttl)
	cur, _ := q.Lease("w2", ttl)
	if _, err := q.Renew(old, ttl); !errors.Is(err, craft21.ErrLeaseLost) {
		t.Fatalf("Renew with the old token: %v", err)
	}
	if err := q.Complete(old, "zombie"); !errors.Is(err, craft21.ErrLeaseLost) {
		t.Fatalf("Complete with the old token: %v", err)
	}
	if err := q.Complete(cur, "owner"); err != nil {
		t.Fatal(err)
	}
	if r, _ := q.Result("a"); r != "owner" {
		t.Fatalf("result %q, want the current owner's", r)
	}
}

// slowWork blocks until released, so the test controls how long a task runs.
type slowWork struct {
	started  chan struct{}
	release  chan struct{}
	canceled chan struct{}
	once     sync.Once
}

func newSlow() *slowWork {
	return &slowWork{started: make(chan struct{}), release: make(chan struct{}), canceled: make(chan struct{})}
}

func (s *slowWork) work(ctx context.Context, p string) (string, error) {
	s.once.Do(func() { close(s.started) })
	select {
	case <-s.release:
		return "done:" + p, nil
	case <-ctx.Done():
		close(s.canceled)
		return "", ctx.Err()
	}
}

func TestLongTaskKeepsItsLease(t *testing.T) {
	// WHY: a task longer than the lease must keep it by renewing (every
	//      TTL/3): otherwise a second worker leases it mid-run and the work is
	//      done twice, and the first worker's Complete is fenced.
	// KIND: fault
	// CATCHES: s11
	// CHAPTER: craft.21 section 2.3
	fs, clk, fx := faults.NewCrashFS(), faults.NewClock(t0), faults.NewEffects()
	q := open(t, fs, clk)
	q.Enqueue("a", "1")
	s := newSlow()
	w1 := &craft21.Worker{Q: q, ID: "w1", TTL: ttl, Clock: clk, Work: s.work, Apply: fx.Apply}
	done := make(chan error, 1)
	go func() { _, err := w1.RunOne(context.Background()); done <- err }()
	<-s.started
	for i := 0; i < 9; i++ { // 9 x 10 s = 90 s: three times the lease
		if err := clk.BlockUntil(1); err != nil {
			t.Fatal(err)
		}
		clk.Advance(ttl / 3)
		if _, err := q.Lease("w2", ttl); !errors.Is(err, craft21.ErrNoTask) {
			t.Fatalf("after %v a second worker got the task: %v", time.Duration(i+1)*ttl/3, err)
		}
	}
	close(s.release)
	if err := <-done; err != nil {
		t.Fatalf("w1 kept its lease, so it completes: %v", err)
	}
	if fx.Deliveries("a") != 1 {
		t.Fatalf("delivered %d times", fx.Deliveries("a"))
	}
}

func TestLostLeaseCancelsWork(t *testing.T) {
	// WHY: when a renewal fails (the lease expired during a pause and
	//      another worker took the task), the work must stop and its effect
	//      must not be applied by the worker that no longer owns the task.
	// KIND: fault
	// CATCHES: s08
	// CHAPTER: craft.21 section 2.3
	fs, clk, fx := faults.NewCrashFS(), faults.NewClock(t0), faults.NewEffects()
	q := open(t, fs, clk)
	q.Enqueue("a", "1")
	s := newSlow()
	w1 := &craft21.Worker{Q: q, ID: "w1", TTL: ttl, Clock: clk, Work: s.work, Apply: fx.Apply}
	done := make(chan error, 1)
	go func() { _, err := w1.RunOne(context.Background()); done <- err }()
	<-s.started
	if err := clk.BlockUntil(1); err != nil {
		t.Fatal(err)
	}
	// A pause longer than the lease: the renewal wakes up too late.
	clk.Advance(2 * ttl)
	select {
	case err := <-done:
		if !errors.Is(err, craft21.ErrLeaseLost) {
			t.Fatalf("RunOne after losing the lease: %v, want ErrLeaseLost", err)
		}
	case <-time.After(5 * time.Second):
		close(s.release)
		t.Fatal("the work was not cancelled when the lease was lost")
	}
	if fx.Deliveries("a") != 0 {
		t.Fatal("a worker that lost its lease applied the effect")
	}
}

func TestCrashAfterApplyIsExactlyOnce(t *testing.T) {
	// WHY: the classic window: the effect is applied, then the worker dies
	//      before Complete. The retry applies it again, so the only thing
	//      that makes it exactly once is the idempotency key: the task id,
	//      the same on every attempt, never the attempt or the token.
	// KIND: fault
	// CATCHES: s10
	// CHAPTER: craft.21 section 2.4
	fs, clk, fx := faults.NewCrashFS(), faults.NewClock(t0), faults.NewEffects()
	q := open(t, fs, clk)
	for _, id := range []string{"a", "b", "c"} {
		q.Enqueue(id, id)
	}
	dying := &craft21.Worker{Q: q, ID: "w1", TTL: ttl, Clock: clk, Work: echo,
		Apply: func(k, v string) error {
			fx.Apply(k, v)
			return errors.New("killed after applying")
		}}
	dying.RunOne(context.Background())
	fs.Crash()
	q = open(t, fs, clk)
	clk.Advance(ttl)
	w2 := worker(q, "w2", clk, fx)
	for {
		ok, err := w2.RunOne(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
	}
	fx.AssertExactlyOnce(t, []string{"a", "b", "c"})
	if fx.Deliveries("a") != 2 {
		t.Fatalf("a was delivered %d times; the test needs the retry to happen", fx.Deliveries("a"))
	}
}
