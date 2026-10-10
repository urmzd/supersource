// Course proofs for craft.08: every seeded defect in the review pull request
// is a real bug, not a matter of taste.
//
// The pull request adds idle-key eviction to gw.03's rate limiter
// (go/gateway/limit). These tests run against three versions of limit.go in
// a scratch module built by test_craft08_review.py: the course reference plus
// the intended PR (every test passes, under the race detector), and the
// intended PR plus exactly one defect (the test named for it fails). Time is
// a fake clock; the sweep cadence (256 Reserve calls) is driven call by call.
// The tests use only the package's exported API.
package limit_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"tinyllm/gateway/limit"
)

type proofClock struct{ t time.Time }

func (c *proofClock) Now() time.Time { return c.t }

// cadence is the PR's sweepEvery: the 256th, 512th, ... Reserve call sweeps.
const cadence = 256

var proofT0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

// limiter counts its Reserve calls so a test can land on a sweep exactly.
type limiter struct {
	*limit.Limiter
	calls int
}

func newLimiter(c *proofClock) *limiter { return &limiter{Limiter: limit.New(c)} }

func (l *limiter) reserve(key string, lim limit.Limits, c limit.Cost) (limit.Reservation, error) {
	l.calls++
	return l.Reserve(context.Background(), key, lim, c)
}

// fillTo makes unlimited Reserve calls on a filler key until call n.
func (l *limiter) fillTo(t *testing.T, n int) {
	t.Helper()
	for l.calls < n {
		if _, err := l.reserve("filler", limit.Limits{}, limit.Cost{Requests: 1}); err != nil {
			t.Fatalf("filler call %d rejected: %v", l.calls, err)
		}
	}
}

func isReject(err error) bool {
	var rej *limit.RejectError
	return errors.As(err, &rej)
}

func TestProofDebtIsNotForgiven(t *testing.T) {
	// WHY: an overrun drives a bucket below zero, and two idle minutes refill
	//      only two capacities; a sweep that drops the key anyway hands it a
	//      full bucket, so the overrun is never paid. TPM 1000: reserve 300,
	//      settle 4000 leaves -3000; 150 s later the level is -500, so the
	//      next request must still be rejected after the sweep.
	// KIND: regression
	// CATCHES: s01
	// CHAPTER: craft.08 section 5, Pitfalls, item 1
	c := &proofClock{t: proofT0}
	l := newLimiter(c)
	lim := limit.Limits{TPM: 1000}
	r, err := l.reserve("k", lim, limit.Cost{Requests: 1, Tokens: 300})
	if err != nil {
		t.Fatal(err)
	}
	r.Settle(limit.Cost{Requests: 1, Tokens: 4000})
	c.t = c.t.Add(150 * time.Second)
	l.fillTo(t, cadence) // this call sweeps
	if _, err := l.reserve("k", lim, limit.Cost{Requests: 1, Tokens: 10}); !isReject(err) {
		t.Fatalf("Reserve after the sweep = %v, want a rejection: the key still owes 500 tokens", err)
	}
}

func TestProofLenIsRaceFree(t *testing.T) {
	// WHY: Len is read by the metrics loop while requests add keys; reading
	//      the map without the limiter's lock is a data race even though the
	//      length is one word. Run under -race, four goroutines add keys
	//      while this one calls Len.
	// KIND: fault
	// CATCHES: s02
	// CHAPTER: craft.08 section 5, Pitfalls, item 2
	l := limit.New(&proofClock{t: proofT0})
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				if _, err := l.Reserve(context.Background(), fmt.Sprintf("k%d-%d", g, i), limit.Limits{}, limit.Cost{Requests: 1}); err != nil {
					t.Error(err)
					return
				}
			}
		}(g)
	}
	seen := 0
	for i := 0; i < 400; i++ {
		seen = max(seen, l.Len())
	}
	wg.Wait()
	if n := l.Len(); n != 800 {
		t.Fatalf("Len() = %d after 800 distinct keys, want 800 (max seen while adding: %d)", n, seen)
	}
}

func TestProofInFlightChargeIsKept(t *testing.T) {
	// WHY: a stream can outlive IdleTTL. Its key holds an open reservation,
	//      so sweep must keep it: otherwise Settle finds no key and the
	//      overrun (5000 used, 100 reserved) is never charged.
	// KIND: regression
	// CATCHES: s03
	// CHAPTER: craft.08 section 5, Pitfalls, item 3
	c := &proofClock{t: proofT0}
	l := newLimiter(c)
	lim := limit.Limits{TPM: 1000}
	r, err := l.reserve("k", lim, limit.Cost{Requests: 1, Tokens: 100})
	if err != nil {
		t.Fatal(err)
	}
	c.t = c.t.Add(3 * time.Minute) // the stream is still running
	l.fillTo(t, cadence)           // this call sweeps
	r.Settle(limit.Cost{Requests: 1, Tokens: 5000})
	if _, err := l.reserve("k", lim, limit.Cost{Requests: 1, Tokens: 10}); !isReject(err) {
		t.Fatalf("Reserve after the settle = %v, want a rejection: the overrun leaves -3900 tokens", err)
	}
}

func TestProofRetryAfterRoundsUp(t *testing.T) {
	// WHY: Retry-After is whole seconds rounded UP and at least 1 (the
	//      contract): a client that waits the advertised time must fit. 1.2 s
	//      is 2, 300 ms is 1, exactly 2 s is 2, and 0 is 1.
	// KIND: boundary
	// CATCHES: s04
	// CHAPTER: craft.08 section 5, Pitfalls, item 4
	for _, tc := range []struct {
		d    time.Duration
		want int
	}{{1200 * time.Millisecond, 2}, {300 * time.Millisecond, 1}, {2 * time.Second, 2}, {0, 1}} {
		if got := limit.RetryAfterSeconds(tc.d); got != tc.want {
			t.Errorf("RetryAfterSeconds(%v) = %d, want %d", tc.d, got, tc.want)
		}
	}
}

func TestProofSweepRunsOncePerCadence(t *testing.T) {
	// WHY: a sweep walks every key under the global lock, so the PR runs it
	//      on every 256th call only. An idle key that becomes evictable after
	//      call 256 must survive call 257 and be gone after call 512.
	// KIND: regression
	// CATCHES: s05
	// CHAPTER: craft.08 section 5, Pitfalls, item 5
	c := &proofClock{t: proofT0}
	l := newLimiter(c)
	r, err := l.reserve("a", limit.Limits{}, limit.Cost{Requests: 1})
	if err != nil {
		t.Fatal(err)
	}
	r.Settle(limit.Cost{Requests: 1})
	l.fillTo(t, cadence) // sweeps, but "a" is not idle yet
	c.t = c.t.Add(3 * time.Minute)
	if _, err := l.reserve("b", limit.Limits{}, limit.Cost{Requests: 1}); err != nil {
		t.Fatal(err)
	}
	if n := l.Len(); n != 3 {
		t.Fatalf("Len() = %d after call %d, want 3 (a, filler, b): call 257 must not sweep", n, l.calls)
	}
	l.fillTo(t, 2*cadence)
	if n := l.Len(); n != 2 {
		t.Fatalf("Len() = %d after call %d, want 2: the 512th call sweeps the idle key a", n, l.calls)
	}
}
