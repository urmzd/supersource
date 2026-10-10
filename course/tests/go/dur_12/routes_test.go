package dur_12

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"tinyllm/activities"
)

var ctx = context.Background()

func TestHandExampleCanaryWeights(t *testing.T) {
	// WHY: section 3's numbers. One backend at 1 becomes 0.9 and the canary
	//      0.1; an even split 0.5/0.5 becomes 0.45/0.45 and 0.1. The weights
	//      still sum to 1, which the gateway requires; a canary already in
	//      the list is replaced, not added twice.
	// KIND: unit
	// CATCHES: s23
	// CHAPTER: dur.12 section 3, worked example
	got := activities.CanaryBackends([]activities.Backend{{ServedModel: "v1", Weight: 1}}, "v2", 0.1)
	want := []activities.Backend{{ServedModel: "v1", Weight: 0.9}, {ServedModel: "v2", Weight: 0.1}}
	if !activities.SameBackends(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	got = activities.CanaryBackends([]activities.Backend{{ServedModel: "a", Weight: 0.5}, {ServedModel: "b", Weight: 0.5}}, "v2", 0.1)
	want = []activities.Backend{{ServedModel: "a", Weight: 0.45}, {ServedModel: "b", Weight: 0.45}, {ServedModel: "v2", Weight: 0.1}}
	if !activities.SameBackends(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	got = activities.CanaryBackends([]activities.Backend{{ServedModel: "v1", Weight: 0.8}, {ServedModel: "v2", Weight: 0.2}}, "v2", 0.1)
	want = []activities.Backend{{ServedModel: "v1", Weight: 0.9}, {ServedModel: "v2", Weight: 0.1}}
	if !activities.SameBackends(got, want) {
		t.Fatalf("re-weighting an existing canary: got %v, want %v", got, want)
	}
	if activities.SameBackends(want, []activities.Backend{{ServedModel: "v1", Weight: 0.9}}) ||
		activities.SameBackends(want, []activities.Backend{{ServedModel: "v1", Weight: 0.9}, {ServedModel: "v3", Weight: 0.1}}) {
		t.Fatal("SameBackends ignores a missing or different backend")
	}
}

func TestImplicitBackendIsTheRouteModel(t *testing.T) {
	// WHY: a route with no backends field sends everything to the model of
	//      its own name (admin.v1.yaml). The canary must keep that share
	//      explicitly, or the old model loses its traffic entirely.
	// KIND: boundary
	// CATCHES: s24
	// CHAPTER: dur.12 section 5, Pitfalls
	b, err := activities.BackendsOf(map[string]any{"model": "tinystories"})
	if err != nil || len(b) != 1 || b[0].ServedModel != "tinystories" || b[0].Weight != 1 {
		t.Fatalf("BackendsOf a route without backends: %v %v", b, err)
	}
	gw := newGateway(t, twoRoutes, "tinystories-10m-v2")
	res, err := gw.client().Canary(ctx, activities.CanaryInput{Route: "tinystories", Served: "tinystories-10m-v2", Weight: 0.25})
	if err != nil || !res.Changed {
		t.Fatalf("canary: %+v %v", res, err)
	}
	got := backendsOf(t, gw.route(t, "tinystories"))
	if len(got) != 2 || !approx(got["tinystories"], 0.75) || !approx(got["tinystories-10m-v2"], 0.25) {
		t.Fatalf("route backends %v, want tinystories 0.75 and the canary 0.25", got)
	}
}

func TestPutCarriesTheETag(t *testing.T) {
	// WHY: the route table is guarded by optimistic concurrency: every PUT
	//      sends If-Match with the ETag of the GET it was computed from (428
	//      without it, 412 with a stale one).
	// KIND: conformance
	// CATCHES: s18
	// CHAPTER: dur.12 section 2, optimistic concurrency
	gw := newGateway(t, twoRoutes, "tinystories-10m-v2")
	if _, err := gw.client().Canary(ctx, activities.CanaryInput{Route: "tinystories", Served: "tinystories-10m-v2", Weight: 0.1}); err != nil {
		t.Fatal(err)
	}
	if len(gw.ifMatch) != 1 || gw.ifMatch[0] != `"routes-7"` || gw.puts != 1 {
		t.Fatalf("If-Match headers %q, puts %d; want one PUT with \"routes-7\"", gw.ifMatch, gw.puts)
	}
}

func TestStaleETagRereadsAndKeepsTheOtherChange(t *testing.T) {
	// WHY: an operator adds a route between our GET and our PUT. The PUT
	//      gets 412; the activity must read again and reapply its change on
	//      the new table, so both changes survive. Giving up loses the
	//      canary; retrying the old body would erase the operator's route.
	// KIND: fault
	// CATCHES: s19, s21
	// CHAPTER: dur.12 section 5, Pitfalls
	gw := newGateway(t, twoRoutes, "tinystories-10m-v2")
	gw.beforePut = func(g *gateway) {
		g.routes = append(g.routes, json.RawMessage(`{"model":"added-by-operator"}`))
		g.epoch++
	}
	res, err := gw.client().Canary(ctx, activities.CanaryInput{Route: "tinystories", Served: "tinystories-10m-v2", Weight: 0.1})
	if err != nil || !res.Changed {
		t.Fatalf("canary after a concurrent edit: %+v %v", res, err)
	}
	gw.route(t, "added-by-operator")
	if b := backendsOf(t, gw.route(t, "tinystories")); !approx(b["tinystories-10m-v2"], 0.1) {
		t.Fatalf("canary lost: %v", b)
	}
	if res.ETag != `"routes-9"` {
		t.Fatalf("result ETag %q, want the ETag of our PUT, \"routes-9\"", res.ETag)
	}
}

func TestOtherRoutesAreUntouched(t *testing.T) {
	// WHY: PUT replaces the whole table. Every route the activity did not
	//      change must go back byte for byte, unknown fields included (a
	//      cascade with accept_if, an x-owner tag): decoding the table into a
	//      struct that lacks a field deletes it from production.
	// KIND: regression
	// CATCHES: s20, s21
	// CHAPTER: dur.12 section 5, Pitfalls
	gw := newGateway(t, twoRoutes, "tinystories-10m-v2")
	other := gw.rawRoute(t, "smart")
	if _, err := gw.client().Canary(ctx, activities.CanaryInput{Route: "tinystories", Served: "tinystories-10m-v2", Weight: 0.1}); err != nil {
		t.Fatal(err)
	}
	if got := gw.rawRoute(t, "smart"); got != other {
		t.Fatalf("the cascade route changed:\n  %s\nwas\n  %s", got, other)
	}
	r := gw.route(t, "tinystories")
	if r["x-owner"] != "ml-team" || len(r["aliases"].([]any)) != 1 {
		t.Fatalf("the changed route lost its other fields: %v", r)
	}
}

func TestCanaryIsIdempotent(t *testing.T) {
	// WHY: an activity can run twice (the worker died after the PUT). The
	//      second run finds the weight in place and does not PUT again, so
	//      0.9/0.1 never becomes 0.81/0.09/0.1.
	// KIND: property
	// CATCHES: s22
	// CHAPTER: dur.12 section 2, idempotent activities
	gw := newGateway(t, twoRoutes, "tinystories-10m-v2")
	in := activities.CanaryInput{Route: "tinystories", Served: "tinystories-10m-v2", Weight: 0.1}
	first, err1 := gw.client().Canary(ctx, in)
	second, err2 := gw.client().Canary(ctx, in)
	if err1 != nil || err2 != nil || !first.Changed || second.Changed || gw.puts != 1 {
		t.Fatalf("changed %v then %v, %d PUTs (%v %v); want true, false, 1", first.Changed, second.Changed, gw.puts, err1, err2)
	}
	if !activities.SameBackends(second.Backends, first.Backends) {
		t.Fatalf("second result %v, first %v", second.Backends, first.Backends)
	}
}

func TestCanaryWaitsForAServingWorker(t *testing.T) {
	// WHY: routing 10% of traffic to a model no engine serves fails 10% of
	//      requests. With no live worker for it (or only a draining one) the
	//      canary refuses with a retryable error, so the activity's retries
	//      wait for the engines to come up.
	// KIND: fault
	// CATCHES: s25
	// CHAPTER: dur.12 section 5, Pitfalls
	gw := newGateway(t, twoRoutes, "tinystories-10m-v1")
	gw.workers = append(gw.workers, map[string]any{"worker_id": "w9", "model": "tinystories-10m-v2", "draining": true})
	_, err := gw.client().Canary(ctx, activities.CanaryInput{Route: "tinystories", Served: "tinystories-10m-v2", Weight: 0.1})
	if !errors.Is(err, activities.ErrNoWorkers) || nonRetryable(err) {
		t.Fatalf("error %v, want a retryable ErrNoWorkers", err)
	}
	if gw.puts != 0 {
		t.Fatal("the canary was applied with no serving worker")
	}
}

func TestUnknownRouteAndBadInputArePermanent(t *testing.T) {
	// WHY: a route that does not exist, a weight outside (0, 1), and a table
	//      the gateway rejects (400) do not get better with retries: each is
	//      non-retryable, so the release fails at once.
	// KIND: boundary
	// CATCHES: s26
	// CHAPTER: dur.12 section 4, the interface
	gw := newGateway(t, twoRoutes, "tinystories-10m-v2")
	_, err := gw.client().Canary(ctx, activities.CanaryInput{Route: "nope", Served: "tinystories-10m-v2", Weight: 0.1})
	if !errors.Is(err, activities.ErrNoRoute) || !nonRetryable(err) {
		t.Fatalf("unknown route: %v, want a non-retryable ErrNoRoute", err)
	}
	if _, err = gw.client().Canary(ctx, activities.CanaryInput{Route: "tinystories", Served: "tinystories-10m-v2", Weight: 1}); !nonRetryable(err) {
		t.Fatalf("weight 1: %v, want non-retryable", err)
	}
	gw.reject = true
	if _, err = gw.client().Canary(ctx, activities.CanaryInput{Route: "tinystories", Served: "tinystories-10m-v2", Weight: 0.1}); !nonRetryable(err) {
		t.Fatalf("a 400 from the gateway: %v, want non-retryable", err)
	}
}

func TestPromoteAndRestore(t *testing.T) {
	// WHY: promote leaves only the new model at weight 1; restore puts the
	//      snapshot back exactly (including fields the canary never touched)
	//      and is idempotent too.
	// KIND: unit
	// CATCHES: s27
	// CHAPTER: dur.12 section 4, the interface
	gw := newGateway(t, twoRoutes, "tinystories-10m-v2")
	c := gw.client()
	snap, err := c.Snapshot(ctx, activities.SnapshotInput{Route: "tinystories"})
	if err != nil || !strings.Contains(string(snap.Route), `"x-owner"`) || snap.ETag != `"routes-7"` {
		t.Fatalf("snapshot %s %q %v", snap.Route, snap.ETag, err)
	}
	if _, err := c.Canary(ctx, activities.CanaryInput{Route: "tinystories", Served: "tinystories-10m-v2", Weight: 0.1}); err != nil {
		t.Fatal(err)
	}
	r1, err := c.Restore(ctx, activities.RestoreInput{Route: "tinystories", Previous: snap.Route})
	r2, err2 := c.Restore(ctx, activities.RestoreInput{Route: "tinystories", Previous: snap.Route})
	if err != nil || err2 != nil || !r1.Changed || r2.Changed {
		t.Fatalf("restore: %+v %v then %+v %v", r1, err, r2, err2)
	}
	if !jsonEqual(t, gw.rawRoute(t, "tinystories"), string(snap.Route)) {
		t.Fatalf("restored %s, snapshot %s", gw.rawRoute(t, "tinystories"), snap.Route)
	}
	p, err := c.Promote(ctx, activities.PromoteInput{Route: "tinystories", Served: "tinystories-10m-v2"})
	b := backendsOf(t, gw.route(t, "tinystories"))
	if err != nil || !p.Changed || len(b) != 1 || b["tinystories-10m-v2"] != 1 {
		t.Fatalf("promote: %v %v, route %v", p, err, b)
	}
	if p2, _ := c.Promote(ctx, activities.PromoteInput{Route: "tinystories", Served: "tinystories-10m-v2"}); p2.Changed {
		t.Fatal("a second promote changed the table again")
	}
	if _, err := c.Restore(ctx, activities.RestoreInput{Route: "tinystories", Previous: json.RawMessage(`{"model":"smart"}`)}); !nonRetryable(err) {
		t.Fatalf("restoring another route's snapshot: %v, want non-retryable", err)
	}
}
