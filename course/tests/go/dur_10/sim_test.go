// The fault simulation of dur.10: three nodes on an in-process network that
// delays, reorders, and drops messages, with partitions and crash-restarts,
// all drawn from a frozen PCG32 seeded per run, so every failure names a seed
// that replays it exactly. Clients append unique values through whichever
// node they believe leads; an append is acknowledged when the node it was
// proposed to applies it. At the end the network heals, and the checks are
// the paper's safety properties plus linearizability of the client history.
package dur_10

import (
	"errors"
	"fmt"
	"sort"
	"testing"

	"tinyllm/durable/raft"
)

// op is one client append. Its value is unique, so its place in the final log
// identifies it.
type op struct {
	client int
	value  string
	invoke int // tick the client started it
	ret    int // tick it was acknowledged; -1 = never (abandoned or still open)
	pos    int // the position it was acknowledged at (1-based, client entries only)
}

type proposal struct {
	node        uint64
	incarnation int
	index, term uint64
	deadline    int
}

type client struct {
	cur    *op
	prop   *proposal
	target uint64
	next   int // ops started so far
	wait   int // ticks to wait before the next attempt
}

type inflight struct {
	m  raft.Message
	at int
}

type simConfig struct {
	ticks      int // ticks with faults
	healTicks  int // ticks after the heal
	clients    int
	dropPer    uint32 // a message is dropped with probability dropPer/1000
	partPer    uint32 // a partition change starts with probability partPer/1000 per tick
	crashPer   uint32 // a crash starts with probability crashPer/1000 per tick
	maxDelay   uint32 // a message is delivered 0..maxDelay-1 ticks after it is sent
	opDeadline int
}

type sim struct {
	t   *testing.T
	cfg simConfig
	rng *pcg32
	c   *cluster
	now int
	net []inflight

	incarnation map[uint64]int
	restartAt   map[uint64]int
	applied     map[uint64][]raft.Entry // this incarnation's applied entries, per node
	global      map[uint64]raft.Entry   // index -> the entry anyone applied there
	leaderOf    map[uint64]uint64       // term -> the node that led it
	clients     []*client
	ops         []*op
	faults      bool
}

func newSim(t *testing.T, seed uint64, cfg simConfig) *sim {
	s := &sim{
		t: t, cfg: cfg, rng: newPCG32(seed, 5),
		incarnation: map[uint64]int{}, restartAt: map[uint64]int{},
		applied: map[uint64][]raft.Entry{}, global: map[uint64]raft.Entry{}, leaderOf: map[uint64]uint64{},
		faults: true,
	}
	c := &cluster{
		t: t, nodes: map[uint64]*raft.Node{}, stores: map[uint64]*memStore{}, timeouts: map[uint64]int{},
		down: map[uint64]bool{}, cut: map[[2]uint64]bool{},
		cfg: raft.Config{ElectionTicks: 10, HeartbeatTicks: 2, MaxEntries: 8},
	}
	for id := uint64(1); id <= 3; id++ {
		c.ids = append(c.ids, id)
		c.stores[id] = &memStore{}
	}
	s.c = c
	for _, id := range c.ids {
		s.start(id)
	}
	for i := 0; i < cfg.clients; i++ {
		s.clients = append(s.clients, &client{target: uint64(1 + s.rng.below(3))})
	}
	return s
}

// start (re)creates a node from its store with seeded election timeouts.
func (s *sim) start(id uint64) {
	cfg := s.c.cfg
	cfg.ID = id
	cfg.Peers = []uint64{1, 2, 3}
	cfg.Timeout = func() int { return cfg.ElectionTicks + int(s.rng.below(uint32(cfg.ElectionTicks))) }
	n, err := raft.New(cfg, s.c.stores[id])
	if err != nil {
		s.t.Fatalf("raft.New(node %d): %v", id, err)
	}
	s.c.nodes[id] = n
	s.c.down[id] = false
	s.incarnation[id]++
	s.applied[id] = nil
}

func (s *sim) crash(id uint64) {
	s.c.nodes[id] = nil
	s.c.down[id] = true
	s.restartAt[id] = s.now + 10 + int(s.rng.below(60))
}

func (s *sim) leader() uint64 {
	var best uint64
	var bestTerm uint64
	for _, id := range s.c.ids {
		if n := s.c.nodes[id]; n != nil {
			if st := n.Status(); st.Role == raft.Leader && st.Term >= bestTerm {
				best, bestTerm = id, st.Term
			}
		}
	}
	return best
}

func (s *sim) fail(format string, args ...any) {
	s.t.Helper()
	s.t.Fatalf("tick %d: "+format, append([]any{s.now}, args...)...)
}

// send queues a node's outgoing messages with a delay, or drops them.
func (s *sim) send(ms []raft.Message) {
	for _, m := range ms {
		if s.faults && s.rng.chance(s.cfg.dropPer, 1000) {
			continue
		}
		delay := 0
		if s.faults && s.cfg.maxDelay > 1 {
			delay = int(s.rng.below(s.cfg.maxDelay))
		}
		s.net = append(s.net, inflight{m: m, at: s.now + delay})
	}
}

func (s *sim) faultStep() {
	if !s.faults {
		return
	}
	if s.rng.chance(s.cfg.partPer, 1000) {
		s.c.heal()
		switch s.rng.below(4) {
		case 0: // healed
		case 1:
			if l := s.leader(); l != 0 {
				s.c.isolate(l)
			}
		default:
			s.c.isolate(uint64(1 + s.rng.below(3)))
		}
	}
	down := 0
	for _, id := range s.c.ids {
		if s.c.down[id] {
			down++
		}
	}
	if down == 0 && s.rng.chance(s.cfg.crashPer, 1000) {
		id := uint64(1 + s.rng.below(3))
		if l := s.leader(); l != 0 && s.rng.chance(1, 2) {
			id = l
		}
		s.crash(id)
	}
}

func (s *sim) restarts() {
	for _, id := range s.c.ids {
		if s.c.down[id] && (s.now >= s.restartAt[id] || !s.faults) {
			s.start(id)
		}
	}
}

func (s *sim) clientStep() {
	for k, cl := range s.clients {
		if cl.wait > 0 {
			cl.wait--
			continue
		}
		if cl.cur == nil {
			if !s.faults {
				continue // no new work after the heal: only finish what is open
			}
			cl.next++
			cl.cur = &op{client: k, value: fmt.Sprintf("c%d-%d", k, cl.next), invoke: s.now, ret: -1}
			s.ops = append(s.ops, cl.cur)
		}
		if cl.prop != nil {
			continue
		}
		n := s.c.nodes[cl.target]
		if n == nil {
			cl.target = uint64(1 + s.rng.below(3))
			cl.wait = 2
			continue
		}
		idx, term, err := n.Propose([]byte(cl.cur.value))
		var nl *raft.NotLeaderError
		switch {
		case err == nil:
			cl.prop = &proposal{node: cl.target, incarnation: s.incarnation[cl.target], index: idx, term: term, deadline: s.now + s.cfg.opDeadline}
		case errors.As(err, &nl) && nl.Leader != 0:
			cl.target = nl.Leader
		case errors.As(err, &nl):
			cl.target = uint64(1 + s.rng.below(3))
			cl.wait = 3
		default:
			s.fail("Propose on node %d: %v", cl.target, err)
		}
		s.send(n.ReadMessages())
	}
}

// resolve acknowledges or releases each client's proposal.
func (s *sim) resolve() {
	for _, cl := range s.clients {
		p := cl.prop
		if p == nil {
			continue
		}
		if s.incarnation[p.node] != p.incarnation || s.c.down[p.node] || s.now > p.deadline {
			// In doubt: the value may or may not commit. The client gives up
			// on it and moves on; the final log decides.
			cl.prop, cl.cur = nil, nil
			continue
		}
		app := s.applied[p.node]
		if uint64(len(app)) < p.index {
			continue
		}
		e := app[p.index-1]
		if e.Term == p.term && string(e.Data) == cl.cur.value {
			pos := 0
			for _, x := range app[:p.index] {
				if len(x.Data) > 0 {
					pos++
				}
			}
			cl.cur.ret, cl.cur.pos = s.now, pos
			cl.prop, cl.cur = nil, nil
			continue
		}
		cl.prop = nil // another entry committed at that index: the proposal was lost, retry
	}
}

func (s *sim) deliverDue() {
	var due, later []inflight
	for _, f := range s.net {
		if f.at <= s.now {
			due = append(due, f)
		} else {
			later = append(later, f)
		}
	}
	s.net = later
	for i := len(due) - 1; i > 0; i-- { // reorder what arrives in the same tick
		j := int(s.rng.below(uint32(i + 1)))
		due[i], due[j] = due[j], due[i]
	}
	for _, f := range due {
		m := f.m
		if s.c.down[m.To] || s.c.cut[[2]uint64{m.From, m.To}] {
			continue
		}
		n := s.c.nodes[m.To]
		if err := n.Step(m); err != nil {
			s.fail("node %d Step: %v", m.To, err)
		}
		s.send(n.ReadMessages())
	}
}

// observe applies committed entries and checks election safety and state
// machine safety as they happen.
func (s *sim) observe() {
	for _, id := range s.c.ids {
		n := s.c.nodes[id]
		if n == nil {
			continue
		}
		st := n.Status()
		if st.Role == raft.Leader {
			if prev, ok := s.leaderOf[st.Term]; ok && prev != id {
				s.fail("election safety: nodes %d and %d both led term %d", prev, id, st.Term)
			}
			s.leaderOf[st.Term] = id
		}
		for _, e := range n.ReadCommitted() {
			want := uint64(len(s.applied[id])) + 1
			if e.Index != want {
				s.fail("node %d applied index %d, want %d (in order, no gaps)", id, e.Index, want)
			}
			if g, ok := s.global[e.Index]; ok && (g.Term != e.Term || string(g.Data) != string(e.Data)) {
				s.fail("state machine safety: index %d applied as (t%d, %q) and as (t%d, %q)", e.Index, g.Term, g.Data, e.Term, e.Data)
			}
			s.global[e.Index] = e
			s.applied[id] = append(s.applied[id], e)
		}
	}
}

func (s *sim) step() {
	s.now++
	s.faultStep()
	s.restarts()
	s.clientStep()
	for _, id := range s.c.ids {
		if n := s.c.nodes[id]; n != nil {
			if err := n.Tick(); err != nil {
				s.fail("node %d Tick: %v", id, err)
			}
			s.send(n.ReadMessages())
		}
	}
	s.deliverDue()
	s.observe()
	s.resolve()
}

// run executes the fault phase, heals, and returns the final committed log.
func (s *sim) run() []raft.Entry {
	for i := 0; i < s.cfg.ticks; i++ {
		s.step()
	}
	s.faults = false
	s.c.heal()
	for i := 0; i < s.cfg.healTicks; i++ {
		s.step()
	}
	s.c.checkStores()
	l := s.leader()
	if l == 0 {
		s.fail("liveness: no leader %d ticks after the network healed", s.cfg.healTicks)
	}
	ls := s.c.nodes[l].Status()
	for _, id := range s.c.ids {
		if st := s.c.nodes[id].Status(); st.Commit != ls.LastIndex || st.Term != ls.Term {
			s.fail("liveness: %d ticks after the heal node %d has %+v, the leader %+v", s.cfg.healTicks, id, st, ls)
		}
	}
	for _, cl := range s.clients {
		if cl.cur != nil {
			s.fail("liveness: client %d still waits on %s after the heal", cl.cur.client, cl.cur.value)
		}
	}
	return cloneEntries(s.c.stores[l].ents[:ls.Commit])
}

// checkHistory: every acknowledged append is in the final log at the position
// it was acknowledged with, no value appears twice, and the history is
// linearizable (see linearizable).
func checkHistory(ops []*op, final []raft.Entry) error {
	posOf := map[string]int{}
	pos := 0
	for _, e := range final {
		if len(e.Data) == 0 {
			continue
		}
		pos++
		if _, dup := posOf[string(e.Data)]; dup {
			return fmt.Errorf("value %q committed twice", e.Data)
		}
		posOf[string(e.Data)] = pos
	}
	var hist []histOp
	for _, o := range ops {
		p, in := posOf[o.value]
		if o.ret >= 0 {
			if !in {
				return fmt.Errorf("acknowledged append %s (position %d, tick %d) is not in the final log", o.value, o.pos, o.ret)
			}
			if p != o.pos {
				return fmt.Errorf("append %s was acknowledged at position %d but sits at %d", o.value, o.pos, p)
			}
		}
		if in {
			ret := o.ret
			if ret < 0 {
				ret = 1 << 30 // never acknowledged: it may take effect any time after its invocation
			}
			hist = append(hist, histOp{name: o.value, invoke: o.invoke, ret: ret, pos: p})
		}
	}
	if len(hist) != pos {
		return fmt.Errorf("the final log holds %d client values but the clients issued only %d of them", pos, len(hist))
	}
	return linearizable(hist)
}

type histOp struct {
	name        string
	invoke, ret int
	pos         int
}

// linearizable decides whether a history of appends to one log, each
// returning its position, has a linearization: a total order that respects
// real time (an op that returned before another was invoked comes first)
// and in which the k-th append returns position k. With positions as outputs
// the order is forced, so the Wing and Gong search of a general checker
// (Porcupine) reduces to one pass: pick each op's linearization point as
// early as possible, no earlier than its invocation or the previous point,
// and fail if that is after its return.
func linearizable(h []histOp) error {
	sort.Slice(h, func(i, j int) bool { return h[i].pos < h[j].pos })
	point := -1
	for k, o := range h {
		if o.pos != k+1 {
			return fmt.Errorf("positions are not 1..%d: %d at rank %d", len(h), o.pos, k+1)
		}
		point = max(point, o.invoke)
		if point > o.ret {
			return fmt.Errorf("not linearizable: %s (invoked %d, returned %d) holds position %d, but an op before it in the log was invoked at %d",
				o.name, o.invoke, o.ret, o.pos, point)
		}
	}
	return nil
}

// -- tests ------------------------------------------------------------------------------

var faulty = simConfig{
	ticks: 1500, healTicks: 300, clients: 3,
	dropPer: 50, partPer: 8, crashPer: 4, maxDelay: 3, opDeadline: 40,
}

func TestSimulatedFaultsKeepSafetyAndLinearizability(t *testing.T) {
	// WHY: the whole protocol under the faults the design names: partitions
	//      (the leader cut off included), dropped and reordered messages, and
	//      crash-restarts from storage, over 40 seeded runs. Checked on every
	//      tick: one leader per term, and no index applied with two different
	//      entries. Checked at the end: the cluster recovers once healed, every
	//      acknowledged append survives at its acknowledged position, and the
	//      client history is linearizable.
	// KIND: fault, property
	// CATCHES: s01, s02, s04, s05, s06, s08, s10, s11, s15, s16, s17
	// CHAPTER: dur.10 section 4, what the tests check
	acked, total := 0, 0
	for seed := uint64(1); seed <= 40; seed++ {
		s := newSim(t, seed, faulty)
		final := s.run()
		if err := checkHistory(s.ops, final); err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		terms := len(s.leaderOf)
		if terms < 3 {
			t.Fatalf("seed %d: only %d leadership terms: the faults never forced an election", seed, terms)
		}
		for _, o := range s.ops {
			total++
			if o.ret >= 0 {
				acked++
			}
		}
	}
	t.Logf("%d of %d appends acknowledged over 40 runs", acked, total)
	if acked < total/2 {
		t.Fatalf("only %d of %d appends were acknowledged: the simulation is not exercising the protocol", acked, total)
	}
}

func TestLeaderKillAckedAppendsSurvive(t *testing.T) {
	// WHY: the HA promise in one scenario: 20 appends acknowledged by the
	//      leader, the leader killed, a new leader elected by the other two,
	//      and all 20 still committed, in order; the old leader restarts from
	//      its storage and catches up to the same log.
	// KIND: fault
	// CATCHES: s16, s17
	// CHAPTER: dur.10 section 3; section 4
	c := newCluster(t, []int{10, 15, 20}, 3, nil)
	lead := c.runUntilLeader(40)
	var vals []string
	for i := 0; i < 20; i++ {
		v := fmt.Sprintf("v%02d", i)
		vals = append(vals, v)
		idx, term, err := c.node(lead).Propose([]byte(v))
		if err != nil {
			t.Fatal(err)
		}
		c.settle()
		got := c.node(lead).ReadCommitted()
		if len(got) == 0 || got[len(got)-1].Index != idx || got[len(got)-1].Term != term {
			t.Fatalf("append %s not acknowledged right after replication: %s", v, fmtEntries(got))
		}
	}
	c.crash(lead)
	var newLead uint64
	for i := 0; i < 60 && newLead == 0; i++ {
		c.tick()
		c.settle()
		for _, ids := range c.leaders() {
			newLead = ids[0]
		}
	}
	if newLead == 0 {
		t.Fatal("the two survivors did not elect a leader")
	}
	var got []string
	for _, e := range c.node(newLead).ReadCommitted() {
		if len(e.Data) > 0 {
			got = append(got, string(e.Data))
		}
	}
	if fmt.Sprint(got) != fmt.Sprint(vals) {
		t.Fatalf("new leader %d committed %v, want the 20 acknowledged appends %v", newLead, got, vals)
	}
	c.start(lead)
	for i := 0; i < 20; i++ {
		c.tick()
		c.settle()
	}
	if !sameEntries(c.stores[lead].ents, c.stores[newLead].ents) {
		t.Fatalf("the restarted old leader's log %s differs from the new leader's %s",
			fmtEntries(c.stores[lead].ents), fmtEntries(c.stores[newLead].ents))
	}
	c.checkStores()
}

func TestLinearizabilityCheckerRejectsStaleOrder(t *testing.T) {
	// WHY: the checker is only worth what it rejects. b returned (tick 5)
	//      before a was invoked (tick 7), so b must come first in the log; a
	//      history where a holds position 1 is not linearizable. Overlapping
	//      ops may take either order.
	// KIND: unit
	bad := []histOp{{"a", 7, 9, 1}, {"b", 1, 5, 2}}
	if err := linearizable(bad); err == nil {
		t.Fatal("the checker accepted a history that violates real-time order")
	}
	ok := []histOp{{"a", 1, 9, 2}, {"b", 2, 5, 1}}
	if err := linearizable(ok); err != nil {
		t.Fatalf("overlapping ops may linearize in either order: %v", err)
	}
	if err := linearizable([]histOp{{"a", 1, 2, 2}}); err == nil {
		t.Fatal("the checker accepted a gap in positions")
	}
}
