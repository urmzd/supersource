// Shared helpers of the dur.10 course tests: an in-memory Storage that
// survives a "crash" (the Node is thrown away, the store is kept), a cluster
// whose messages the test delivers one by one or all at once, and a frozen
// PCG32 for the seeded simulation (course tests never import math/rand, D35).
package dur_10

import (
	"bytes"
	"errors"
	"fmt"
	"slices"
	"testing"

	"tinyllm/durable/raft"
)

// -- storage ----------------------------------------------------------------------

// memStore is the stable storage of one node. It checks the Storage contract
// as it goes (contiguous entries, no gap), counts writes, and can be told to
// fail its next write.
type memStore struct {
	hs        raft.HardState
	ents      []raft.Entry
	hsSaves   int
	appends   int
	failNext  error
	violation error // the first contract violation a node committed against this store
}

func (s *memStore) Load() (raft.HardState, []raft.Entry, error) {
	return s.hs, cloneEntries(s.ents), nil
}

func (s *memStore) SaveHardState(h raft.HardState) error {
	if err := s.takeFailure(); err != nil {
		return err
	}
	s.hs = h
	s.hsSaves++
	return nil
}

func (s *memStore) Append(es []raft.Entry) error {
	if err := s.takeFailure(); err != nil {
		return err
	}
	if len(es) == 0 {
		s.violate(errors.New("Append called with no entries"))
		return nil
	}
	first := es[0].Index
	if first == 0 || first > uint64(len(s.ents))+1 {
		s.violate(fmt.Errorf("Append starts at index %d, the store holds 1..%d (a gap)", first, len(s.ents)))
		return nil
	}
	for i, e := range es {
		if e.Index != first+uint64(i) {
			s.violate(fmt.Errorf("Append entries are not contiguous: %d after %d", e.Index, first+uint64(i)-1))
			return nil
		}
	}
	s.ents = append(s.ents[:first-1], cloneEntries(es)...)
	s.appends++
	return nil
}

func (s *memStore) takeFailure() error {
	err := s.failNext
	s.failNext = nil
	return err
}

func (s *memStore) violate(err error) {
	if s.violation == nil {
		s.violation = err
	}
}

func cloneEntries(es []raft.Entry) []raft.Entry {
	out := make([]raft.Entry, len(es))
	for i, e := range es {
		out[i] = raft.Entry{Index: e.Index, Term: e.Term, Data: slices.Clone(e.Data)}
	}
	return out
}

// ents builds a log from terms: ents(1, 1, 2) is (1,t1) (2,t1) (3,t2), each
// with Data "i@t" so two entries are equal only when index and term are.
func ents(terms ...uint64) []raft.Entry {
	out := make([]raft.Entry, len(terms))
	for i, t := range terms {
		out[i] = raft.Entry{Index: uint64(i + 1), Term: t, Data: []byte(fmt.Sprintf("%d@%d", i+1, t))}
	}
	return out
}

func sameEntries(a, b []raft.Entry) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Index != b[i].Index || a[i].Term != b[i].Term || !bytes.Equal(a[i].Data, b[i].Data) {
			return false
		}
	}
	return true
}

func fmtEntries(es []raft.Entry) string {
	var b bytes.Buffer
	b.WriteString("[")
	for i, e := range es {
		if i > 0 {
			b.WriteString(" ")
		}
		fmt.Fprintf(&b, "(%d,t%d,%q)", e.Index, e.Term, e.Data)
	}
	b.WriteString("]")
	return b.String()
}

// -- a cluster driven by hand ---------------------------------------------------------

type cluster struct {
	t        testing.TB
	ids      []uint64
	nodes    map[uint64]*raft.Node
	stores   map[uint64]*memStore
	timeouts map[uint64]int
	cfg      raft.Config // template: ID and Peers are filled per node
	inbox    []raft.Message
	down     map[uint64]bool
	cut      map[[2]uint64]bool // directed links that drop messages
}

// newCluster starts n nodes with fixed election timeouts (timeouts[i] for
// node i+1). Stores may be preloaded through pre (nil = empty).
func newCluster(t testing.TB, timeouts []int, heartbeat int, pre map[uint64]*memStore) *cluster {
	t.Helper()
	c := &cluster{
		t: t, nodes: map[uint64]*raft.Node{}, stores: map[uint64]*memStore{},
		timeouts: map[uint64]int{}, down: map[uint64]bool{}, cut: map[[2]uint64]bool{},
		cfg: raft.Config{ElectionTicks: 10, HeartbeatTicks: heartbeat},
	}
	for i, to := range timeouts {
		id := uint64(i + 1)
		c.ids = append(c.ids, id)
		c.timeouts[id] = to
		if s, ok := pre[id]; ok {
			c.stores[id] = s
		} else {
			c.stores[id] = &memStore{}
		}
	}
	for _, id := range c.ids {
		c.start(id)
	}
	return c
}

func (c *cluster) config(id uint64) raft.Config {
	cfg := c.cfg
	cfg.ID = id
	cfg.Peers = slices.Clone(c.ids)
	to := c.timeouts[id]
	cfg.Timeout = func() int { return to }
	return cfg
}

// start (re)creates node id from its store: a crash loses everything else.
func (c *cluster) start(id uint64) {
	c.t.Helper()
	n, err := raft.New(c.config(id), c.stores[id])
	if err != nil {
		c.t.Fatalf("raft.New(node %d): %v", id, err)
	}
	c.nodes[id] = n
	c.down[id] = false
}

func (c *cluster) crash(id uint64) {
	c.down[id] = true
	c.nodes[id] = nil
	kept := c.inbox[:0]
	for _, m := range c.inbox {
		if m.To != id && m.From != id {
			kept = append(kept, m)
		}
	}
	c.inbox = kept
}

func (c *cluster) node(id uint64) *raft.Node {
	c.t.Helper()
	n := c.nodes[id]
	if n == nil {
		c.t.Fatalf("node %d is down", id)
	}
	return n
}

func (c *cluster) st(id uint64) raft.Status { return c.node(id).Status() }

// collect moves every queued outgoing message into the inbox.
func (c *cluster) collect() {
	for _, id := range c.ids {
		if n := c.nodes[id]; n != nil {
			c.inbox = append(c.inbox, n.ReadMessages()...)
		}
	}
}

// tick advances the listed nodes (all when none) by one tick each.
func (c *cluster) tick(ids ...uint64) {
	c.t.Helper()
	if len(ids) == 0 {
		ids = c.ids
	}
	for _, id := range ids {
		if n := c.nodes[id]; n != nil {
			if err := n.Tick(); err != nil {
				c.t.Fatalf("node %d Tick: %v", id, err)
			}
		}
	}
	c.collect()
}

// take removes and returns the queued messages that match (from, to, type);
// 0 matches any id and any type.
func (c *cluster) take(from, to uint64, typ raft.MsgType) []raft.Message {
	c.collect()
	var got, kept []raft.Message
	for _, m := range c.inbox {
		if (from == 0 || m.From == from) && (to == 0 || m.To == to) && (typ == 0 || m.Type == typ) {
			got = append(got, m)
		} else {
			kept = append(kept, m)
		}
	}
	c.inbox = kept
	return got
}

// deliver steps each message into its recipient (dropping it when the
// recipient is down or the link is cut) and queues the replies.
func (c *cluster) deliver(ms ...raft.Message) {
	c.t.Helper()
	for _, m := range ms {
		if c.down[m.To] || c.down[m.From] || c.cut[[2]uint64{m.From, m.To}] {
			continue
		}
		if err := c.nodes[m.To].Step(m); err != nil {
			c.t.Fatalf("node %d Step(%v from %d): %v", m.To, m.Type, m.From, err)
		}
	}
	c.collect()
}

// settle delivers everything queued, and every reply, until no message is left.
func (c *cluster) settle() {
	c.t.Helper()
	for i := 0; ; i++ {
		c.collect()
		if len(c.inbox) == 0 {
			return
		}
		if i > 10000 {
			c.t.Fatalf("messages never stop: %d queued, e.g. %+v", len(c.inbox), c.inbox[0])
		}
		ms := c.inbox
		c.inbox = nil
		c.deliver(ms...)
	}
}

// isolate cuts every link to and from id; heal restores them all.
func (c *cluster) isolate(id uint64) {
	for _, o := range c.ids {
		if o != id {
			c.cut[[2]uint64{id, o}] = true
			c.cut[[2]uint64{o, id}] = true
		}
	}
}

func (c *cluster) heal() { c.cut = map[[2]uint64]bool{} }

// leaders returns the ids that believe they lead, by term.
func (c *cluster) leaders() map[uint64][]uint64 {
	out := map[uint64][]uint64{}
	for _, id := range c.ids {
		if n := c.nodes[id]; n != nil {
			if s := n.Status(); s.Role == raft.Leader {
				out[s.Term] = append(out[s.Term], id)
			}
		}
	}
	return out
}

// runUntilLeader ticks and settles until exactly one live node leads.
func (c *cluster) runUntilLeader(maxTicks int) uint64 {
	c.t.Helper()
	for i := 0; i < maxTicks; i++ {
		c.tick()
		c.settle()
		var lead []uint64
		for _, ids := range c.leaders() {
			lead = append(lead, ids...)
		}
		if len(lead) == 1 {
			return lead[0]
		}
	}
	c.t.Fatalf("no single leader after %d ticks: %v", maxTicks, c.leaders())
	return 0
}

func (c *cluster) checkStores() {
	c.t.Helper()
	for _, id := range c.ids {
		if v := c.stores[id].violation; v != nil {
			c.t.Fatalf("node %d broke the Storage contract: %v", id, v)
		}
	}
}

// -- frozen PCG32 ---------------------------------------------------------------------

// pcg32 transcribes course/tests/_lib/pcg32.py (spec/pcg32.md): PCG-XSH-RR
// 64/32, seeded as pcg32_srandom_r(seed, seq).
type pcg32 struct{ state, inc uint64 }

func newPCG32(seed, seq uint64) *pcg32 {
	p := &pcg32{inc: seq<<1 | 1}
	p.next()
	p.state += seed
	p.next()
	return p
}

func (p *pcg32) next() uint32 {
	old := p.state
	p.state = old*6364136223846793005 + p.inc
	xs := uint32(((old >> 18) ^ old) >> 27)
	rot := uint32(old >> 59)
	return xs>>rot | xs<<((-rot)&31)
}

// below is the unbiased draw of spec/pcg32.md: uniform in [0, n).
func (p *pcg32) below(n uint32) uint32 {
	t := -n % n
	for {
		if r := p.next(); r >= t {
			return r % n
		}
	}
}

// chance is true with probability num/den.
func (p *pcg32) chance(num, den uint32) bool { return p.below(den) < num }
