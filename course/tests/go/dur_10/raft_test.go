// Course tests for dur.10: Raft for the durable log (go/durable/raft).
//
// Every test here drives the nodes by hand: fixed election timeouts, one
// tick at a time, each message delivered when the test says so. Nothing is
// random and nothing sleeps, so each case is a scenario you can replay on
// paper with the chapter's figures. The seeded fault simulation is in
// sim_test.go.
package dur_10

import (
	"errors"
	"testing"

	"tinyllm/durable/raft"
)

func mustStep(t *testing.T, n *raft.Node, m raft.Message) {
	t.Helper()
	if err := n.Step(m); err != nil {
		t.Fatalf("Step(%v): %v", m.Type, err)
	}
}

func one(t *testing.T, ms []raft.Message, what string) raft.Message {
	t.Helper()
	if len(ms) != 1 {
		t.Fatalf("%s: want exactly 1 message, got %d: %+v", what, len(ms), ms)
	}
	return ms[0]
}

// -- the worked example --------------------------------------------------------------

func TestHandExampleElectionAndFirstCommit(t *testing.T) {
	// WHY: the chapter's worked example, tick by tick: timeouts 10, 15, 20 and
	//      heartbeat 3, so node 1 times out first, wins term 1 with node 2's
	//      vote, appends its no-op at index 1, commits it once one follower
	//      holds it, then commits a client entry at index 2, and the followers
	//      learn commit 2 from the next heartbeat.
	// KIND: unit
	// CATCHES: s08, s15, s17
	// CHAPTER: dur.10 section 3, worked example
	c := newCluster(t, []int{10, 15, 20}, 3, nil)

	for i := 0; i < 9; i++ {
		c.tick()
	}
	if got := c.take(0, 0, 0); len(got) != 0 {
		t.Fatalf("before tick 10 nobody may send anything; got %+v", got)
	}
	c.tick() // tick 10: node 1's timeout
	s1 := c.st(1)
	if s1.Role != raft.Candidate || s1.Term != 1 || s1.Vote != 1 {
		t.Fatalf("tick 10: node 1 = %+v, want a candidate in term 1 that voted for itself", s1)
	}
	if hs := c.stores[1].hs; hs != (raft.HardState{Term: 1, Vote: 1}) {
		t.Fatalf("node 1 asked for votes before saving term 1 and its own vote: store holds %+v", hs)
	}
	v2 := one(t, c.take(1, 2, raft.MsgVote), "RequestVote to node 2")
	v3 := one(t, c.take(1, 3, raft.MsgVote), "RequestVote to node 3")
	if v2.Term != 1 || v2.LastLogIndex != 0 || v2.LastLogTerm != 0 {
		t.Fatalf("RequestVote = %+v, want term 1, last log (0, 0)", v2)
	}

	c.deliver(v2) // node 2 grants
	if s2 := c.st(2); s2.Term != 1 || s2.Vote != 1 || s2.Role != raft.Follower {
		t.Fatalf("node 2 after the vote request = %+v, want a follower of term 1 that voted for 1", s2)
	}
	r2 := one(t, c.take(2, 1, raft.MsgVoteResp), "node 2's vote")
	if !r2.Granted || r2.Term != 1 {
		t.Fatalf("node 2's reply = %+v, want granted in term 1", r2)
	}
	if s1 := c.st(1); s1.Role != raft.Candidate {
		t.Fatalf("one vote (its own) of three is not a majority; node 1 = %+v", s1)
	}

	c.deliver(r2) // 2 votes of 3: node 1 leads term 1
	s1 = c.st(1)
	if s1.Role != raft.Leader || s1.Leader != 1 || s1.LastIndex != 1 || s1.LastTerm != 1 || s1.Commit != 0 {
		t.Fatalf("node 1 after a majority = %+v, want the leader of term 1 holding its no-op at index 1, commit 0", s1)
	}
	if !sameEntries(c.stores[1].ents, []raft.Entry{{Index: 1, Term: 1}}) {
		t.Fatalf("the leader's stored log = %s, want just the no-op (1,t1,\"\")", fmtEntries(c.stores[1].ents))
	}
	a2 := one(t, c.take(1, 2, raft.MsgApp), "AppendEntries to node 2")
	a3 := one(t, c.take(1, 3, raft.MsgApp), "AppendEntries to node 3")
	if a2.PrevLogIndex != 0 || a2.PrevLogTerm != 0 || a2.Commit != 0 || len(a2.Entries) != 1 || a2.Entries[0].Index != 1 || len(a2.Entries[0].Data) != 0 {
		t.Fatalf("first AppendEntries = %+v, want prev (0, 0), entries [no-op at 1], commit 0", a2)
	}

	c.deliver(v3) // node 3's vote arrives late: granted, and ignored by the leader
	c.deliver(c.take(3, 1, raft.MsgVoteResp)...)
	if s1 := c.st(1); s1.Role != raft.Leader || s1.Term != 1 {
		t.Fatalf("a late vote must not disturb the leader: %+v", s1)
	}

	c.deliver(a2)
	ok2 := one(t, c.take(2, 1, raft.MsgAppResp), "node 2's answer")
	if !ok2.Success || ok2.MatchIndex != 1 {
		t.Fatalf("node 2's answer = %+v, want success, match 1", ok2)
	}
	if s2 := c.st(2); s2.Leader != 1 || s2.LastIndex != 1 {
		t.Fatalf("node 2 = %+v, want it to follow 1 and hold index 1", s2)
	}
	c.deliver(ok2)
	if s1 := c.st(1); s1.Commit != 1 {
		t.Fatalf("leader + node 2 hold index 1 of term 1: a majority, so commit = 1; got %+v", s1)
	}
	if got := c.node(1).ReadCommitted(); !sameEntries(got, []raft.Entry{{Index: 1, Term: 1}}) {
		t.Fatalf("ReadCommitted = %s, want the no-op", fmtEntries(got))
	}
	c.deliver(a3)
	c.deliver(c.take(3, 1, raft.MsgAppResp)...)

	idx, term, err := c.node(1).Propose([]byte("x"))
	if err != nil || idx != 2 || term != 1 {
		t.Fatalf("Propose(x) = (%d, %d, %v), want (2, 1, nil)", idx, term, err)
	}
	p2 := one(t, c.take(1, 2, raft.MsgApp), "the replication of x to node 2")
	if p2.PrevLogIndex != 1 || p2.PrevLogTerm != 1 || p2.Commit != 1 || len(p2.Entries) != 1 || string(p2.Entries[0].Data) != "x" {
		t.Fatalf("AppendEntries for x = %+v, want prev (1, 1), entries [x at 2], commit 1", p2)
	}
	c.deliver(p2)
	c.deliver(c.take(2, 1, raft.MsgAppResp)...)
	if got := c.node(1).ReadCommitted(); len(got) != 1 || got[0].Index != 2 || string(got[0].Data) != "x" {
		t.Fatalf("after node 2 acknowledged x the leader must commit it: ReadCommitted = %s", fmtEntries(got))
	}
	if s2 := c.st(2); s2.Commit != 1 {
		t.Fatalf("node 2 learns commit 2 only from the next AppendEntries; it has %d", s2.Commit)
	}
	c.settle()
	for i := 0; i < 3; i++ { // one heartbeat interval
		c.tick(1)
	}
	c.settle()
	for _, id := range []uint64{2, 3} {
		if s := c.st(id); s.Commit != 2 {
			t.Fatalf("after the heartbeat node %d has commit %d, want 2", id, s.Commit)
		}
		got := c.node(id).ReadCommitted()
		if len(got) != 2 || string(got[1].Data) != "x" {
			t.Fatalf("node %d ReadCommitted = %s, want the no-op then x", id, fmtEntries(got))
		}
	}
	c.checkStores()
}

// -- elections ---------------------------------------------------------------------------

func TestVoteRequiresUpToDateLog(t *testing.T) {
	// WHY: election restriction (section 5.4.1): a voter whose last entry is
	//      (index 3, term 2) refuses a candidate whose log ends in an older
	//      term even when it is longer, and one with the same last term but a
	//      shorter log; it grants a later last term even when shorter. Without
	//      it a leader can be elected that lacks committed entries.
	// KIND: boundary
	// CATCHES: s01, s02
	// CHAPTER: dur.10 section 2.2; section 5, Pitfalls, item 1
	cases := []struct {
		name             string
		lastIdx, lastTrm uint64
		grant            bool
	}{
		{"older last term, longer log", 9, 1, false},
		{"same last term, shorter log", 2, 2, false},
		{"same last term, same length", 3, 2, true},
		{"same last term, longer log", 4, 2, true},
		{"later last term, shorter log", 1, 3, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := &memStore{hs: raft.HardState{Term: 2}, ents: ents(1, 2, 2)}
			n, err := raft.New(raft.Config{ID: 2, Peers: []uint64{1, 2, 3}, ElectionTicks: 10, HeartbeatTicks: 3, Timeout: func() int { return 10 }}, st)
			if err != nil {
				t.Fatal(err)
			}
			mustStep(t, n, raft.Message{Type: raft.MsgVote, From: 1, To: 2, Term: 3, LastLogIndex: tc.lastIdx, LastLogTerm: tc.lastTrm})
			r := one(t, n.ReadMessages(), "the vote reply")
			if r.Granted != tc.grant || r.Term != 3 {
				t.Fatalf("voter last (3, t2), candidate last (%d, t%d): reply %+v, want granted=%v in term 3",
					tc.lastIdx, tc.lastTrm, r, tc.grant)
			}
		})
	}
}

func TestOneVotePerTermSurvivesRestart(t *testing.T) {
	// WHY: a node votes at most once per term, and the vote is on disk before
	//      the reply leaves: a voter that forgot its vote in a restart could
	//      elect a second leader in the same term. The same candidate asking
	//      again (its reply was lost) gets the vote again.
	// KIND: fault
	// CATCHES: s03, s04
	// CHAPTER: dur.10 section 2.2; section 5, Pitfalls, item 2
	st := &memStore{}
	cfg := raft.Config{ID: 2, Peers: []uint64{1, 2, 3}, ElectionTicks: 10, HeartbeatTicks: 3, Timeout: func() int { return 10 }}
	n, err := raft.New(cfg, st)
	if err != nil {
		t.Fatal(err)
	}
	ask := func(n *raft.Node, from uint64) bool {
		mustStep(t, n, raft.Message{Type: raft.MsgVote, From: from, To: 2, Term: 1})
		return one(t, n.ReadMessages(), "the vote reply").Granted
	}
	if !ask(n, 1) {
		t.Fatal("the first candidate of term 1 must get the vote")
	}
	if st.hs != (raft.HardState{Term: 1, Vote: 1}) {
		t.Fatalf("the vote must be saved before the reply: store holds %+v", st.hs)
	}
	if ask(n, 3) {
		t.Fatal("a second candidate in the same term got a second vote")
	}
	if !ask(n, 1) {
		t.Fatal("the same candidate asking again must get the same vote")
	}
	n2, err := raft.New(cfg, st) // crash and restart
	if err != nil {
		t.Fatal(err)
	}
	if s := n2.Status(); s.Term != 1 || s.Vote != 1 {
		t.Fatalf("after a restart the node must remember term 1 and its vote: %+v", s)
	}
	if ask(n2, 3) {
		t.Fatal("after a restart the node voted again in term 1: two leaders are now possible")
	}
}

func TestMajorityOfThreeNeedsTwo(t *testing.T) {
	// WHY: a majority of 3 is 2. A node cut off from both peers must stay a
	//      candidate (one vote, its own), and a leader cut off from both
	//      followers must not commit what it alone holds.
	// KIND: boundary
	// CATCHES: s08
	// CHAPTER: dur.10 section 2.1; section 5, Pitfalls, item 6
	c := newCluster(t, []int{10, 15, 20}, 3, nil)
	c.isolate(1)
	for i := 0; i < 10; i++ {
		c.tick(1)
	}
	c.settle()
	if s := c.st(1); s.Role != raft.Candidate {
		t.Fatalf("an isolated node with its own vote only became %v", s.Role)
	}

	c2 := newCluster(t, []int{10, 15, 20}, 3, nil)
	lead := c2.runUntilLeader(40)
	c2.settle()
	c2.isolate(lead)
	commit := c2.st(lead).Commit
	if _, _, err := c2.node(lead).Propose([]byte("alone")); err != nil {
		t.Fatal(err)
	}
	c2.settle()
	if s := c2.st(lead); s.Commit != commit {
		t.Fatalf("a leader holding an entry alone committed it: commit %d -> %d", commit, s.Commit)
	}
}

func TestCandidateStepsDownForSameTermLeader(t *testing.T) {
	// WHY: two nodes time out together and both campaign in term 1; node 3
	//      votes for node 1, which wins. Node 2 is still a candidate in term 1
	//      when node 1's AppendEntries arrives: same term, so node 1 is this
	//      term's one leader, and node 2 must become its follower instead of
	//      campaigning again and bumping everyone to term 2.
	// KIND: unit
	// CATCHES: s04, s08, s10, s12, s15
	// CHAPTER: dur.10 section 2.2; section 5, Pitfalls, item 13
	c := newCluster(t, []int{10, 10, 30}, 3, nil)
	for i := 0; i < 10; i++ {
		c.tick()
	}
	c.deliver(c.take(1, 3, raft.MsgVote)...)
	c.deliver(c.take(2, 3, raft.MsgVote)...) // refused: node 3 voted for 1
	c.take(1, 2, raft.MsgVote)               // lost
	c.take(2, 1, raft.MsgVote)               // lost
	c.deliver(c.take(3, 0, raft.MsgVoteResp)...)
	if s := c.st(1); s.Role != raft.Leader || s.Term != 1 {
		t.Fatalf("node 1 = %+v, want the leader of term 1", s)
	}
	if s := c.st(2); s.Role != raft.Candidate || s.Term != 1 {
		t.Fatalf("setup: node 2 = %+v, want still a candidate in term 1", s)
	}
	c.deliver(c.take(1, 2, raft.MsgApp)...)
	if s := c.st(2); s.Role != raft.Follower || s.Leader != 1 || s.Term != 1 {
		t.Fatalf("node 2 after the term-1 leader's AppendEntries = %+v, want a follower of 1 in term 1", s)
	}
	c.settle()
	for i := 0; i < 60; i++ {
		c.tick()
		c.settle()
	}
	if s := c.st(1); s.Role != raft.Leader || s.Term != 1 {
		t.Fatalf("60 ticks later node 1 = %+v: the old candidate disrupted the leader", s)
	}
}

func TestHigherTermReplyMakesLeaderStepDown(t *testing.T) {
	// WHY: any message with a higher term, a reply included, means this
	//      node's term is over (figure 2, rules for all servers). A leader that
	//      ignores a reply from term 7 keeps sending into a cluster that has
	//      moved on.
	// KIND: unit
	// CATCHES: s09
	// CHAPTER: dur.10 section 2.1; section 5, Pitfalls, item 7
	c := newCluster(t, []int{10, 15, 20}, 3, nil)
	lead := c.runUntilLeader(40)
	c.settle()
	n := c.node(lead)
	term := n.Status().Term
	other := uint64(2)
	if lead == 2 {
		other = 3
	}
	mustStep(t, n, raft.Message{Type: raft.MsgAppResp, From: other, To: lead, Term: term + 6, Success: false})
	s := n.Status()
	if s.Role != raft.Follower || s.Term != term+6 || s.Leader != 0 {
		t.Fatalf("leader after a reply from term %d = %+v, want a follower of term %d with no known leader", term+6, s, term+6)
	}
	if hs := c.stores[lead].hs; hs.Term != term+6 || hs.Vote != 0 {
		t.Fatalf("the new term must be saved with no vote: store holds %+v", hs)
	}
}

func TestStableLeaderNoSpuriousElections(t *testing.T) {
	// WHY: with no fault, the first leader keeps its term forever: heartbeats
	//      every HeartbeatTicks reset each follower's election timer. A leader
	//      that skips heartbeats, or a follower that does not reset its timer
	//      on AppendEntries, causes an election every timeout.
	// KIND: unit
	// CATCHES: s10, s15
	// CHAPTER: dur.10 section 2.3; section 5, Pitfalls, item 8
	c := newCluster(t, []int{10, 14, 18}, 3, nil)
	lead := c.runUntilLeader(40)
	term := c.st(lead).Term
	for i := 0; i < 400; i++ {
		c.tick()
		c.settle()
	}
	for _, id := range c.ids {
		if s := c.st(id); s.Term != term || s.Leader != lead {
			t.Fatalf("after 400 quiet ticks node %d = %+v, want term %d led by %d", id, s, term, lead)
		}
	}
}

// -- replication ------------------------------------------------------------------------

func TestAppendConsistencyCheckAndHint(t *testing.T) {
	// WHY: a follower accepts entries only after the entry before them
	//      matches (index and term). On a mismatch it says where to retry:
	//      past its log end, its last index + 1; inside a conflicting term,
	//      the first index of that term, so the leader skips the term at once.
	// KIND: unit
	// CATCHES: s18
	// CHAPTER: dur.10 section 2.4
	mk := func() *raft.Node {
		st := &memStore{hs: raft.HardState{Term: 2}, ents: ents(1, 1, 2, 2, 2)}
		n, err := raft.New(raft.Config{ID: 2, Peers: []uint64{1, 2, 3}, ElectionTicks: 10, HeartbeatTicks: 3, Timeout: func() int { return 10 }}, st)
		if err != nil {
			t.Fatal(err)
		}
		return n
	}
	n := mk()
	mustStep(t, n, raft.Message{Type: raft.MsgApp, From: 1, To: 2, Term: 3, PrevLogIndex: 8, PrevLogTerm: 3})
	r := one(t, n.ReadMessages(), "reply past the end")
	if r.Success || r.ConflictIndex != 6 {
		t.Fatalf("prev 8 on a 5-entry log: reply %+v, want failure with ConflictIndex 6", r)
	}
	n = mk()
	mustStep(t, n, raft.Message{Type: raft.MsgApp, From: 1, To: 2, Term: 3, PrevLogIndex: 5, PrevLogTerm: 3})
	r = one(t, n.ReadMessages(), "reply inside term 2")
	if r.Success || r.ConflictIndex != 3 {
		t.Fatalf("prev (5, t3) where the follower has t2 from index 3: reply %+v, want failure with ConflictIndex 3", r)
	}
	if s := n.Status(); s.LastIndex != 5 {
		t.Fatalf("a rejected AppendEntries must not change the log: %+v", s)
	}
}

func TestAppendReplacesOnlyTheConflictingSuffix(t *testing.T) {
	// WHY: entries that conflict (same index, other term) and everything after
	//      them are replaced by the leader's; entries before the conflict stay.
	// KIND: unit
	// CATCHES: s16
	// CHAPTER: dur.10 section 2.4
	st := &memStore{hs: raft.HardState{Term: 2}, ents: ents(1, 1, 2, 2)}
	n, err := raft.New(raft.Config{ID: 2, Peers: []uint64{1, 2, 3}, ElectionTicks: 10, HeartbeatTicks: 3, Timeout: func() int { return 10 }}, st)
	if err != nil {
		t.Fatal(err)
	}
	leader := []raft.Entry{{Index: 3, Term: 3, Data: []byte("3@3")}, {Index: 4, Term: 3, Data: []byte("4@3")}}
	mustStep(t, n, raft.Message{Type: raft.MsgApp, From: 1, To: 2, Term: 3, PrevLogIndex: 2, PrevLogTerm: 1, Entries: leader})
	r := one(t, n.ReadMessages(), "reply")
	if !r.Success || r.MatchIndex != 4 {
		t.Fatalf("reply %+v, want success with MatchIndex 4", r)
	}
	want := append(ents(1, 1), leader...)
	if !sameEntries(st.ents, want) {
		t.Fatalf("stored log %s, want %s", fmtEntries(st.ents), fmtEntries(want))
	}
	if s := n.Status(); s.LastIndex != 4 || s.LastTerm != 3 {
		t.Fatalf("status %+v, want last entry (4, t3)", s)
	}
}

func TestStaleShortAppendDoesNotTruncate(t *testing.T) {
	// WHY: the network reorders. An old AppendEntries carrying only entry 2
	//      can arrive after a newer one carried entries 2 to 4. Every entry in
	//      it matches, so nothing is truncated: dropping 3 and 4 here would
	//      lose entries the leader already counted toward a commit.
	// KIND: fault
	// CATCHES: s05
	// CHAPTER: dur.10 section 2.4; section 5, Pitfalls, item 3
	st := &memStore{hs: raft.HardState{Term: 1}, ents: ents(1, 1, 1, 1)}
	n, err := raft.New(raft.Config{ID: 2, Peers: []uint64{1, 2, 3}, ElectionTicks: 10, HeartbeatTicks: 3, Timeout: func() int { return 10 }}, st)
	if err != nil {
		t.Fatal(err)
	}
	old := ents(1, 1)[1:]
	mustStep(t, n, raft.Message{Type: raft.MsgApp, From: 1, To: 2, Term: 1, PrevLogIndex: 1, PrevLogTerm: 1, Entries: old})
	r := one(t, n.ReadMessages(), "reply")
	if !r.Success || r.MatchIndex != 2 {
		t.Fatalf("reply %+v, want success with MatchIndex 2 (what this message proves)", r)
	}
	if s := n.Status(); s.LastIndex != 4 || len(st.ents) != 4 {
		t.Fatalf("a stale, shorter AppendEntries truncated the log: last index %d, stored %d entries", s.LastIndex, len(st.ents))
	}
}

func TestFollowerCommitStopsAtVerifiedEntries(t *testing.T) {
	// WHY: a follower commits min(leaderCommit, index of the last entry this
	//      message verified). Here the follower's entries 2 and 3 are from an
	//      old term and differ from the leader's; a heartbeat that proves only
	//      index 1 must not commit them, even though the leader's commit is 3.
	// KIND: unit
	// CATCHES: s06
	// CHAPTER: dur.10 section 2.4; section 5, Pitfalls, item 4
	st := &memStore{hs: raft.HardState{Term: 1}, ents: ents(1, 1, 1)}
	n, err := raft.New(raft.Config{ID: 2, Peers: []uint64{1, 2, 3}, ElectionTicks: 10, HeartbeatTicks: 3, Timeout: func() int { return 10 }}, st)
	if err != nil {
		t.Fatal(err)
	}
	mustStep(t, n, raft.Message{Type: raft.MsgApp, From: 1, To: 2, Term: 2, PrevLogIndex: 1, PrevLogTerm: 1, Commit: 3})
	n.ReadMessages()
	if s := n.Status(); s.Commit != 1 {
		t.Fatalf("follower commit = %d after a heartbeat that verified only index 1; want 1", s.Commit)
	}
	if got := n.ReadCommitted(); len(got) != 1 {
		t.Fatalf("ReadCommitted = %s, want only entry 1: entries 2 and 3 may differ from the leader's", fmtEntries(got))
	}
}

func TestLaggingFollowerIsRepaired(t *testing.T) {
	// WHY: a follower whose log diverged in an old term is repaired by the
	//      leader: nextIndex backs off until the logs agree, then the leader's
	//      entries overwrite the follower's. After that every log is equal.
	// KIND: unit
	// CATCHES: s11, s15, s16
	// CHAPTER: dur.10 section 2.4; section 5, Pitfalls, item 9
	pre := map[uint64]*memStore{
		1: {hs: raft.HardState{Term: 3}, ents: ents(1, 1, 3, 3)},
		2: {hs: raft.HardState{Term: 3}, ents: ents(1, 1, 3)},
		3: {hs: raft.HardState{Term: 3}, ents: ents(1, 2, 2, 2, 2)},
	}
	c := newCluster(t, []int{10, 15, 20}, 3, pre)
	lead := c.runUntilLeader(40)
	if lead == 3 {
		t.Fatalf("node 3's last term is 2: it cannot win against logs ending in term 3")
	}
	if _, _, err := c.node(lead).Propose([]byte("after-repair")); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		c.tick()
		c.settle()
	}
	want := c.stores[lead].ents
	for _, id := range c.ids {
		if !sameEntries(c.stores[id].ents, want) {
			t.Fatalf("node %d log %s, the leader's %s", id, fmtEntries(c.stores[id].ents), fmtEntries(want))
		}
		if s := c.st(id); s.Commit != uint64(len(want)) {
			t.Fatalf("node %d commit %d, want %d", id, s.Commit, len(want))
		}
	}
	c.checkStores()
}

// -- the commit rule ----------------------------------------------------------------------

func TestFigure8OldTermEntryNotCommittedByCounting(t *testing.T) {
	// WHY: figure 8 of the paper. Node 1 leads term 4 holding (2, t2) from an
	//      old term; node 3 holds (2, t3). Once node 2 also holds (2, t2), that
	//      entry is on a majority, but it is NOT committed: node 3 (last term
	//      3) could still win an election with node 2's vote and overwrite it.
	//      It commits only when the leader's own term-4 entry after it reaches
	//      a majority. MaxEntries 1 makes the two arrive in separate messages.
	// KIND: unit
	// CATCHES: s07, s08, s11, s17
	// CHAPTER: dur.10 section 2.5; section 5, Pitfalls, item 5
	pre := map[uint64]*memStore{
		1: {hs: raft.HardState{Term: 3}, ents: ents(1, 2)},
		2: {hs: raft.HardState{Term: 3}, ents: ents(1)},
		3: {hs: raft.HardState{Term: 3}, ents: ents(1, 3)},
	}
	c := newCluster(t, []int{10, 15, 20}, 3, pre)
	c.cfg.MaxEntries = 1
	for _, id := range c.ids {
		c.start(id)
	}
	c.isolate(3)
	// Term 4: node 1 times out, node 2 votes for it.
	for i := 0; i < 10; i++ {
		c.tick(1, 2)
	}
	c.deliver(c.take(1, 2, raft.MsgVote)...)
	c.deliver(c.take(2, 1, raft.MsgVoteResp)...)
	c.take(1, 3, 0) // node 3 is cut off
	if s := c.st(1); s.Role != raft.Leader || s.Term != 4 || s.LastIndex != 3 {
		t.Fatalf("node 1 = %+v, want the leader of term 4 with its no-op at index 3", s)
	}
	// Node 1 probes node 2 and walks back until node 2 accepts (2, t2) alone.
	c.deliver(c.take(1, 2, raft.MsgApp)...) // prev (2, t2): node 2 lacks index 2
	c.deliver(c.take(2, 1, raft.MsgAppResp)...)
	c.deliver(c.take(1, 2, raft.MsgApp)...) // prev (1, t1), entries [(2, t2)]
	got := c.take(2, 1, raft.MsgAppResp)
	if r := one(t, got, "node 2's answer"); !r.Success || r.MatchIndex != 2 {
		t.Fatalf("setup: node 2 should now match through index 2: %+v", r)
	}
	c.deliver(got...)
	if s := c.st(1); s.Commit != 0 {
		t.Fatalf("(2, t2) sits on 2 of 3 nodes but is from term 2: commit must stay 0 until a term-4 entry commits; got %d", s.Commit)
	}
	c.settle() // (3, t4) reaches node 2
	if s := c.st(1); s.Commit != 3 {
		t.Fatalf("the term-4 no-op on a majority commits it and everything before it: commit %d, want 3", s.Commit)
	}
}

func TestNewLeaderCommitsEarlierTermsThroughItsNoop(t *testing.T) {
	// WHY: a leader cannot count replicas of earlier-term entries, so it
	//      appends a no-op of its own term when elected. Without it, entries
	//      a crashed leader replicated stay uncommitted until some client
	//      happens to write again.
	// KIND: fault
	// CATCHES: s11, s15, s17
	// CHAPTER: dur.10 section 2.5; section 5, Pitfalls, item 12
	pre := map[uint64]*memStore{
		1: {hs: raft.HardState{Term: 2}, ents: ents(1, 2, 2)},
		2: {hs: raft.HardState{Term: 2}, ents: ents(1, 2, 2)},
		3: {hs: raft.HardState{Term: 2}, ents: ents(1, 2)},
	}
	c := newCluster(t, []int{10, 15, 20}, 3, pre)
	lead := c.runUntilLeader(40)
	for i := 0; i < 10; i++ {
		c.tick()
		c.settle()
	}
	for _, id := range c.ids {
		if s := c.st(id); s.Commit < 3 {
			t.Fatalf("node %d commit %d: the old entries 1 to 3 should commit through leader %d's no-op", id, s.Commit, lead)
		}
	}
}

// -- the client side ----------------------------------------------------------------------

func TestProposeOnFollowerReturnsNotLeader(t *testing.T) {
	// WHY: only the leader may append. A follower answers with the leader it
	//      knows, so the server can name it in the tl-raft-leader trailer; an
	//      empty proposal is refused because empty Data is the no-op.
	// KIND: unit
	// CATCHES: s13
	// CHAPTER: dur.10 section 4; section 5, Pitfalls, item 10
	c := newCluster(t, []int{10, 15, 20}, 3, nil)
	lead := c.runUntilLeader(40)
	c.settle()
	for i := 0; i < 3; i++ {
		c.tick()
		c.settle()
	}
	for _, id := range c.ids {
		if id == lead {
			continue
		}
		_, _, err := c.node(id).Propose([]byte("x"))
		var nl *raft.NotLeaderError
		if !errors.As(err, &nl) || !errors.Is(err, raft.ErrNotLeader) || nl.Leader != lead {
			t.Fatalf("Propose on follower %d: err %v, want *NotLeaderError{Leader: %d} wrapping ErrNotLeader", id, err, lead)
		}
		if s := c.st(id); s.LastIndex != c.st(lead).LastIndex {
			t.Fatalf("a refused proposal changed follower %d's log", id)
		}
	}
	if _, _, err := c.node(lead).Propose(nil); !errors.Is(err, raft.ErrEmptyProposal) {
		t.Fatalf("Propose(nil) on the leader: %v, want ErrEmptyProposal", err)
	}
}

func TestStorageFailureSendsNothing(t *testing.T) {
	// WHY: a node must not send a message that depends on state it failed to
	//      save. If saving its own vote fails, it may not ask for votes: after
	//      a restart it would not know it voted, and could vote again.
	// KIND: fault
	// CATCHES: s14
	// CHAPTER: dur.10 section 2.6; section 5, Pitfalls, item 11
	st := &memStore{}
	n, err := raft.New(raft.Config{ID: 1, Peers: []uint64{1, 2, 3}, ElectionTicks: 10, HeartbeatTicks: 3, Timeout: func() int { return 2 }}, st)
	if err != nil {
		t.Fatal(err)
	}
	if err := n.Tick(); err != nil {
		t.Fatal(err)
	}
	st.failNext = errors.New("disk full")
	err = n.Tick() // the timeout: campaign, and the save fails
	if err == nil {
		t.Fatal("Tick must report the failed save")
	}
	if ms := n.ReadMessages(); len(ms) != 0 {
		t.Fatalf("the node sent %d message(s) after failing to save its vote: %+v", len(ms), ms)
	}
	if err := n.Step(raft.Message{Type: raft.MsgVote, From: 2, To: 1, Term: 9}); err == nil {
		t.Fatal("a node whose storage failed must stay stopped")
	}
}

func TestPCG32MatchesSpec(t *testing.T) {
	// WHY: the simulation's faults come from this generator; it must be the
	//      spec's PCG32 so a seed names the same fault schedule everywhere.
	//      pcg32(42) starts a15c02b7 7b47f409 (spec/pcg32.md).
	// KIND: golden
	p := newPCG32(42, 54)
	if a, b := p.next(), p.next(); a != 0xa15c02b7 || b != 0x7b47f409 {
		t.Fatalf("pcg32(42) = %08x %08x, want a15c02b7 7b47f409", a, b)
	}
}
