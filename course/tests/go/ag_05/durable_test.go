// Course tests for ag.05, durable agent runs. The model is a counting fake
// (a scripted Provider, or the course's faketool server when the run lives
// in a child process), so "the model was not called again" is a number the
// test reads. Crashes are real where it matters: TestKillResumesWithoutRecall
// SIGKILLs a child process in the middle of a read and in the middle of a
// write. Elsewhere a store that fails one append stands in for a crash at
// exactly that point.
package ag_05

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"tinyllm/agent/durableagent"
	"tinyllm/agent/gate"
	"tinyllm/agent/loop"
	"tinyllm/agent/tool"
	"tinyllm/agent/types"
)

// script is a counting fake model: call i gets replies[i].
type script struct {
	mu      sync.Mutex
	n       int
	replies []types.Message
}

func (s *script) ChatStream(ctx context.Context, msgs []types.Message, _ []types.ToolDef, _ ...types.CallOption) (<-chan types.Delta, error) {
	s.mu.Lock()
	r := s.replies[min(s.n, len(s.replies)-1)]
	s.n++
	s.mu.Unlock()
	ch := make(chan types.Delta, 8)
	if r.Content != "" {
		ch <- types.TextDelta{Text: r.Content}
	}
	for i, c := range r.ToolCalls {
		ch <- types.ToolCallEndDelta{Index: i, Call: c}
	}
	ch <- types.UsageDelta{In: 10, Out: 2}
	ch <- types.DoneDelta{FinishReason: "stop"}
	close(ch)
	return ch, nil
}

func (s *script) calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.n
}

// ticketModel: look the policy up, file a ticket, then answer.
func ticketModel() *script {
	return &script{replies: []types.Message{
		{Role: types.RoleAssistant, ToolCalls: []types.ToolCall{{ID: "c1", Name: "lookup", Args: json.RawMessage(`{"q":"access policy"}`)}}},
		{Role: types.RoleAssistant, ToolCalls: []types.ToolCall{{ID: "c2", Name: "create_ticket", Args: json.RawMessage(`{"title":"grant access"}`)}}},
		{Role: types.RoleAssistant, Content: "Ticket filed."},
	}}
}

type counters struct{ lookup, ticket atomic.Int32 }

func registry(c *counters) *tool.Registry {
	return tool.NewRegistry().MustRegister(
		tool.New("lookup", "", `{"type":"object"}`, func(context.Context, json.RawMessage) (string, error) {
			c.lookup.Add(1)
			return "policy: tickets need approval", nil
		}),
		tool.New("create_ticket", "", `{"type":"object"}`, func(context.Context, json.RawMessage) (string, error) {
			c.ticket.Add(1)
			return "TICKET-1", nil
		}),
	)
}

func policyGate() *gate.PolicyGate {
	return gate.New(gate.Policy{Read: []string{"lookup"}, Write: []string{"create_ticket"}})
}

var input = []types.Message{{Role: types.RoleUser, Content: "Please file a ticket to grant me access."}}

func types_(recs []durableagent.Record) []string {
	var out []string
	for _, r := range recs {
		s := r.Type
		if r.Step != "" {
			s += " " + r.Step
		}
		out = append(out, s)
	}
	return out
}

func TestHandExample(t *testing.T) {
	// WHY: section 3 by hand. Run 1 records seven entries (the input, then
	//      start and result of llm/1, tool/1/0/lookup, llm/2) and stops
	//      awaiting approval of create_ticket. An approve signal, then Run 2:
	//      llm/1, lookup, and llm/2 come from the journal (still 2 model
	//      calls, 1 lookup), the ticket is filed once, llm/3 answers. Run 3
	//      returns the recorded answer and calls nothing.
	// KIND: unit
	// CATCHES: s01, s08
	// CHAPTER: ag.05 section 3, worked example
	store, err := durableagent.OpenFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	p, c := ticketModel(), &counters{}
	cfg := durableagent.Config{Agent: loop.Config{Provider: p, Tools: registry(c)}, Gate: policyGate(), Store: store}
	ctx := context.Background()
	out, err := durableagent.Run(ctx, cfg, "a1", input)
	if err != nil || out.Status != durableagent.AwaitingApproval || len(out.Pending) != 1 {
		t.Fatalf("run 1: %+v, %v; want awaiting approval of one call", out, err)
	}
	recs, _ := store.Load(ctx, "a1")
	want := []string{"run.started", "step.started llm/1", "step.completed llm/1", "step.started tool/1/0/lookup",
		"step.completed tool/1/0/lookup", "step.started llm/2", "step.completed llm/2"}
	if got := types_(recs); !reflect.DeepEqual(got, want) {
		t.Fatalf("journal after run 1:\n got %v\nwant %v", got, want)
	}
	if err := durableagent.Signal(ctx, store, "a1", "approve", map[string]string{"Marker": out.Pending[0].Marker}); err != nil {
		t.Fatal(err)
	}
	out, err = durableagent.Run(ctx, cfg, "a1", input)
	if err != nil || out.Status != durableagent.Completed || out.Message.Content != "Ticket filed." {
		t.Fatalf("run 2: %+v, %v", out, err)
	}
	if p.calls() != 3 || c.lookup.Load() != 1 || c.ticket.Load() != 1 {
		t.Fatalf("after run 2: %d model calls, %d lookups, %d tickets; want 3, 1, 1", p.calls(), c.lookup.Load(), c.ticket.Load())
	}
	out, err = durableagent.Run(ctx, cfg, "a1", nil)
	if err != nil || out.Message.Content != "Ticket filed." || p.calls() != 3 || c.ticket.Load() != 1 {
		t.Fatalf("run 3 (a completed run): %+v, %v, %d calls, %d tickets", out, err, p.calls(), c.ticket.Load())
	}
}

// crashStore fails the append of one record, as if the process died just
// before writing it.
type crashStore struct {
	durableagent.Store
	typ, step string
	armed     bool
}

func (s *crashStore) Append(ctx context.Context, runID string, r durableagent.Record) error {
	if s.armed && r.Type == s.typ && r.Step == s.step {
		s.armed = false
		return errors.New("crash: process died before this record was written")
	}
	return s.Store.Append(ctx, runID, r)
}

func TestIndeterminateWrite(t *testing.T) {
	// WHY: the ticket was filed but the crash came before its result was
	//      recorded. Running create_ticket again could file it twice;
	//      skipping it could lose it. The run stops with ErrIndeterminate
	//      naming the step, until a reconcile signal says what happened:
	//      "done" records the human's result and never runs the write
	//      again; "retry" runs it once more.
	// KIND: fault
	// CATCHES: s02, s07
	// CHAPTER: ag.05 section 2.3
	for _, outcome := range []string{"done", "retry"} {
		t.Run(outcome, func(t *testing.T) {
			fs, _ := durableagent.OpenFileStore(t.TempDir())
			p, c := ticketModel(), &counters{}
			ctx := context.Background()
			cfg := durableagent.Config{Agent: loop.Config{Provider: p, Tools: registry(c)}, Gate: policyGate(), Store: fs}
			out, _ := durableagent.Run(ctx, cfg, "a1", input)
			durableagent.Signal(ctx, fs, "a1", "approve", map[string]string{"Marker": out.Pending[0].Marker})
			cfg.Store = &crashStore{Store: fs, typ: durableagent.StepCompleted, step: "tool/2/0/create_ticket", armed: true}
			if _, err := durableagent.Run(ctx, cfg, "a1", input); err == nil {
				t.Fatal("the crashed attempt must fail")
			}
			cfg.Store = fs
			out, err := durableagent.Run(ctx, cfg, "a1", input)
			if err != nil || out.Status != durableagent.Indeterminate || out.Step != "tool/2/0/create_ticket" || c.ticket.Load() != 1 {
				t.Fatalf("after the crash: %+v, %v, %d tickets; want indeterminate at tool/2/0/create_ticket, 1 ticket", out, err, c.ticket.Load())
			}
			if out2, _ := durableagent.Run(ctx, cfg, "a1", input); out2.Status != durableagent.Indeterminate || c.ticket.Load() != 1 {
				t.Fatalf("without a signal it stays indeterminate: %+v, %d tickets", out2, c.ticket.Load())
			}
			durableagent.Signal(ctx, fs, "a1", "reconcile", durableagent.Reconcile{Step: out.Step, Outcome: outcome, Result: "TICKET-1 (checked by hand)"})
			out, err = durableagent.Run(ctx, cfg, "a1", input)
			wantTickets := map[string]int32{"done": 1, "retry": 2}[outcome]
			if err != nil || out.Status != durableagent.Completed || c.ticket.Load() != wantTickets || p.calls() != 3 {
				t.Fatalf("after reconcile %s: %+v, %v, %d tickets, %d model calls; want completed, %d, 3", outcome, out, err, c.ticket.Load(), p.calls(), wantTickets)
			}
		})
	}
}

func TestReadStepRerunsAfterCrash(t *testing.T) {
	// WHY: a read that started but never recorded its result is safe to run
	//      again (reading twice changes nothing), so it must not block the
	//      run the way an unrecorded write does; the model call before it is
	//      still not repeated.
	// KIND: fault
	// CATCHES: s03
	// CHAPTER: ag.05 section 2.3
	fs, _ := durableagent.OpenFileStore(t.TempDir())
	p, c := ticketModel(), &counters{}
	ctx := context.Background()
	cfg := durableagent.Config{Agent: loop.Config{Provider: p, Tools: registry(c)}, Gate: policyGate(),
		Store: &crashStore{Store: fs, typ: durableagent.StepCompleted, step: "tool/1/0/lookup", armed: true}}
	if _, err := durableagent.Run(ctx, cfg, "a1", input); err == nil {
		t.Fatal("the crashed attempt must fail")
	}
	cfg.Store = fs
	out, err := durableagent.Run(ctx, cfg, "a1", input)
	if err != nil || out.Status != durableagent.AwaitingApproval || c.lookup.Load() != 2 || p.calls() != 2 {
		t.Fatalf("%+v, %v, %d lookups, %d model calls; want awaiting approval after 2 lookups and 2 model calls", out, err, c.lookup.Load(), p.calls())
	}
}

func TestInputMismatch(t *testing.T) {
	// WHY: a run id names one run. Starting it again with the same input
	//      resumes it (a retried start is harmless); starting it with other
	//      input is a client bug that must not silently answer the old
	//      question.
	// KIND: boundary
	// CATCHES: s10
	// CHAPTER: ag.05 section 4
	fs, _ := durableagent.OpenFileStore(t.TempDir())
	p, c := ticketModel(), &counters{}
	cfg := durableagent.Config{Agent: loop.Config{Provider: p, Tools: registry(c)}, Gate: policyGate(), Store: fs}
	ctx := context.Background()
	if _, err := durableagent.Run(ctx, cfg, "a1", input); err != nil {
		t.Fatal(err)
	}
	if _, err := durableagent.Run(ctx, cfg, "a1", input); err != nil {
		t.Fatalf("same input: %v", err)
	}
	other := []types.Message{{Role: types.RoleUser, Content: "Delete my account."}}
	if _, err := durableagent.Run(ctx, cfg, "a1", other); !errors.Is(err, durableagent.ErrInputMismatch) {
		t.Fatalf("other input: %v, want ErrInputMismatch", err)
	}
	if _, err := durableagent.Run(ctx, cfg, "nope", nil); err == nil {
		t.Fatal("resuming a run that does not exist must fail")
	}
}

func TestFileStoreTornTail(t *testing.T) {
	// WHY: a crash in the middle of a write leaves half a line at the end of
	//      the journal. Load must ignore it (the record was never
	//      acknowledged) instead of refusing the whole run, and the next
	//      Append must cut it off instead of gluing a record onto it.
	// KIND: fault
	// CATCHES: s05, s06
	// CHAPTER: ag.05 section 2.2
	dir := t.TempDir()
	fs, _ := durableagent.OpenFileStore(dir)
	ctx := context.Background()
	for i := 0; i < 2; i++ {
		if err := fs.Append(ctx, "r1", durableagent.Record{Seq: int64(i + 1), Type: durableagent.StepStarted, Step: "llm/1"}); err != nil {
			t.Fatal(err)
		}
	}
	f, _ := os.OpenFile(filepath.Join(dir, "r1.jsonl"), os.O_APPEND|os.O_WRONLY, 0)
	f.WriteString(`{"seq":3,"type":"step.compl`)
	f.Close()
	recs, err := fs.Load(ctx, "r1")
	if err != nil || len(recs) != 2 {
		t.Fatalf("Load with a torn tail: %d records, %v; want 2, nil", len(recs), err)
	}
	if err := fs.Append(ctx, "r1", durableagent.Record{Seq: 3, Type: durableagent.StepCompleted, Step: "llm/1"}); err != nil {
		t.Fatal(err)
	}
	recs, err = fs.Load(ctx, "r1")
	if err != nil || len(recs) != 3 || recs[2].Type != durableagent.StepCompleted {
		t.Fatalf("after Append: %v, %v", types_(recs), err)
	}
	b, _ := os.ReadFile(filepath.Join(dir, "r1.jsonl"))
	if strings.Contains(string(b), "step.compl`") || strings.Count(string(b), "\n") != 3 {
		t.Fatalf("journal bytes:\n%s", b)
	}
}

func TestRunIDValidation(t *testing.T) {
	// WHY: the run id becomes a file name; "../../etc/passwd" or "a/b" must
	//      never reach the file system (a run id comes from a CLI flag or an
	//      API request).
	// KIND: boundary
	// CATCHES: s11
	// CHAPTER: ag.05 section 5
	dir := t.TempDir()
	fs, _ := durableagent.OpenFileStore(filepath.Join(dir, "runs"))
	for _, id := range []string{"../escape", "a/b", "", "..", ".hidden"} {
		if err := fs.Append(context.Background(), id, durableagent.Record{Type: durableagent.RunStarted}); err == nil {
			t.Errorf("run id %q accepted", id)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "escape.jsonl")); err == nil {
		t.Fatal("a file was written outside the store directory")
	}
	if err := durableagent.Signal(context.Background(), fs, "missing", "approve", map[string]string{"Marker": "x"}); err == nil {
		t.Fatal("a signal for a run that does not exist must fail")
	}
}
