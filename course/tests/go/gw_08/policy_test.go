package gw_08_test

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"tinyllm/gateway/auth"
	"tinyllm/gateway/policy"
	"tinyllm/gateway/route"
)

type fixedEmbedder []float64

func (f fixedEmbedder) Embed(context.Context, string, string) ([]float64, error) {
	return []float64(f), nil
}

// WHY: Policy decisions must agree with the head exported by Python (D33).
// KIND: differential
func TestLinearHeadFixtureParity(t *testing.T) {
	b, err := os.ReadFile("../../../fixtures/L6.5/linear_head.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Head       policy.Head `json:"head"`
		Embeddings [][]float64 `json:"embeddings"`
		Probs      [][]float64 `json:"probs"`
	}
	if err = json.Unmarshal(b, &fixture); err != nil {
		t.Fatal(err)
	}
	for i, x := range fixture.Embeddings {
		got, err := policy.Probabilities(fixture.Head, x)
		if err != nil {
			t.Fatal(err)
		}
		for j, p := range fixture.Probs[i] {
			if math.Abs(got[j]-p) > 1e-6 {
				t.Fatalf("row %d class %d: got %.9g want %.9g", i, j, got[j], p)
			}
		}
	}
}

// WHY: Rule order and max_tokens direction are externally visible semantics.
// KIND: boundary
func TestPolicyOrderingAndModelTenantRules(t *testing.T) {
	cap := 1024
	e := policy.Evaluator{Document: policy.Document{Version: 1, DefaultAction: "allow", Rules: []policy.Rule{
		{ID: "free-cap", Match: policy.Match{Tenant: []string{"free"}, MaxTokens: &cap}, Action: "deny", Reason: "cap"},
		{ID: "model", Match: policy.Match{Model: []string{"blocked"}}, Action: "deny", Reason: "model"},
	}}}
	d, err := e.Check(context.Background(), auth.Principal{Tenant: "free"}, &route.InferenceRequest{Model: "blocked"}, 2048, "", "r1")
	if err != nil || d.Allow || d.RuleID != "free-cap" {
		t.Fatalf("first match: %+v, %v", d, err)
	}
	d, err = e.Check(context.Background(), auth.Principal{Tenant: "paid"}, &route.InferenceRequest{Model: "blocked"}, 20, "", "r2")
	if err != nil || d.Allow || d.RuleID != "model" {
		t.Fatalf("model match: %+v, %v", d, err)
	}
}

// WHY: Zero and very large vectors must not produce NaN probabilities.
// KIND: property
func TestZeroAndLargeEmbeddingsAreFinite(t *testing.T) {
	h := policy.Head{Dim: 2, Classes: []string{"safe", "unsafe"}, W: [][]float64{{0, 0}, {1, -1}}, B: []float64{0, 0}, Threshold: .9}
	for _, x := range [][]float64{{0, 0}, {1e300, -1e300}} {
		p, err := policy.Probabilities(h, x)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(p[0]) || math.IsInf(p[0], 0) || math.Abs(p[0]+p[1]-1) > 1e-12 {
			t.Fatalf("invalid probabilities: %v", p)
		}
	}
}

// WHY: Missing classifier data must not silently disable a deny rule.
// KIND: fault
func TestEmbeddingsFailureFailsClosed(t *testing.T) {
	e := policy.Evaluator{Document: policy.Document{Version: 1, DefaultAction: "allow", Rules: []policy.Rule{{ID: "unsafe", Match: policy.Match{}, Classifier: &policy.ClassifierRule{Head: "missing", Class: "unsafe"}, Action: "deny", Reason: "blocked"}}}}
	d, err := e.Check(context.Background(), auth.Principal{Tenant: "acme"}, &route.InferenceRequest{Model: "m"}, 10, "prompt", "r1")
	if err == nil || d.Allow {
		t.Fatalf("missing classifier data did not fail closed: %+v, %v", d, err)
	}
}

// WHY: Both explicit and default denials need a durable decision record.
// KIND: boundary
func TestEveryDenyIsAudited(t *testing.T) {
	var events []policy.AuditEvent
	audit := func(_ context.Context, event policy.AuditEvent) error {
		events = append(events, event)
		return nil
	}
	request := &route.InferenceRequest{Model: "m"}
	firstMatch := policy.Evaluator{
		Document: policy.Document{Version: 1, DefaultAction: "allow", Rules: []policy.Rule{{ID: "blocked-model", Match: policy.Match{Model: []string{"m"}}, Action: "deny", Reason: "blocked"}}},
		Audit:    audit,
	}
	if d, err := firstMatch.Check(context.Background(), auth.Principal{Tenant: "acme"}, request, 1, "private prompt", "r1"); err != nil || d.Allow {
		t.Fatalf("rule denial: %+v, %v", d, err)
	}
	defaultDeny := policy.Evaluator{Document: policy.Document{Version: 1, DefaultAction: "deny"}, Audit: audit}
	if d, err := defaultDeny.Check(context.Background(), auth.Principal{Tenant: "acme"}, request, 1, "private prompt", "r2"); err != nil || d.Allow {
		t.Fatalf("default denial: %+v, %v", d, err)
	}
	if len(events) != 2 || events[0].RuleID != "blocked-model" || events[1].RuleID != "default-deny" {
		t.Fatalf("deny audit events: %+v", events)
	}
}

// WHY: Logs must redact every data.05 category without flagging lookalikes.
// KIND: boundary
func TestRedactPII(t *testing.T) {
	cases := []struct{ input, want string }{
		{"Write to ana.lopez@example.org", "Write to <EMAIL>"},
		{"Call +1 (415) 555-0132", "Call <PHONE>"},
		{"Card 4539 1488 0343 6467", "Card <CARD>"},
		{"Order 4539 1488 0343 6468", "Order 4539 1488 0343 6468"},
		{"IP 203.0.113.42 and 2001:db8:85a3::8a2e:370:7334", "IP <IP> and <IP>"},
		{"AKIAIOSFODNN7EXAMPLE", "<KEY>"},
		{"Bearer tl_k7f3_9s8d7f6g5h4j3k2l1m0nq8w7", "Bearer <KEY>"},
		{"trace_id=4bf92f3577b34da6a3ce929d0e0e4736", "trace_id=4bf92f3577b34da6a3ce929d0e0e4736"},
	}
	for _, tc := range cases {
		if got := policy.Redact(tc.input); got != tc.want {
			t.Errorf("Redact(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

// WHY: Configuration must decode both match forms and reject artifact traversal.
// KIND: boundary
func TestPolicyYAMLLoad(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "models"), 0o700); err != nil {
		t.Fatal(err)
	}
	head := policy.Head{EmbeddingModel: "m", Dim: 2, Classes: []string{"safe", "unsafe"}, W: [][]float64{{0, 0}, {1, 1}}, B: []float64{0, 0}, Threshold: .9}
	b, err := json.Marshal(head)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(root, "models/head.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	yml := `version: 1
default_action: allow
rules:
  - id: cap
    match:
      model: blocked
      tenant: [free, trial]
      max_tokens: 10
    action: deny
    reason: capped
  - id: classifier
    match: {}
    classifier:
      head: models/head.json
      class: unsafe
      threshold: 0.9
    action: deny
    reason: unsafe
`
	path := filepath.Join(root, "policy.yaml")
	if err = os.WriteFile(path, []byte(yml), 0o600); err != nil {
		t.Fatal(err)
	}
	e, err := policy.Load(path, root, fixedEmbedder{0, 0}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := e.Document.Rules[0].Match.Model; len(got) != 1 || got[0] != "blocked" {
		t.Fatalf("scalar match was not decoded: %v", got)
	}
	if got := e.Document.Rules[0].Match.Tenant; len(got) != 2 {
		t.Fatalf("list match was not decoded: %v", got)
	}
	if len(e.Heads) != 1 {
		t.Fatalf("head not loaded: %+v", e.Heads)
	}
	if err = os.WriteFile(path, []byte(strings.Replace(yml, "models/head.json", "../escape.json", 1)), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err = policy.Load(path, root, fixedEmbedder{0, 0}, nil); err == nil {
		t.Fatal("head traversal was accepted")
	}
}
