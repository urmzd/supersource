// Course tests for ag.02: the Tool interface, the registry, and JSON Schema
// validation of the arguments a model writes. The model is untrusted input:
// every case here is a way a call can be wrong, and the rule under test is
// the same each time: the registry answers with a tool error the model can
// read, runs nothing on invalid arguments, and never panics.
package ag_02

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"tinyllm/agent/tool"
	"tinyllm/agent/types"
)

// weatherSchema is the chapter's worked example (section 3).
const weatherSchema = `{
  "type": "object",
  "properties": {
    "city": {"type": "string", "minLength": 1},
    "days": {"type": "integer", "minimum": 1, "maximum": 7},
    "unit": {"enum": ["c", "f"]}
  },
  "required": ["city"],
  "additionalProperties": false
}`

func compile(t *testing.T, s string) *tool.Schema {
	t.Helper()
	sc, err := tool.Compile(json.RawMessage(s))
	if err != nil {
		t.Fatalf("Compile(%s): %v", s, err)
	}
	return sc
}

func msgs(errs []tool.ValidationError) []string {
	out := []string{}
	for _, e := range errs {
		out = append(out, e.Error())
	}
	return out
}

func TestHandWorkedExample(t *testing.T) {
	// WHY: section 3 by hand. The model sends {"days": "3", "unit":
	//      "kelvin", "country": "FR"} to get_weather: four violations, in
	//      path order, each one sentence the model can act on; the tool does
	//      not run; the valid call runs once with its arguments untouched.
	// KIND: unit
	// CATCHES: s06, s07, s10, s15
	// CHAPTER: ag.02 section 3, worked example
	sc := compile(t, weatherSchema)
	got := msgs(sc.ValidateJSON([]byte(`{"days": "3", "unit": "kelvin", "country": "FR"}`)))
	want := []string{
		`$: missing required property "city"`,
		`$: unexpected property "country"`,
		`$.days: expected integer, got string`,
		`$.unit: "kelvin" is not one of ["c","f"]`,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("errors\n got %q\nwant %q", got, want)
	}
	var runs atomic.Int32
	var seen json.RawMessage
	reg := tool.NewRegistry().MustRegister(tool.New("get_weather", "Weather forecast for a city", weatherSchema,
		func(_ context.Context, args json.RawMessage) (string, error) {
			runs.Add(1)
			seen = args
			return "sunny", nil
		}))
	res := reg.Call(context.Background(), types.ToolCall{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"days": "3", "unit": "kelvin", "country": "FR"}`)})
	if !res.IsError || runs.Load() != 0 {
		t.Fatalf("invalid call: IsError %v after %d runs; want an error and no run", res.IsError, runs.Load())
	}
	m := res.Message()
	wantMsg := "error: invalid arguments for get_weather:\n" + strings.Join(want, "\n")
	if m.Role != types.RoleTool || m.ToolCallID != "call_1" || m.Name != "get_weather" || m.Content != wantMsg {
		t.Fatalf("message %+v\nwant content %q", m, wantMsg)
	}
	res = reg.Call(context.Background(), types.ToolCall{ID: "call_2", Name: "get_weather", Args: json.RawMessage(`{"city":"Paris","days":3}`)})
	if res.IsError || res.Content != "sunny" || runs.Load() != 1 || string(seen) != `{"city":"Paris","days":3}` {
		t.Fatalf("valid call: %+v after %d runs, args %s", res, runs.Load(), seen)
	}
	if m := res.Message(); m.Content != "sunny" {
		t.Fatalf("a successful result is sent as is, got %q", m.Content)
	}
}

func TestValidateTable(t *testing.T) {
	// WHY: one row per rule of the subset, including the edges a hand-rolled
	//      validator gets wrong: 3.0 is an integer and 3.5 is not; number
	//      accepts integers; minimum is inclusive; length counts characters,
	//      not bytes; a pattern matches anywhere unless it anchors; enum
	//      compares numbers by value; error paths name the array index.
	// KIND: unit
	// CATCHES: s01, s02, s03, s04, s05, s09, s17
	// CHAPTER: ag.02 section 2.2
	for _, tc := range []struct {
		name, schema, value string
		want                []string
	}{
		{"integer accepts 3.0", `{"type":"integer"}`, `3.0`, nil},
		{"integer rejects 3.5", `{"type":"integer"}`, `3.5`, []string{"$: expected integer, got number"}},
		{"number accepts 3", `{"type":"number"}`, `3`, nil},
		{"union with null", `{"type":["string","null"]}`, `null`, nil},
		{"union mismatch", `{"type":["string","null"]}`, `true`, []string{"$: expected string or null, got boolean"}},
		{"minimum is inclusive", `{"type":"integer","minimum":1}`, `1`, nil},
		{"below minimum", `{"type":"integer","minimum":1}`, `0`, []string{"$: 0 is less than minimum 1"}},
		{"above maximum", `{"type":"number","maximum":2.5}`, `2.75`, []string{"$: 2.75 is greater than maximum 2.5"}},
		{"length counts characters", `{"type":"string","maxLength":5}`, `"héllo"`, nil},
		{"too long", `{"type":"string","maxLength":4}`, `"héllo"`, []string{"$: has 5 characters, more than maxLength 4"}},
		{"pattern is unanchored", `{"type":"string","pattern":"[0-9]{3}"}`, `"ab123cd"`, nil},
		{"anchored pattern", `{"type":"string","pattern":"^[A-Z]{2}$"}`, `"FRA"`, []string{`$: "FRA" does not match pattern "^[A-Z]{2}$"`}},
		{"enum by value", `{"enum":[1,"x",null]}`, `1.0`, nil},
		{"const", `{"const":"on"}`, `"off"`, []string{`$: "off" is not one of ["on"]`}},
		{"array items path", `{"type":"array","items":{"type":"string"}}`, `["a",2,"c"]`, []string{"$[1]: expected string, got integer"}},
		{"minItems", `{"type":"array","minItems":2}`, `[1]`, []string{"$: has 1 items, fewer than minItems 2"}},
		{"nested object path", `{"type":"object","properties":{"loc":{"type":"object","properties":{"lat":{"type":"number"}}}}}`, `{"loc":{"lat":"north"}}`, []string{"$.loc.lat: expected number, got string"}},
		{"additionalProperties schema", `{"type":"object","additionalProperties":{"type":"integer"}}`, `{"a":1,"b":"x"}`, []string{"$.b: expected integer, got string"}},
		{"anyOf", `{"anyOf":[{"type":"string"},{"type":"integer"}]}`, `[]`, []string{"$: matches none of the 2 anyOf schemas"}},
		{"not JSON", `{"type":"object"}`, `{"city": Paris}`, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := msgs(compile(t, tc.schema).ValidateJSON([]byte(tc.value)))
			if tc.name == "not JSON" {
				if len(got) != 1 || !strings.HasPrefix(got[0], "$: invalid JSON") {
					t.Fatalf("malformed JSON: %q; want one `$: invalid JSON ...` error", got)
				}
				return
			}
			if len(got) == 0 && len(tc.want) == 0 {
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("%s on %s\n got %q\nwant %q", tc.schema, tc.value, got, tc.want)
			}
		})
	}
}

func TestCompileRejectsUnsupported(t *testing.T) {
	// WHY: a keyword the validator does not implement ("format": "email")
	//      would be silently ignored, so a schema that promises an email
	//      would accept anything. Fail closed at registration instead.
	//      Annotations (description, title, default) are fine.
	// KIND: boundary
	// CATCHES: s08
	// CHAPTER: ag.02 section 2.2
	for _, bad := range []string{
		`{"type":"object","properties":{"to":{"type":"string","format":"email"}}}`,
		`{"type":"strng"}`,
		`{"type":"string","pattern":"("}`,
		`{"enum":[]}`,
		`[1,2]`,
	} {
		if _, err := tool.Compile(json.RawMessage(bad)); err == nil {
			t.Errorf("Compile(%s) succeeded; want an error", bad)
		}
	}
	if _, err := tool.Compile(json.RawMessage(`{"type":"object","title":"t","description":"d","properties":{"q":{"type":"string","default":"x","description":"query"}}}`)); err != nil {
		t.Fatalf("annotations must be accepted: %v", err)
	}
}

func noop(context.Context, json.RawMessage) (string, error) { return "ok", nil }

func TestRegisterRules(t *testing.T) {
	// WHY: the contract allows tool names matching ^[A-Za-z0-9_-]{1,64}$ and
	//      parameters of type object; a provider rejects the whole request
	//      for one bad tool, and a duplicate name makes the model's call
	//      ambiguous.
	// KIND: unit
	// CATCHES: s13, s14
	// CHAPTER: ag.02 section 4
	reg := tool.NewRegistry()
	if err := reg.Register(tool.New("search_docs", "", `{"type":"object"}`, noop)); err != nil {
		t.Fatal(err)
	}
	for name, tl := range map[string]tool.Tool{
		"space in name":     tool.New("get weather", "", `{"type":"object"}`, noop),
		"65 characters":     tool.New(strings.Repeat("a", 65), "", `{"type":"object"}`, noop),
		"duplicate":         tool.New("search_docs", "", `{"type":"object"}`, noop),
		"array parameters":  tool.New("list", "", `{"type":"array"}`, noop),
		"invalid schema":    tool.New("bad", "", `{"type":"object","oneOf":[]}`, noop),
		"parameters no obj": tool.New("bad2", "", `{"properties":{}}`, noop),
	} {
		if err := reg.Register(tl); err == nil {
			t.Errorf("%s: Register succeeded; want an error", name)
		}
	}
	if err := reg.Register(tool.New("ping", "no parameters", "", noop)); err != nil {
		t.Fatalf("a tool with no parameters schema means {\"type\":\"object\"}: %v", err)
	}
}

func TestDefinitionsSorted(t *testing.T) {
	// WHY: the tool list is part of the prompt. Sorted by name, it is the
	//      same bytes on every run, so runs are reproducible and a provider's
	//      prompt cache can hit.
	// KIND: unit
	// CATCHES: s12
	// CHAPTER: ag.02 section 2.3
	reg := tool.NewRegistry()
	for _, n := range []string{"query_usage", "get_weather", "search_docs"} {
		reg.MustRegister(tool.New(n, "d "+n, `{"type":"object"}`, noop))
	}
	var got []string
	for _, d := range reg.Definitions() {
		got = append(got, d.Name)
		if d.Description != "d "+d.Name || string(d.Parameters) != `{"type":"object"}` {
			t.Fatalf("definition %+v lost its description or parameters", d)
		}
	}
	if !reflect.DeepEqual(got, []string{"get_weather", "query_usage", "search_docs"}) {
		t.Fatalf("Definitions order %v", got)
	}
}

func TestCallNeverPanics(t *testing.T) {
	// WHY: every way a call can fail ends as a tool error the model reads:
	//      an unknown tool (with the names it may use), malformed JSON, a
	//      tool that returns an error, and a tool that panics (no stack trace
	//      in the reply). Empty arguments mean {}.
	// KIND: fault
	// CATCHES: s10, s11, s16
	// CHAPTER: ag.02 section 2.3
	var runs atomic.Int32
	reg := tool.NewRegistry().MustRegister(
		tool.New("echo", "", `{"type":"object","properties":{"x":{"type":"string"}}}`, func(_ context.Context, a json.RawMessage) (string, error) {
			runs.Add(1)
			return string(a), nil
		}),
		tool.New("fails", "", `{"type":"object"}`, func(context.Context, json.RawMessage) (string, error) {
			return "", errors.New("upstream timed out")
		}),
		tool.New("boom", "", `{"type":"object"}`, func(context.Context, json.RawMessage) (string, error) {
			var m map[string]int
			m["x"] = 1 // nil map write: a panic
			return "", nil
		}),
	)
	ctx := context.Background()
	r := reg.Call(ctx, types.ToolCall{ID: "1", Name: "delete_everything", Args: json.RawMessage(`{}`)})
	if !r.IsError || !strings.Contains(r.Content, `unknown tool "delete_everything"`) || !strings.Contains(r.Content, "boom, echo, fails") {
		t.Fatalf("unknown tool: %+v", r)
	}
	r = reg.Call(ctx, types.ToolCall{ID: "2", Name: "echo", Args: json.RawMessage(`{"x": "unterminated`)})
	if !r.IsError || runs.Load() != 0 {
		t.Fatalf("malformed JSON: %+v after %d runs", r, runs.Load())
	}
	r = reg.Call(ctx, types.ToolCall{ID: "3", Name: "echo", Args: json.RawMessage(`{"x": 5}`)})
	if !r.IsError || runs.Load() != 0 {
		t.Fatalf("schema violation must not run the tool: %+v after %d runs", r, runs.Load())
	}
	r = reg.Call(ctx, types.ToolCall{ID: "4", Name: "echo", Args: nil})
	if r.IsError || r.Content != "{}" || runs.Load() != 1 {
		t.Fatalf("empty arguments: %+v; want the tool run with {}", r)
	}
	r = reg.Call(ctx, types.ToolCall{ID: "5", Name: "fails", Args: json.RawMessage(`{}`)})
	if !r.IsError || r.Content != "upstream timed out" || r.CallID != "5" {
		t.Fatalf("Execute error: %+v", r)
	}
	r = reg.Call(ctx, types.ToolCall{ID: "6", Name: "boom", Args: json.RawMessage(`{}`)})
	if !r.IsError || !strings.Contains(r.Content, "panicked") || strings.Contains(r.Content, "goroutine") || r.CallID != "6" {
		t.Fatalf("panic: %+v; want an error result without a stack trace", r)
	}
}
