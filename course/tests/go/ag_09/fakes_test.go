package ag_09

// Frozen fakes for the eval runner's tests: scorers with known behaviour,
// a scripted chat provider, and a fake clock. None of them calls learner
// code beyond the interfaces under test.

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"sync"
	"time"

	"tinyllm/agent/eval"
	"tinyllm/agent/types"
)

// valueScorer scores the number in the output (a JSON number or a string
// holding one); outputs listed in fail make it return an error.
type valueScorer struct {
	name string
	fail map[string]bool
}

func (s valueScorer) Name() string { return s.name }

func (s valueScorer) Score(_ context.Context, o eval.Observation) (eval.Score, error) {
	if s.fail[o.ID] {
		return eval.Score{}, errors.New("cannot score " + o.ID)
	}
	var v float64
	if err := json.Unmarshal(o.Output, &v); err != nil {
		var str string
		if err := json.Unmarshal(o.Output, &str); err != nil {
			return eval.Score{}, err
		}
		v, err = strconv.ParseFloat(str, 64)
		if err != nil {
			return eval.Score{}, err
		}
	}
	return eval.Score{Value: v}, nil
}

// trickScorer misbehaves in the way its mode says, on every observation.
type trickScorer struct{ name, mode string }

func (s trickScorer) Name() string { return s.name }

func (s trickScorer) Score(_ context.Context, o eval.Observation) (eval.Score, error) {
	switch s.mode {
	case "panic":
		panic("scorer bug")
	case "nan":
		return eval.Score{Value: math.NaN()}, nil
	case "inf":
		return eval.Score{Value: math.Inf(1)}, nil
	case "error-field":
		return eval.Score{Value: 0.25, Error: "judge output unparsable"}, nil
	}
	return eval.Score{Value: 1}, nil
}

func num(v float64) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

func str(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

// fakeClock advances by step on every call.
type fakeClock struct {
	mu   sync.Mutex
	t    time.Time
	step time.Duration
}

func (c *fakeClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(c.step)
	return c.t
}

type reply struct {
	text  []string
	calls []types.ToolCall
	usage types.Usage
	fail  error // an error delta after the text
}

// script is a fake Provider: call i gets replies[i] (the last repeats).
type script struct {
	mu      sync.Mutex
	replies []reply
	n       int
}

func (s *script) ChatStream(ctx context.Context, msgs []types.Message, tools []types.ToolDef, _ ...types.CallOption) (<-chan types.Delta, error) {
	s.mu.Lock()
	r := s.replies[min(s.n, len(s.replies)-1)]
	s.n++
	s.mu.Unlock()
	ch := make(chan types.Delta, 64)
	go func() {
		defer close(ch)
		for _, t := range r.text {
			ch <- types.TextDelta{Text: t}
		}
		for i, c := range r.calls {
			ch <- types.ToolCallStartDelta{Index: i, ID: c.ID, Name: c.Name}
			ch <- types.ToolCallEndDelta{Index: i, Call: c}
		}
		if r.usage != (types.Usage{}) {
			ch <- types.UsageDelta{In: r.usage.In, Out: r.usage.Out}
		}
		if r.fail != nil {
			ch <- types.ErrorDelta{Err: r.fail}
			return
		}
		finish := "stop"
		if len(r.calls) > 0 {
			finish = "tool_calls"
		}
		ch <- types.DoneDelta{FinishReason: finish}
	}()
	return ch, nil
}

func (s *script) calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.n
}
