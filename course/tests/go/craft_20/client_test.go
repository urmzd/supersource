//go:build primer && coursepact

// Course tests for the craft.20 kata (primers/craft.20/engineclient.go): the
// gateway's client of the engine API checked against fake engines that
// behave as openai-subset.v1.yaml says an engine does.
//
// The `primer && coursepact` build tags keep this file out of the coursetests
// module and out of your own test runs: craft.20's check copies it next to
// your kata in a scratch module named craft20 and runs it with those tags,
// separately from your consumer tests, which it grades by mutation.
package craft20_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"craft20"
)

const tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-b7ad6b7169203331-01"

type seen struct {
	header http.Header
	path   string
	body   string
}

// engine answers every request with h and records what reached it.
func engine(t *testing.T, h func(w http.ResponseWriter, r *http.Request)) (*httptest.Server, func() seen) {
	t.Helper()
	var mu sync.Mutex
	var last seen
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		last = seen{r.Header.Clone(), r.URL.Path, string(b)}
		mu.Unlock()
		h(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv, func() seen { mu.Lock(); defer mu.Unlock(); return last }
}

func send(t *testing.T, base string, f craft20.Forward) *http.Response {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	t.Cleanup(cancel)
	req, err := (&craft20.Client{Base: base}).NewRequest(ctx, f)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func hostile() http.Header {
	h := http.Header{}
	h.Set("Authorization", "Bearer tl_abcdefghijkl_0123456789abcdefghijklmnopqrstuv")
	h.Set("X-TL-Priority", "99")
	h.Set("X-TL-KV-Handle", "forged")
	h.Set("User-Agent", "openai-python/1.0")
	return h
}

func sse(events ...string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		for _, e := range events {
			io.WriteString(w, e)
			fl.Flush()
		}
	}
}

const (
	role   = `data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"smol-135m@v3","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}` + "\n\n"
	once   = `data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"smol-135m@v3","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}` + "\n\n"
	upon   = `data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"smol-135m@v3","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":"length"}]}` + "\n\n"
	usage  = `data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"smol-135m@v3","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":30,"total_tokens":42}}` + "\n\n"
	done   = "data: [DONE]\n\n"
	failed = `data: {"error":{"message":"lost","type":"server_error","param":null,"code":"internal_error"}}` + "\n\n"
)

func TestCourseHandExampleStream(t *testing.T) {
	// WHY: the chapter's worked example (section 3): the stream of the pact
	//      interaction "a streamed chat with usage" reads as 3 content chunks,
	//      text "Once upon", finish_reason length, and usage 12 + 30 = 42,
	//      with a ": ping" comment and every event split across two writes.
	// KIND: unit
	// CATCHES: s07, s08, m01, m02
	// CHAPTER: craft.20 section 3, Worked example by hand
	split := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		for _, e := range []string{role, ": ping\n\n", once, upon, usage, done} {
			io.WriteString(w, e[:len(e)/2])
			fl.Flush()
			io.WriteString(w, e[len(e)/2:])
			fl.Flush()
		}
	}
	srv, _ := engine(t, split)
	var text string
	sum, err := craft20.ReadStream(send(t, srv.URL, craft20.Forward{Path: "/v1/chat/completions", Body: []byte(`{"stream":true}`)}),
		func(c craft20.Chunk) error { text += c.Content; return nil })
	if err != nil || text != "Once upon" || sum.Chunks != 3 || sum.FinishReason != "length" ||
		sum.Usage == nil || *sum.Usage != (craft20.Usage{PromptTokens: 12, CompletionTokens: 30, TotalTokens: 42}) {
		t.Fatalf("err %v text %q summary %+v usage %+v", err, text, sum, sum.Usage)
	}
}

func TestCourseRequestHeaders(t *testing.T) {
	// WHY: what the engine may see: never the client's key (the engine tier has
	//      no auth and logs headers), never a client-chosen X-TL-Priority or a
	//      forged X-TL-KV-Handle (case priority.internal), always the key's
	//      priority, the request id, and the gateway's traceparent.
	// KIND: conformance
	// CATCHES: s01, s02, s03, s10, s15
	// CHAPTER: craft.20 section 2.2
	srv, last := engine(t, func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, `{"choices":[]}`) })
	resp := send(t, srv.URL, craft20.Forward{Path: "/v1/chat/completions", Body: []byte(`{"model":"smol"}`), Header: hostile(),
		Priority: 5, RequestID: "req-7", Traceparent: tp})
	resp.Body.Close()
	h := last().header
	if h.Get("Authorization") != "" || h.Get("X-Tl-Kv-Handle") != "" || h.Get("X-Tl-Priority") != "5" ||
		h.Get("X-Request-Id") != "req-7" || h.Get("Traceparent") != tp || h.Get("User-Agent") != "openai-python/1.0" {
		t.Fatalf("the engine saw headers %v", h)
	}
}

func TestCourseStreamAsksForUsage(t *testing.T) {
	// WHY: the gateway meters streams, and an engine sends a stream's usage
	//      only for stream_options.include_usage: true, so every streamed
	//      request goes upstream with it (a client's other stream_options kept).
	// KIND: unit
	// CATCHES: s04
	// CHAPTER: craft.20 section 2.2
	srv, last := engine(t, sse(role, done))
	resp := send(t, srv.URL, craft20.Forward{Path: "/v1/chat/completions", Body: []byte(`{"model":"smol","stream":true}`)})
	resp.Body.Close()
	if b := last().body; !strings.Contains(b, `"include_usage":true`) {
		t.Fatalf("the engine received %s", b)
	}
}

func TestCourseStreamEnds(t *testing.T) {
	// WHY: the three ways a stream ends: [DONE] (success), an error event
	//      (an *APIError with the event's code, after the chunks before it),
	//      and the connection closing early (ErrTruncated, never a short
	//      success the gateway would bill and cache).
	// KIND: fault
	// CATCHES: s05, s06
	// CHAPTER: craft.20 section 2.3
	srv, _ := engine(t, sse(role, once, failed))
	n := 0
	_, err := craft20.ReadStream(send(t, srv.URL, craft20.Forward{Path: "/v1/chat/completions", Body: []byte(`{"stream":true}`)}),
		func(craft20.Chunk) error { n++; return nil })
	var ae *craft20.APIError
	if !errors.As(err, &ae) || ae.Code != "internal_error" || ae.Status != 200 || n != 2 {
		t.Fatalf("error event: err %v, %d chunks", err, n)
	}
	srv2, _ := engine(t, sse(role, once))
	if _, err := craft20.ReadStream(send(t, srv2.URL, craft20.Forward{Path: "/v1/chat/completions", Body: []byte(`{"stream":true}`)}), nil); !errors.Is(err, craft20.ErrTruncated) {
		t.Fatalf("cut-off stream: err %v, want ErrTruncated", err)
	}
}

func TestCourseDataWithoutSpace(t *testing.T) {
	// WHY: the SSE grammar allows "data:x" as well as "data: x"; a strict
	//      reader works against one engine and breaks on the next.
	// KIND: boundary
	// CATCHES: s09
	// CHAPTER: craft.20 section 2.3
	srv, _ := engine(t, sse(strings.Replace(once, "data: ", "data:", 1), "data:[DONE]\n\n"))
	var text string
	_, err := craft20.ReadStream(send(t, srv.URL, craft20.Forward{Path: "/v1/chat/completions", Body: []byte(`{"stream":true}`)}),
		func(c craft20.Chunk) error { text += c.Content; return nil })
	if err != nil || text != "Once" {
		t.Fatalf("err %v text %q", err, text)
	}
}

func TestCourseErrorsAndRetryAfter(t *testing.T) {
	// WHY: a refusal is an *APIError with the status, type, and code of the
	//      OpenAI error shape (null fields as ""), and a 429's Retry-After in
	//      seconds, which gw.05 uses to try another worker.
	// KIND: conformance
	// CATCHES: s11, s12, s16
	// CHAPTER: craft.20 section 2.4
	srv, _ := engine(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "3")
		w.WriteHeader(429)
		io.WriteString(w, `{"error":{"message":"full","type":"rate_limit_error","param":null,"code":"rate_limit_exceeded"}}`)
	})
	_, err := craft20.ReadCompletion(send(t, srv.URL, craft20.Forward{Path: "/v1/chat/completions", Body: []byte(`{}`)}))
	var ae *craft20.APIError
	if !errors.As(err, &ae) || ae.Status != 429 || ae.Type != "rate_limit_error" || ae.Code != "rate_limit_exceeded" ||
		ae.Param != "" || ae.RetryAfter != 3*time.Second {
		t.Fatalf("err %v (%+v)", err, ae)
	}
}

func TestCourseCompletionShapes(t *testing.T) {
	// WHY: a chat answer carries choices[0].message.content and a text
	//      completion choices[0].text; both carry usage, which the ledger bills.
	// KIND: unit
	// CATCHES: s14
	// CHAPTER: craft.20 section 2.4
	srv, _ := engine(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"id":"x","object":"text_completion","created":1,"model":"smol","choices":[{"index":0,"text":" upon","finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`)
	})
	c, err := craft20.ReadCompletion(send(t, srv.URL+"/", craft20.Forward{Path: "/v1/completions", Body: []byte(`{}`)}))
	if err != nil || c.Text != " upon" || c.Usage.TotalTokens != 3 {
		t.Fatalf("err %v completion %+v", err, c)
	}
}

func TestCourseTokenize(t *testing.T) {
	// WHY: the gateway prices a request for its TPM limit by asking the engine
	//      to tokenize: POST /v1/tokenize {model, text} answers {ids}.
	// KIND: conformance
	// CATCHES: s13, m03
	// CHAPTER: craft.20 section 2.4
	srv, last := engine(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"ids":[7454,2402,257,640]}`)
	})
	n, err := (&craft20.Client{Base: srv.URL}).Tokenize(context.Background(), "smol", "Once upon a time")
	if err != nil || n != 4 || last().path != "/v1/tokenize" || !strings.Contains(last().body, `"text":"Once upon a time"`) {
		t.Fatalf("n %d err %v, engine saw %s %s", n, err, last().path, last().body)
	}
}

func TestCourseBaseWithTrailingSlash(t *testing.T) {
	// WHY: a base URL configured with a trailing slash must not turn the path
	//      into //v1/chat/completions, which an engine's router answers 404 or 301.
	// KIND: boundary
	// CATCHES: m04
	// CHAPTER: craft.20 section 5, Pitfall 10
	srv, last := engine(t, func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, `{"choices":[]}`) })
	resp := send(t, srv.URL+"/", craft20.Forward{Path: "/v1/chat/completions", Body: []byte(`{}`)})
	resp.Body.Close()
	if p := last().path; p != "/v1/chat/completions" {
		t.Fatalf("the engine saw path %q", p)
	}
}
