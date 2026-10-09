// Course tests for gw.00, the tracer gateway (go/gateway/proxy).
//
// Every test starts a fake engine with net/http/httptest, puts your
// proxy.NewProxy in front of it, and talks to the gateway as a client would.
// Nothing sleeps for a fixed time: a test waits on a channel and gives up
// after `patience`, so a correct proxy passes in milliseconds and a broken
// one fails with a message instead of hanging.
package gw_00

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"tinyllm/gateway/proxy"
)

const (
	key      = "tl_test_7f3a9c"
	patience = 3 * time.Second
	// The W3C trace-context example (https://www.w3.org/TR/trace-context/).
	callerTrace  = "4bf92f3577b34da6a3ce929d0e0e4736"
	callerParent = "00f067aa0ba902b7"
)

// v0Stream is a recorded engine answer to a streamed /v1/completions request
// (openai-subset.v0.yaml): one event per token, the last one with
// finish_reason, then [DONE]. The fake engine sends it one event at a time.
var v0Stream = []string{
	`data: {"id":"cmpl-7","object":"text_completion","created":1760000000,"model":"tracer","choices":[{"index":0,"text":"H","finish_reason":null}]}` + "\n\n",
	`data: {"id":"cmpl-7","object":"text_completion","created":1760000000,"model":"tracer","choices":[{"index":0,"text":"","finish_reason":null}]}` + "\n\n",
	`data: {"id":"cmpl-7","object":"text_completion","created":1760000000,"model":"tracer","choices":[{"index":0,"text":"é","finish_reason":null}]}` + "\n\n",
	`data: {"id":"cmpl-7","object":"text_completion","created":1760000000,"model":"tracer","choices":[{"index":0,"text":"!","finish_reason":"length"}]}` + "\n\n",
	"data: [DONE]\n\n",
}

const completionBody = `{"model":"tracer","prompt":"Once upon a time","max_tokens":4,"stream":true}`

// engine is a fake upstream that records every request that reaches it.
type engine struct {
	srv  *httptest.Server
	mu   sync.Mutex
	reqs []seenReq
}

type seenReq struct {
	method, path, query string
	header              http.Header
	body                string
}

func newEngine(t *testing.T, h http.HandlerFunc) *engine {
	t.Helper()
	e := &engine{}
	e.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		e.mu.Lock()
		e.reqs = append(e.reqs, seenReq{r.Method, r.URL.Path, r.URL.RawQuery, r.Header.Clone(), string(b)})
		e.mu.Unlock()
		h(w, r)
	}))
	t.Cleanup(e.srv.Close)
	return e
}

func (e *engine) count() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.reqs)
}

func (e *engine) last(t *testing.T) seenReq {
	t.Helper()
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.reqs) == 0 {
		t.Fatal("the request never reached the engine")
	}
	return e.reqs[len(e.reqs)-1]
}

// gateway puts NewProxy in front of upstream with the test key.
func gateway(t *testing.T, cfg proxy.Config) *httptest.Server {
	t.Helper()
	if cfg.APIKey == "" {
		cfg.APIKey = key
	}
	g := httptest.NewServer(proxy.NewProxy(cfg))
	t.Cleanup(g.Close)
	return g
}

// post sends the worked example's request through the gateway. The whole
// exchange is bounded by patience, so a gateway that holds the response
// headers back fails the test instead of hanging it.
func post(t *testing.T, ctx context.Context, url, auth string, hdr map[string]string) *http.Response {
	t.Helper()
	ctx, cancel := context.WithTimeout(ctx, patience)
	t.Cleanup(cancel)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/v1/completions", strings.NewReader(completionBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if auth != "" {
		req.Header.Set("Authorization", auth)
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request through the gateway failed: %v (no response headers within %v means the gateway holds them back: flush)", err, patience)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func readAll(t *testing.T, r io.Reader) string {
	t.Helper()
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("reading the gateway's body: %v", err)
	}
	return string(b)
}

// streamEvents is an engine that sends v0Stream one flushed event at a time.
func streamEvents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("X-Request-Id", r.Header.Get("X-Request-Id"))
	for _, ev := range v0Stream {
		io.WriteString(w, ev)
		http.NewResponseController(w).Flush()
	}
}

// firstEvent reads up to the first blank line, or fails after patience.
func firstEvent(t *testing.T, br *bufio.Reader) string {
	t.Helper()
	got := make(chan string, 1)
	go func() {
		var sb strings.Builder
		for {
			line, err := br.ReadString('\n')
			sb.WriteString(line)
			if err != nil || line == "\n" {
				got <- sb.String()
				return
			}
		}
	}()
	select {
	case s := <-got:
		return s
	case <-time.After(patience):
		t.Fatalf("no complete SSE event within %v: the gateway is holding the stream back", patience)
		return ""
	}
}

type apiError struct {
	Error map[string]any `json:"error"`
}

// checkError asserts the OpenAI error shape of openai-subset.v0.yaml.
func checkError(t *testing.T, resp *http.Response, status int, typ, code string) {
	t.Helper()
	if resp.StatusCode != status {
		t.Fatalf("status = %d, want %d", resp.StatusCode, status)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", ct)
	}
	var e apiError
	body := readAll(t, resp.Body)
	if err := json.Unmarshal([]byte(body), &e); err != nil || e.Error == nil {
		t.Fatalf("body is not {\"error\": {...}}: %q", body)
	}
	for _, k := range []string{"message", "type", "param", "code"} {
		if _, ok := e.Error[k]; !ok {
			t.Fatalf("error object lacks %q (all four keys are required, null allowed): %q", k, body)
		}
	}
	if e.Error["type"] != typ || e.Error["code"] != code {
		t.Fatalf("error type/code = %v/%v, want %s/%s", e.Error["type"], e.Error["code"], typ, code)
	}
	if e.Error["param"] != nil {
		t.Fatalf("error param = %v, want null", e.Error["param"])
	}
}

func TestRejectsMissingKey(t *testing.T) {
	// WHY: the gateway is the only door to the engine; a request without the
	//      key must stop here, in the OpenAI error shape every client parses.
	// KIND: unit
	// CHAPTER: gw.00 section 3, worked example, request 1
	eng := newEngine(t, streamEvents)
	g := gateway(t, proxy.Config{Upstream: eng.srv.URL})
	resp := post(t, context.Background(), g.URL, "", nil)
	checkError(t, resp, http.StatusUnauthorized, "invalid_request_error", "invalid_api_key")
	if resp.Header.Get("X-Request-Id") == "" {
		t.Fatal("a 401 still carries X-Request-Id, so the client can quote it")
	}
	if n := eng.count(); n != 0 {
		t.Fatalf("the engine received %d request(s); the key check must come before any upstream call", n)
	}
}

func TestRejectsWrongKeys(t *testing.T) {
	// WHY: only the exact key opens the door. A prefix, an extension, another
	//      scheme, or a different case is a different key.
	// KIND: boundary
	// CATCHES: s01, s06
	// CHAPTER: gw.00 section 5, Pitfalls, item 1
	eng := newEngine(t, streamEvents)
	g := gateway(t, proxy.Config{Upstream: eng.srv.URL})
	for _, auth := range []string{
		"Bearer",
		"Bearer ",
		"Bearer tl_test",        // a prefix of the key
		"Bearer " + key + "0",   // the key with one more byte
		"Bearer TL_TEST_7F3A9C", // the key in another case
		"Basic " + key,          // another scheme
		key,                     // no scheme at all
	} {
		resp := post(t, context.Background(), g.URL, auth, nil)
		if resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("Authorization %q: status %d, want 401", auth, resp.StatusCode)
		}
	}
	if n := eng.count(); n != 0 {
		t.Fatalf("the engine received %d request(s) with a wrong key", n)
	}
	// The control: the right key goes through.
	if resp := post(t, context.Background(), g.URL, "Bearer "+key, nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("the right key: status %d, want 200", resp.StatusCode)
	}
}

func TestEmptyConfiguredKeyRejectsEverything(t *testing.T) {
	// WHY: a gateway started without TL_API_KEY must fail closed. With an
	//      empty key, "Bearer " compares equal to "" and opens the door.
	// KIND: boundary
	// CATCHES: s06
	// CHAPTER: gw.00 section 5, Pitfalls, item 2
	eng := newEngine(t, streamEvents)
	g := httptest.NewServer(proxy.NewProxy(proxy.Config{Upstream: eng.srv.URL, APIKey: ""}))
	defer g.Close()
	for _, auth := range []string{"", "Bearer ", "Bearer x"} {
		if resp := post(t, context.Background(), g.URL, auth, nil); resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("APIKey empty, Authorization %q: status %d, want 401", auth, resp.StatusCode)
		}
	}
	if n := eng.count(); n != 0 {
		t.Fatalf("the engine received %d request(s) through a gateway with no key", n)
	}
}

func TestForwardsRequestWithoutTheKey(t *testing.T) {
	// WHY: the engine must see the client's method, path, query, headers, and
	//      body, but never the gateway's key: the engine tier has no auth, and
	//      a key that travels further leaks into engine logs.
	// KIND: unit
	// CATCHES: s03, s10
	// CHAPTER: gw.00 section 4, The interface
	for _, base := range []string{"", "/"} { // Upstream with and without a trailing slash
		eng := newEngine(t, func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "{}") })
		g := gateway(t, proxy.Config{Upstream: eng.srv.URL + base})
		ctx, cancel := context.WithTimeout(context.Background(), patience)
		defer cancel()
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, g.URL+"/v1/completions?trace=1", strings.NewReader(completionBody))
		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Client", "curl")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		readAll(t, resp.Body)
		resp.Body.Close()
		got := eng.last(t)
		if got.method != http.MethodPost || got.path != "/v1/completions" || got.query != "trace=1" {
			t.Fatalf("Upstream %q: engine saw %s %s?%s, want POST /v1/completions?trace=1", eng.srv.URL+base, got.method, got.path, got.query)
		}
		if got.body != completionBody {
			t.Fatalf("engine body = %q, want the client's body unchanged", got.body)
		}
		if got.header.Get("Content-Type") != "application/json" || got.header.Get("X-Client") != "curl" {
			t.Fatalf("end-to-end headers not forwarded: %v", got.header)
		}
		if a := got.header.Get("Authorization"); a != "" {
			t.Fatalf("the engine received Authorization %q; strip the gateway's key before forwarding", a)
		}
	}
}

func TestStreamsWithoutBuffering(t *testing.T) {
	// WHY: a streamed completion is only useful if each token reaches the
	//      client when the engine emits it. The fake engine holds its second
	//      event until the client has read the first through the gateway, so
	//      a gateway that buffers can never pass.
	// KIND: unit
	// CATCHES: s02
	// CHAPTER: gw.00 section 2, Principles, why a proxy must flush
	release := make(chan struct{})
	var once sync.Once
	free := func() { once.Do(func() { close(release) }) }
	defer free()
	eng := newEngine(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, v0Stream[0])
		http.NewResponseController(w).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		for _, ev := range v0Stream[1:] {
			io.WriteString(w, ev)
			http.NewResponseController(w).Flush()
		}
	})
	g := gateway(t, proxy.Config{Upstream: eng.srv.URL})
	t.Cleanup(free) // runs before the servers close, so no handler is left waiting
	resp := post(t, context.Background(), g.URL, "Bearer "+key, nil)
	br := bufio.NewReader(resp.Body)
	if ev := firstEvent(t, br); ev != v0Stream[0] {
		t.Fatalf("first event = %q, want %q", ev, v0Stream[0])
	}
	free()
	if rest := readAll(t, br); rest != strings.Join(v0Stream[1:], "") {
		t.Fatalf("rest of the stream = %q", rest)
	}
}

func TestV0StreamPassesThroughByteExact(t *testing.T) {
	// WHY: the gateway forwards bytes, it does not re-frame SSE. The recorded
	//      v0 stream (including an empty-text chunk held for UTF-8) must reach
	//      the client byte for byte, as text/event-stream with status 200.
	// KIND: conformance
	// CHAPTER: gw.00 section 3, worked example, request 2
	eng := newEngine(t, streamEvents)
	g := gateway(t, proxy.Config{Upstream: eng.srv.URL})
	resp := post(t, context.Background(), g.URL, "Bearer "+key, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}
	if got, want := readAll(t, resp.Body), strings.Join(v0Stream, ""); got != want {
		t.Fatalf("stream differs from what the engine sent:\n got %q\nwant %q", got, want)
	}
}

func TestV0ResponsesPassThroughUnchanged(t *testing.T) {
	// WHY: an engine answer that is not a stream (a completion, or a 400 in
	//      the error shape) reaches the client with its status, headers, and
	//      body unchanged. Writing the body before the status turns every
	//      engine error into a 200.
	// KIND: conformance
	// CATCHES: s11
	// CHAPTER: gw.00 section 5, Pitfalls, item 6
	cases := []struct {
		status int
		body   string
	}{
		{200, `{"id":"cmpl-1","object":"text_completion","created":1760000000,"model":"tracer","choices":[{"index":0,"text":"He","finish_reason":"length"}],"usage":{"prompt_tokens":16,"completion_tokens":2,"total_tokens":18}}`},
		{400, `{"error":{"message":"temperature must be in [0, 2]","type":"invalid_request_error","param":"temperature","code":null}}`},
	}
	for _, c := range cases {
		eng := newEngine(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Engine", "tracer")
			w.WriteHeader(c.status)
			io.WriteString(w, c.body)
		})
		g := gateway(t, proxy.Config{Upstream: eng.srv.URL})
		resp := post(t, context.Background(), g.URL, "Bearer "+key, nil)
		if resp.StatusCode != c.status {
			t.Fatalf("engine answered %d, client got %d", c.status, resp.StatusCode)
		}
		if resp.Header.Get("X-Engine") != "tracer" || resp.Header.Get("Content-Type") != "application/json" {
			t.Fatalf("engine response headers not copied: %v", resp.Header)
		}
		if got := readAll(t, resp.Body); got != c.body {
			t.Fatalf("status %d body = %q, want %q", c.status, got, c.body)
		}
	}
}

func TestUpstreamDownIs503(t *testing.T) {
	// WHY: when the engine cannot be reached before the first byte, the
	//      contract answer is 503 server_error/no_capacity in the error shape,
	//      not a hung request or a plain-text 502.
	// KIND: boundary
	// CATCHES: s08
	// CHAPTER: gw.00 section 5, Pitfalls, item 7
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	dead := "http://" + l.Addr().String()
	l.Close() // nothing listens there now: connection refused
	g := gateway(t, proxy.Config{Upstream: dead})
	resp := post(t, context.Background(), g.URL, "Bearer "+key, nil)
	checkError(t, resp, http.StatusServiceUnavailable, "server_error", "no_capacity")
}

func TestUpstreamFailureMidStreamSendsErrorEvent(t *testing.T) {
	// WHY: once 200 and the first event are sent, the status cannot change.
	//      If the engine dies mid-stream the gateway must say so with an SSE
	//      error event; silently ending looks like a short, valid answer.
	// KIND: fault
	// CATCHES: s09
	// CHAPTER: gw.00 section 5, Pitfalls, item 8
	eng := newEngine(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, v0Stream[0])
		rc := http.NewResponseController(w)
		rc.Flush()
		conn, _, err := rc.Hijack() // the engine process "dies": the socket closes mid-response
		if err == nil {
			conn.Close()
		}
	})
	g := gateway(t, proxy.Config{Upstream: eng.srv.URL})
	resp := post(t, context.Background(), g.URL, "Bearer "+key, nil)
	body := readAll(t, resp.Body)
	if !strings.HasPrefix(body, v0Stream[0]) {
		t.Fatalf("the event sent before the failure was lost: %q", body)
	}
	rest := strings.TrimPrefix(body, v0Stream[0])
	if !strings.HasPrefix(rest, "data: ") || !strings.HasSuffix(rest, "\n\n") {
		t.Fatalf("after the failure the client got %q, want one `data: {\"error\": ...}` event", rest)
	}
	var e apiError
	if err := json.Unmarshal([]byte(strings.TrimSuffix(strings.TrimPrefix(rest, "data: "), "\n\n")), &e); err != nil || e.Error == nil {
		t.Fatalf("the final event is not an error object: %q", rest)
	}
	if e.Error["type"] != "server_error" {
		t.Fatalf("error event type = %v, want server_error", e.Error["type"])
	}
}

func TestRequestID(t *testing.T) {
	// WHY: X-Request-Id ties a client's complaint to the gateway's and the
	//      engine's logs. The client's id is kept when it is sane; otherwise
	//      the gateway makes one. Either way the engine and the client see
	//      the same id.
	// KIND: unit
	// CATCHES: s07
	// CHAPTER: gw.00 section 3, worked example, request 2
	eng := newEngine(t, streamEvents)
	g := gateway(t, proxy.Config{Upstream: eng.srv.URL})
	seen := map[string]bool{}
	for _, c := range []struct{ in, want string }{
		{"", ""},                       // none sent: generated
		{"req-42", "req-42"},           // sane: kept
		{"has space", ""},              // not printable ASCII: replaced
		{strings.Repeat("a", 129), ""}, // over 128 bytes: replaced
		{strings.Repeat("b", 128), strings.Repeat("b", 128)}, // exactly 128: kept
	} {
		hdr := map[string]string{}
		if c.in != "" {
			hdr["X-Request-Id"] = c.in
		}
		resp := post(t, context.Background(), g.URL, "Bearer "+key, hdr)
		readAll(t, resp.Body)
		out := resp.Header.Get("X-Request-Id")
		up := eng.last(t).header.Get("X-Request-Id")
		if out == "" || out != up {
			t.Fatalf("in %q: client got X-Request-Id %q, engine got %q; both must carry the same id", c.in, out, up)
		}
		if c.want != "" && out != c.want {
			t.Fatalf("in %q: id %q, want the client's id kept", c.in, out)
		}
		if c.want == "" && (out == c.in || seen[out]) {
			t.Fatalf("in %q: id %q, want a fresh generated id", c.in, out)
		}
		seen[out] = true
	}
}

func TestTraceparentChildOfCaller(t *testing.T) {
	// WHY: one trace must span client, gateway, and engine. The engine's span
	//      has to hang under the gateway's span, so the gateway sends the
	//      caller's trace id with its OWN span id, keeps the flags and
	//      tracestate, and reports that span through OnSpan.
	// KIND: unit
	// CATCHES: s04
	// CHAPTER: gw.00 section 3, worked example, request 2
	spans := make(chan proxy.Span, 1)
	eng := newEngine(t, streamEvents)
	g := gateway(t, proxy.Config{Upstream: eng.srv.URL, OnSpan: func(s proxy.Span) { spans <- s }})
	resp := post(t, context.Background(), g.URL, "Bearer "+key, map[string]string{
		"traceparent": "00-" + callerTrace + "-" + callerParent + "-01",
		"tracestate":  "vendor=v1",
	})
	readAll(t, resp.Body)
	tp := eng.last(t).header.Get("Traceparent")
	trace, span, flags, ok := proxy.ParseTraceparent(tp)
	if !ok || trace != callerTrace || flags != "01" {
		t.Fatalf("engine traceparent = %q, want 00-%s-<new span id>-01", tp, callerTrace)
	}
	if span == callerParent {
		t.Fatalf("engine traceparent reuses the caller's span id %s: the engine span would be a sibling of the gateway span, not its child", span)
	}
	if ts := eng.last(t).header.Get("Tracestate"); ts != "vendor=v1" {
		t.Fatalf("tracestate = %q, want it forwarded unchanged with a valid traceparent", ts)
	}
	select {
	case s := <-spans:
		if s.Name != "gateway.proxy" || s.TraceID != callerTrace || s.ParentSpanID != callerParent || s.SpanID != span {
			t.Fatalf("OnSpan got %+v, want name gateway.proxy, trace %s, parent %s, span %s", s, callerTrace, callerParent, span)
		}
		if s.RequestID != resp.Header.Get("X-Request-Id") || s.StatusCode != http.StatusOK {
			t.Fatalf("OnSpan request id %q status %d, want %q and 200", s.RequestID, s.StatusCode, resp.Header.Get("X-Request-Id"))
		}
		if s.End.Before(s.Start) || s.Start.IsZero() {
			t.Fatalf("span times start %v end %v", s.Start, s.End)
		}
	case <-time.After(patience):
		t.Fatal("OnSpan was never called")
	}
}

func TestTraceparentStartsTrace(t *testing.T) {
	// WHY: with no traceparent, or one that breaks the W3C rules, the gateway
	//      starts a new trace (new trace id, no parent, sampled) instead of
	//      forwarding garbage the engine would reject.
	// KIND: boundary
	// CATCHES: s04, s12
	// CHAPTER: gw.00 section 2, Principles, W3C trace context
	for _, in := range []string{
		"",
		"garbage",
		"00-00000000000000000000000000000000-" + callerParent + "-01", // all-zero trace id
		"00-" + strings.ToUpper(callerTrace) + "-" + callerParent + "-01",
	} {
		spans := make(chan proxy.Span, 1)
		eng := newEngine(t, streamEvents)
		g := gateway(t, proxy.Config{Upstream: eng.srv.URL, OnSpan: func(s proxy.Span) { spans <- s }})
		hdr := map[string]string{}
		if in != "" {
			hdr["traceparent"] = in
		}
		readAll(t, post(t, context.Background(), g.URL, "Bearer "+key, hdr).Body)
		tp := eng.last(t).header.Get("Traceparent")
		trace, span, flags, ok := proxy.ParseTraceparent(tp)
		if !ok || flags != "01" || strings.Contains(in, trace) {
			t.Fatalf("in %q: engine traceparent %q, want a valid, sampled, new trace", in, tp)
		}
		select {
		case s := <-spans:
			if s.ParentSpanID != "" || s.TraceID != trace || s.SpanID != span {
				t.Fatalf("in %q: OnSpan %+v, want a root span (no parent) for trace %s span %s", in, s, trace, span)
			}
		case <-time.After(patience):
			t.Fatalf("in %q: OnSpan was never called", in)
		}
	}
}

func TestParseTraceparent(t *testing.T) {
	// WHY: the W3C rules, one per row: version 00 only, lowercase hex, exact
	//      lengths, and all-zero ids are invalid.
	// KIND: unit
	// CATCHES: s12
	// CHAPTER: gw.00 section 2, Principles, W3C trace context
	valid := "00-" + callerTrace + "-" + callerParent + "-01"
	if tr, p, f, ok := proxy.ParseTraceparent(valid); !ok || tr != callerTrace || p != callerParent || f != "01" {
		t.Fatalf("ParseTraceparent(%q) = %q %q %q %v", valid, tr, p, f, ok)
	}
	for _, bad := range []string{
		"",
		"01-" + callerTrace + "-" + callerParent + "-01", // unknown version
		"ff-" + callerTrace + "-" + callerParent + "-01", // forbidden version
		"00-" + strings.ToUpper(callerTrace) + "-" + callerParent + "-01",
		"00-00000000000000000000000000000000-" + callerParent + "-01",
		"00-" + callerTrace + "-0000000000000000-01",
		"00-" + callerTrace[:31] + "-" + callerParent + "-01",
		"00-" + callerTrace + "-" + callerParent + "-1",
		"00-" + callerTrace + "-" + callerParent[:15] + "g-01",
		"00-" + callerTrace + "-" + callerParent,
	} {
		if _, _, _, ok := proxy.ParseTraceparent(bad); ok {
			t.Fatalf("ParseTraceparent(%q) ok, want invalid", bad)
		}
	}
}

func TestClientDisconnectCancelsUpstream(t *testing.T) {
	// WHY: a client that hangs up mid-stream must stop the engine's work, or
	//      every abandoned request keeps generating. The upstream request has
	//      to run on the client's request context.
	// KIND: fault
	// CATCHES: s05
	// CHAPTER: gw.00 section 5, Pitfalls, item 5
	cancelled := make(chan struct{})
	eng := newEngine(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, v0Stream[0])
		http.NewResponseController(w).Flush()
		select {
		case <-r.Context().Done():
			close(cancelled)
		case <-time.After(2 * patience):
		}
	})
	g := gateway(t, proxy.Config{Upstream: eng.srv.URL})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resp := post(t, ctx, g.URL, "Bearer "+key, nil)
	firstEvent(t, bufio.NewReader(resp.Body))
	cancel()
	select {
	case <-cancelled:
	case <-time.After(patience):
		t.Fatalf("the engine request was still running %v after the client left", patience)
	}
}
