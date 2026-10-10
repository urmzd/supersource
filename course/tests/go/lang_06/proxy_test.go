//go:build primer

// Course tests for the lang.06 primer (primers/lang.06/proxy.go).
//
// The `primer` build tag keeps this file out of the coursetests module: the
// artifact check (course/tests/lang.06/check) copies it next to your primer
// in a scratch module named lang06 and runs `go test -tags primer`.
//
// Reading guide: every test starts real servers with net/http/httptest and
// coordinates them with channels. Nothing sleeps for a fixed time; a test
// waits on a channel and gives up after `patience`.
package proxy_test

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"testing/iotest"
	"time"

	proxy "lang06"
)

const patience = 3 * time.Second

func serve(t *testing.T, h http.Handler) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(h)
	t.Cleanup(s.Close)
	return s
}

// get is bounded by patience, so a proxy that holds the response headers
// back fails the test instead of hanging it.
func get(t *testing.T, ctx context.Context, url string) *http.Response {
	t.Helper()
	ctx, cancel := context.WithTimeout(ctx, patience)
	t.Cleanup(cancel)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET %s through the proxy: %v (no headers within %v: flush after writing them)", url, err, patience)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

// readWithin reads the whole body, or fails if it has not ended after limit.
func readWithin(t *testing.T, r io.Reader, limit time.Duration) string {
	t.Helper()
	type result struct {
		s   string
		err error
	}
	done := make(chan result, 1)
	go func() {
		b, err := io.ReadAll(r)
		done <- result{string(b), err}
	}()
	select {
	case res := <-done:
		return res.s // a cut stream may end with an error; the bytes are what count
	case <-time.After(limit):
		t.Fatalf("the body was still open after %v", limit)
		return ""
	}
}

func TestForwardsAndCopiesBack(t *testing.T) {
	// WHY: a proxy is invisible: the upstream sees the client's method, path,
	//      query, headers, and body; the client sees the upstream's status,
	//      headers, and body.
	// KIND: unit
	up := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		if r.Method != http.MethodPut || r.URL.Path != "/v1/items" || r.URL.RawQuery != "id=7" ||
			string(b) != "payload" || r.Header.Get("X-Client") != "primer" {
			http.Error(w, "upstream saw "+r.Method+" "+r.URL.String()+" "+string(b), http.StatusTeapot)
			return
		}
		w.Header().Set("X-Upstream", "yes")
		w.WriteHeader(http.StatusCreated)
		io.WriteString(w, "created")
	}))
	p := serve(t, proxy.New(up.URL, patience))
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, p.URL+"/v1/items?id=7", strings.NewReader("payload"))
	req.Header.Set("X-Client", "primer")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusCreated || resp.Header.Get("X-Upstream") != "yes" || string(body) != "created" {
		t.Fatalf("got %d %q (X-Upstream %q), want 201 \"created\" (X-Upstream \"yes\")",
			resp.StatusCode, body, resp.Header.Get("X-Upstream"))
	}
}

func TestFlushesEachChunk(t *testing.T) {
	// WHY: streaming means the client gets each chunk when the upstream sends
	//      it. The upstream holds its second chunk until the client has read
	//      the first through the proxy, so a proxy that buffers cannot pass.
	// KIND: unit
	release := make(chan struct{})
	var once sync.Once
	free := func() { once.Do(func() { close(release) }) }
	up := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "data: 1\n\n")
		http.NewResponseController(w).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		io.WriteString(w, "data: 2\n\n")
	}))
	p := serve(t, proxy.New(up.URL, patience))
	t.Cleanup(free)
	resp := get(t, context.Background(), p.URL)
	br := bufio.NewReader(resp.Body)
	first := make(chan string, 1)
	go func() {
		a, _ := br.ReadString('\n')
		b, _ := br.ReadString('\n')
		first <- a + b
	}()
	select {
	case got := <-first:
		if got != "data: 1\n\n" {
			t.Fatalf("first chunk = %q", got)
		}
	case <-time.After(patience):
		t.Fatal("the first chunk never arrived while the upstream waited: the proxy is buffering (flush after each write)")
	}
	free()
	if rest := readWithin(t, br, patience); rest != "data: 2\n\n" {
		t.Fatalf("second chunk = %q", rest)
	}
}

func TestSlowHeadersGet504(t *testing.T) {
	// WHY: the deadline protects the client from an upstream that never
	//      answers: no headers by the deadline means 504, and the upstream's
	//      request is cancelled so it stops working too.
	// KIND: boundary
	cancelled := make(chan struct{})
	up := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			close(cancelled)
		case <-time.After(2 * patience):
		}
	}))
	p := serve(t, proxy.New(up.URL, 100*time.Millisecond))
	resp := get(t, context.Background(), p.URL)
	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("status = %d, want 504", resp.StatusCode)
	}
	select {
	case <-cancelled:
	case <-time.After(patience):
		t.Fatal("the upstream request was not cancelled at the deadline")
	}
}

func TestDeadlineCutsALongStream(t *testing.T) {
	// WHY: the deadline also covers the body. An upstream that sends one
	//      chunk and then stalls must not keep the client hanging: the body
	//      ends at the deadline, after the chunk that did arrive.
	// KIND: boundary
	up := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "partial")
		http.NewResponseController(w).Flush()
		select {
		case <-r.Context().Done():
		case <-time.After(2 * patience):
		}
	}))
	p := serve(t, proxy.New(up.URL, 200*time.Millisecond))
	resp := get(t, context.Background(), p.URL)
	// Half of patience: the request itself is bounded by patience, and the
	// proxy's own deadline must end the body well before that.
	if got := readWithin(t, resp.Body, patience/2); got != "partial" {
		t.Fatalf("body = %q, want the chunk sent before the deadline", got)
	}
}

func TestDeadlineCountsFromTheStart(t *testing.T) {
	// WHY: a deadline is a point in time, not an idle timeout. An upstream
	//      that trickles a byte every 20 ms is never idle, yet the body must
	//      still end once `timeout` has passed since the request started.
	// KIND: boundary
	up := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tick := time.NewTicker(20 * time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-r.Context().Done():
				return
			case <-tick.C:
				io.WriteString(w, ".")
				http.NewResponseController(w).Flush()
			}
		}
	}))
	p := serve(t, proxy.New(up.URL, 300*time.Millisecond))
	resp := get(t, context.Background(), p.URL)
	if got := readWithin(t, resp.Body, patience/2); len(got) == 0 {
		t.Fatal("no bytes arrived before the deadline")
	}
}

func TestClientGoneCancelsUpstream(t *testing.T) {
	// WHY: when the client hangs up, the upstream request must stop too, or
	//      abandoned requests pile up. Derive the deadline from the request's
	//      own context, never from context.Background().
	// KIND: fault
	cancelled := make(chan struct{})
	up := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "hello\n")
		http.NewResponseController(w).Flush()
		select {
		case <-r.Context().Done():
			close(cancelled)
		case <-time.After(2 * patience):
		}
	}))
	p := serve(t, proxy.New(up.URL, time.Minute))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resp := get(t, ctx, p.URL)
	if line, _ := bufio.NewReader(resp.Body).ReadString('\n'); line != "hello\n" {
		t.Fatalf("first line = %q", line)
	}
	cancel()
	select {
	case <-cancelled:
	case <-time.After(patience):
		t.Fatalf("the upstream request was still running %v after the client left", patience)
	}
}

func TestUnreachableIs502(t *testing.T) {
	// WHY: an upstream that refuses the connection is a different failure
	//      from a slow one: 502 Bad Gateway, not 504, and not a hang.
	// KIND: boundary
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	dead := "http://" + l.Addr().String()
	l.Close()
	p := serve(t, proxy.New(dead, patience))
	if resp := get(t, context.Background(), p.URL); resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", resp.StatusCode)
	}
}

// flushCounter is an http.ResponseWriter that counts Flush calls.
type flushCounter struct {
	*httptest.ResponseRecorder
	flushes int
}

func (f *flushCounter) Flush() { f.flushes++; f.ResponseRecorder.Flush() }

func TestCopyFlush(t *testing.T) {
	// WHY: CopyFlush is the loop at the heart of streaming: write what each
	//      Read returned, flush, repeat. It counts what it wrote, treats
	//      io.EOF as success, and returns any other read error.
	// KIND: unit
	w := &flushCounter{ResponseRecorder: httptest.NewRecorder()}
	n, err := proxy.CopyFlush(w, iotest.OneByteReader(strings.NewReader("abc")))
	if n != 3 || err != nil || w.Body.String() != "abc" {
		t.Fatalf("CopyFlush = (%d, %v), body %q; want (3, nil), \"abc\"", n, err, w.Body.String())
	}
	if w.flushes < 3 {
		t.Fatalf("%d flushes for 3 one-byte reads; flush after every write", w.flushes)
	}
	boom := errors.New("upstream reset")
	w = &flushCounter{ResponseRecorder: httptest.NewRecorder()}
	n, err = proxy.CopyFlush(w, io.MultiReader(strings.NewReader("ab"), iotest.ErrReader(boom)))
	if n != 2 || !errors.Is(err, boom) || w.Body.String() != "ab" {
		t.Fatalf("CopyFlush over a failing reader = (%d, %v), body %q; want (2, upstream reset), \"ab\"", n, err, w.Body.String())
	}
}
