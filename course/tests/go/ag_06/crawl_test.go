package ag_06

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"tinyllm/agent/rag/source"
)

// fakeClock moves only when the crawler sleeps, and records every sleep.
type fakeClock struct {
	mu    sync.Mutex
	now   time.Time
	slept []time.Duration
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Sleep(d time.Duration) {
	c.mu.Lock()
	c.now = c.now.Add(d)
	c.slept = append(c.slept, d)
	c.mu.Unlock()
}

// site is a test web site: path -> (status, content type, body); every hit is
// logged with the fake clock's time.
type site struct {
	mu    sync.Mutex
	pages map[string][3]string
	hits  []string
	at    []time.Time
	clock *fakeClock
	srv   *httptest.Server
}

func newSite(t *testing.T, clock *fakeClock, pages map[string][3]string) *site {
	s := &site{pages: pages, clock: clock}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.hits = append(s.hits, r.URL.Path)
		if clock != nil {
			s.at = append(s.at, clock.Now())
		}
		s.mu.Unlock()
		p, ok := s.pages[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		if strings.HasPrefix(p[0], "3") {
			http.Redirect(w, r, p[2], 302)
			return
		}
		w.Header().Set("Content-Type", p[1])
		var code int
		fmt.Sscanf(p[0], "%d", &code)
		w.WriteHeader(code)
		w.Write([]byte(p[2]))
	}))
	t.Cleanup(s.srv.Close)
	return s
}

func (s *site) port() string { _, p, _ := net.SplitHostPort(s.srv.Listener.Addr().String()); return p }

func (s *site) hitList() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.hits...)
}

// resolver maps the test's host names; anything else does not exist.
func resolver(m map[string][]string) func(context.Context, string) ([]netip.Addr, error) {
	return func(_ context.Context, host string) ([]netip.Addr, error) {
		var out []netip.Addr
		for _, a := range m[host] {
			out = append(out, netip.MustParseAddr(a))
		}
		if len(out) == 0 {
			return nil, fmt.Errorf("no such host %s", host)
		}
		return out, nil
	}
}

// loopbackOK lets the crawler reach the test servers on 127.0.0.1, and
// nothing else that PublicAddr refuses.
func loopbackOK(a netip.Addr) bool { return a.IsLoopback() || source.PublicAddr(a) }

const html = "text/html; charset=utf-8"

func TestCrawlerFetches(t *testing.T) {
	// WHY: the practice go/01 crawler, as an ingest source: breadth first
	//      from the seed, links resolved against the page, fragments dropped
	//      so /a and /a#top are one page, hosts outside the allowlist (an
	//      outside site, a lookalike that starts with an allowed name)
	//      never requested, robots.txt Disallow honored, and every skip
	//      listed with its reason.
	// KIND: unit
	// CATCHES: s18, s24
	// CHAPTER: ag.06 section 2.1
	s := newSite(t, nil, map[string][3]string{
		"/robots.txt": {"200", "text/plain", "User-agent: *\nDisallow: /private\n"},
		"/":           {"200", html, `<a href="/a">a</a> <a href="b">b</a> <a href="/a#top">a again</a> <a href="/private/x">p</a> <a href="https://other.example/">o</a> <a href="http://docs.example.evil.example/">lookalike</a>`},
		"/a":          {"200", html, `<title>A</title><p>page a</p><a href="/c">c</a>`},
		"/b":          {"200", "text/plain", "page b"},
		"/c":          {"200", "text/markdown", "# C\n\npage c"},
		"/private/x":  {"200", html, "secret"},
	})
	base := "http://docs.example:" + s.port()
	c := source.NewCrawler(source.CrawlerConfig{
		Seeds: []string{base + "/"}, AllowHosts: []string{"docs.example"}, RPS: 1000,
		Resolver: resolver(map[string][]string{"docs.example": {"127.0.0.1"}, "docs.example.evil.example": {"127.0.0.1"}}), AddrAllowed: loopbackOK,
	})
	docs, err := c.Fetch(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, d := range docs {
		got = append(got, strings.TrimPrefix(d.URI, base))
		if d.Source != "web" {
			t.Fatalf("source %q", d.Source)
		}
	}
	if strings.Join(got, " ") != "/ /a /b /c" {
		t.Fatalf("fetched %v, want [/ /a /b /c] in breadth-first order", got)
	}
	reasons := map[string]string{}
	for _, r := range c.Refused() {
		reasons[r.URL] = r.Reason
	}
	if !strings.Contains(reasons[base+"/private/x"], "robots") || !strings.Contains(reasons["https://other.example/"], "not allowed") ||
		!strings.Contains(reasons["http://docs.example.evil.example/"], "not allowed") {
		t.Fatalf("refusals %v", reasons)
	}
	for _, h := range s.hitList() {
		if strings.HasPrefix(h, "/private") {
			t.Fatal("a path disallowed by robots.txt was requested")
		}
	}
}

func TestCrawlerRateLimitPerHost(t *testing.T) {
	// WHY: politeness is per host: at RPS 2, two requests to one host are at
	//      least 500 ms apart (robots.txt counts), measured on a fake clock
	//      that only moves when the crawler sleeps. A second host has its own
	//      slots, so the crawl does not serialize across hosts.
	// KIND: unit
	// CATCHES: s19, s20
	// CHAPTER: ag.06 section 2.1
	clock := &fakeClock{now: time.Unix(1_000_000, 0)}
	pages := func(other string) map[string][3]string {
		return map[string][3]string{
			"/robots.txt": {"404", "text/plain", ""},
			"/":           {"200", html, `<a href="/1">1</a><a href="/2">2</a>` + other},
			"/1":          {"200", "text/plain", "one"},
			"/2":          {"200", "text/plain", "two"},
		}
	}
	a := newSite(t, clock, pages(""))
	b := newSite(t, clock, pages(""))
	c := source.NewCrawler(source.CrawlerConfig{
		Seeds:      []string{"http://a.example:" + a.port() + "/", "http://b.example:" + b.port() + "/"},
		AllowHosts: []string{"a.example", "b.example"}, RPS: 2, Clock: clock,
		Resolver: resolver(map[string][]string{"a.example": {"127.0.0.1"}, "b.example": {"127.0.0.1"}}), AddrAllowed: loopbackOK,
	})
	docs, err := c.Fetch(context.Background())
	if err != nil || len(docs) != 6 {
		t.Fatalf("%d docs, %v; want 6", len(docs), err)
	}
	for name, s := range map[string]*site{"a": a, "b": b} {
		if len(s.at) != 4 {
			t.Fatalf("host %s got %d requests, want 4 (robots.txt and 3 pages)", name, len(s.at))
		}
		for i := 1; i < len(s.at); i++ {
			if gap := s.at[i].Sub(s.at[i-1]); gap < 500*time.Millisecond {
				t.Fatalf("host %s: requests %d and %d are %v apart, want at least 500ms", name, i-1, i, gap)
			}
		}
	}
	// Per host: a at 0, 0.5, 1.0, 1.5 s and b at 0.5, 1.0, 1.5, 2.0 s, so the
	// crawl ends at 2 s; one slot shared by both hosts would take 3.5 s.
	if total := clock.Now().Sub(time.Unix(1_000_000, 0)); total != 2*time.Second {
		t.Fatalf("the crawl took %v of fake time, want 2s (requests to different hosts must not wait for each other)", total)
	}
}

func TestRobots(t *testing.T) {
	// WHY: RFC 9309 rules the crawler must follow: the group naming our
	//      user agent wins over "*"; within a group the longest matching
	//      rule wins (Allow: /docs/public beats Disallow: /docs); "*" and
	//      "$" in patterns; a 404 robots.txt allows everything; a 5xx means
	//      the site cannot say, so nothing is fetched.
	// KIND: unit
	// CATCHES: s21, s22, s23
	// CHAPTER: ag.06 section 2.1
	robots := "User-agent: *\nDisallow: /\n\nUser-agent: tinyllm-rag\nDisallow: /docs\nAllow: /docs/public\nDisallow: /*.pdf$\n"
	s := newSite(t, nil, map[string][3]string{
		"/robots.txt":    {"200", "text/plain", robots},
		"/":              {"200", html, `<a href="/docs/x">x</a><a href="/docs/public/y">y</a><a href="/z.pdf">pdf</a><a href="/z.pdfx">pdfx</a>`},
		"/docs/public/y": {"200", "text/plain", "y"},
		"/z.pdfx":        {"200", "text/plain", "z"},
		"/docs/x":        {"200", "text/plain", "x"},
		"/z.pdf":         {"200", "text/plain", "pdf"},
	})
	res := resolver(map[string][]string{"docs.example": {"127.0.0.1"}})
	base := "http://docs.example:" + s.port()
	c := source.NewCrawler(source.CrawlerConfig{Seeds: []string{base + "/"}, AllowHosts: []string{"docs.example"}, RPS: 1000, Resolver: res, AddrAllowed: loopbackOK})
	docs, _ := c.Fetch(context.Background())
	var got []string
	for _, d := range docs {
		got = append(got, strings.TrimPrefix(d.URI, base))
	}
	if strings.Join(got, " ") != "/ /docs/public/y /z.pdfx" {
		t.Fatalf("fetched %v, want [/ /docs/public/y /z.pdfx]", got)
	}
	for code, want := range map[string]int{"404": 2, "503": 0} {
		s := newSite(t, nil, map[string][3]string{
			"/robots.txt": {code, "text/plain", ""},
			"/":           {"200", html, `<a href="/1">1</a>`},
			"/1":          {"200", "text/plain", "one"},
		})
		c := source.NewCrawler(source.CrawlerConfig{Seeds: []string{"http://docs.example:" + s.port() + "/"}, AllowHosts: []string{"docs.example"}, RPS: 1000, Resolver: res, AddrAllowed: loopbackOK})
		docs, _ := c.Fetch(context.Background())
		if len(docs) != want {
			t.Fatalf("robots.txt %s: %d pages fetched, want %d", code, len(docs), want)
		}
	}
}

func TestSSRF(t *testing.T) {
	// WHY: a crawler fetches URLs that came from documents, which an
	//      attacker can write. With the default address policy it never
	//      connects to a private (10.0.0.5), link-local (169.254.169.254,
	//      the cloud metadata service), or loopback address, even when the
	//      host name is on the allowlist; a name that resolves to one allowed
	//      and one private address is refused; and a redirect is checked
	//      like a new request (host allowlist, then address).
	// KIND: fault
	// CATCHES: s14, s16, s17
	// CHAPTER: ag.06 section 2.5
	local := newSite(t, nil, map[string][3]string{"/robots.txt": {"404", "text/plain", ""}, "/": {"200", "text/plain", "internal"}})
	p := local.port()
	res := resolver(map[string][]string{
		"evil.example":     {"10.0.0.5"},
		"meta.example":     {"169.254.169.254"},
		"mixed.example":    {"127.0.0.1", "10.1.2.3"},
		"loop.example":     {"127.0.0.1"},
		"docs.example":     {"127.0.0.1"},
		"intranet.example": {"10.9.9.9"},
	})
	seeds := []string{"http://evil.example:" + p + "/", "http://meta.example:" + p + "/latest/meta-data/",
		"http://loop.example:" + p + "/", "http://127.0.0.1:" + p + "/", "http://[::ffff:127.0.0.1]:" + p + "/"}
	c := source.NewCrawler(source.CrawlerConfig{Seeds: seeds, AllowHosts: []string{"evil.example", "meta.example", "loop.example", "127.0.0.1", "::ffff:127.0.0.1"},
		RPS: 1000, Resolver: res})
	docs, err := c.Fetch(context.Background())
	if err != nil || len(docs) != 0 {
		t.Fatalf("%d docs, %v; want none", len(docs), err)
	}
	if hits := local.hitList(); len(hits) != 0 {
		t.Fatalf("the loopback server was reached: %v", hits)
	}
	for _, r := range c.Refused() {
		if !strings.Contains(r.Reason, "address not allowed") {
			t.Fatalf("%s refused for %q; want an address refusal", r.URL, r.Reason)
		}
	}

	// A name with one allowed address (the local server, allowed for this
	// crawl) and one private address: refused, so the server is never hit.
	c = source.NewCrawler(source.CrawlerConfig{Seeds: []string{"http://mixed.example:" + p + "/"}, AllowHosts: []string{"mixed.example"},
		RPS: 1000, Resolver: res, AddrAllowed: loopbackOK})
	if docs, _ := c.Fetch(context.Background()); len(docs) != 0 || len(local.hitList()) != 0 {
		t.Fatalf("mixed.example: %d docs, server hits %v; a name with any private address must be refused", len(docs), local.hitList())
	}

	// Redirects from an allowed page: to a host off the allowlist, and to an
	// allowlisted name that resolves to a private address.
	s := newSite(t, nil, map[string][3]string{
		"/robots.txt": {"404", "text/plain", ""},
		"/":           {"200", html, `<a href="/out">out</a><a href="/in">in</a>`},
		"/out":        {"302", "", "http://outside.example:" + p + "/"},
		"/in":         {"302", "", "http://intranet.example:" + p + "/admin"},
	})
	c = source.NewCrawler(source.CrawlerConfig{Seeds: []string{"http://docs.example:" + s.port() + "/"}, AllowHosts: []string{"docs.example", "intranet.example"},
		RPS: 1000, Resolver: res, AddrAllowed: func(a netip.Addr) bool { return a == netip.MustParseAddr("127.0.0.1") || source.PublicAddr(a) }})
	docs, _ = c.Fetch(context.Background())
	if len(docs) != 1 {
		t.Fatalf("%d docs; want only the start page", len(docs))
	}
	reasons := map[string]string{}
	for _, r := range c.Refused() {
		u, _ := url.Parse(r.URL)
		reasons[u.Path] = r.Reason
	}
	if !strings.Contains(reasons["/out"], "redirect") || !strings.Contains(reasons["/in"], "not allowed") {
		t.Fatalf("redirect refusals %v", reasons)
	}
}

func TestPublicAddrTable(t *testing.T) {
	// WHY: the address policy as a table, including the forms that slip
	//      past a string check: IPv4-mapped IPv6 (::ffff:10.0.0.1 is
	//      10.0.0.1), NAT64 (64:ff9b::a00:1 reaches 10.0.0.1), carrier-grade
	//      NAT, 0.0.0.0, and IPv6 link-local and unique-local addresses.
	// KIND: unit
	// CATCHES: s14, s15
	// CHAPTER: ag.06 section 2.5
	for a, want := range map[string]bool{
		"93.184.216.34": true, "2606:4700::1111": true,
		"127.0.0.1": false, "10.0.0.1": false, "172.16.5.4": false, "192.168.1.1": false,
		"169.254.169.254": false, "100.64.0.1": false, "0.0.0.0": false, "0.1.2.3": false, "224.0.0.1": false,
		"::1": false, "fe80::1": false, "fd00::1": false, "::ffff:10.0.0.1": false, "::ffff:169.254.169.254": false,
		"64:ff9b::a00:1": false, "::": false, "::ffff:100.64.0.1": false, "::ffff:0.0.0.7": false,
	} {
		if got := source.PublicAddr(netip.MustParseAddr(a)); got != want {
			t.Errorf("PublicAddr(%s) = %v, want %v", a, got, want)
		}
	}
}
