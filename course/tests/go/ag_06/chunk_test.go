// Course tests for ag.06, RAG ingest: the chunker (token budget, UTF-8
// safety, the learner's tokenizer over /v1/tokenize), the embedder (batches,
// order by index, retries), the store (rag-index.md files, fingerprint
// dedup), text extraction, and the crawler (crawl_test.go: allowlist,
// robots.txt, per-host rate under a fake clock, SSRF).
//
// Embeddings come from a fake /v1/embeddings in this file: a deterministic
// hash of the text, so every run sees the same vectors and no model is
// needed (DEVIATIONS B121-05).
package ag_06

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"tinyllm/agent/provider"
	"tinyllm/agent/rag/chunk"
	"tinyllm/agent/rag/embed"
	"tinyllm/agent/rag/source"
	"tinyllm/agent/rag/store"
)

const (
	p1 = "Paged attention stores KV in blocks."
	s1 = "Blocks are fixed size."
	s2 = "A block table maps token positions to blocks."
	p3 = "Eviction frees whole blocks."
)

func sha(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func TestHandExample(t *testing.T) {
	// WHY: section 3 by hand with the tracer's byte tokenizer and a budget
	//      of 60. Paragraph 2 (68 bytes) is over budget alone, so it is cut
	//      into its sentences (22 and 45). Packing: 36 + 2 + 22 = 60 fits
	//      exactly (the budget is inclusive); adding the next sentence makes
	//      106; 45 + 2 + 28 = 75 does not fit either. Three chunks of 60,
	//      45, and 28 tokens, ids <doc>#0..2, fingerprints the SHA-256 of
	//      each text.
	// KIND: unit
	// CATCHES: s01, s02, s05, s06
	// CHAPTER: ag.06 section 3, worked example
	doc := source.Doc{ID: "docs/kv.md", Text: p1 + "\n\n" + s1 + " " + s2 + "\n\n" + p3}
	cs, err := chunk.TokenChunker{Tok: chunk.Bytes{}, MaxTokens: 60}.Chunk(context.Background(), doc)
	if err != nil {
		t.Fatal(err)
	}
	want := []struct {
		text string
		n    int
	}{{p1 + "\n\n" + s1, 60}, {s2, 45}, {p3, 28}}
	if len(cs) != len(want) {
		t.Fatalf("%d chunks: %+v", len(cs), cs)
	}
	for i, w := range want {
		c := cs[i]
		if c.Text != w.text || c.Tokens != w.n || c.Index != i || c.ID != fmt.Sprintf("docs/kv.md#%d", i) || c.DocID != "docs/kv.md" || c.Fingerprint != sha(w.text) {
			t.Errorf("chunk %d = %+v\nwant text %q, %d tokens", i, c, w.text, w.n)
		}
	}
}

func TestSplitOrder(t *testing.T) {
	// WHY: when a paragraph is over budget the chunker cuts at the largest
	//      unit that fits: sentences before words, words before characters,
	//      so a chunk never ends in the middle of a sentence it could have
	//      kept whole.
	// KIND: unit
	// CATCHES: s04
	// CHAPTER: ag.06 section 2.2
	for _, tc := range []struct {
		text string
		max  int
		want []string
	}{
		{"One two three. Four five six seven.", 20, []string{"One two three.", "Four five six seven."}},
		{"alpha beta gamma delta", 11, []string{"alpha beta", "gamma delta"}},
		{"abcdefghij", 4, []string{"abcd", "efgh", "ij"}},
	} {
		cs, err := chunk.TokenChunker{Tok: chunk.Bytes{}, MaxTokens: tc.max}.Chunk(context.Background(), source.Doc{ID: "d", Text: tc.text})
		if err != nil {
			t.Fatal(err)
		}
		var got []string
		for _, c := range cs {
			got = append(got, c.Text)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%q at %d: %q, want %q", tc.text, tc.max, got, tc.want)
		}
	}
}

// sepTok counts words, plus 2 tokens for every paragraph break: the count
// of a joined text is more than the sum of its parts, as with real
// tokenizers that spend tokens on separators.
type sepTok struct {
	mu    sync.Mutex
	calls int
}

func (s *sepTok) Count(_ context.Context, text string) (int, error) {
	s.mu.Lock()
	s.calls++
	s.mu.Unlock()
	return len(strings.Fields(text)) + 2*strings.Count(text, "\n\n"), nil
}

func corpusDoc(i int) source.Doc {
	words := strings.Fields("kv cache blocks hold keys and values for every token the decoder has seen so far and the scheduler frees them when a request ends")
	var paras []string
	for p := 0; p < 3+i%4; p++ {
		var b strings.Builder
		for w := 0; w < 5+(i*7+p*3)%23; w++ {
			if w > 0 {
				b.WriteString(" ")
			}
			b.WriteString(words[(i+p+w)%len(words)])
			if (w+p)%9 == 8 {
				b.WriteString(".")
			}
		}
		paras = append(paras, b.String())
	}
	return source.Doc{ID: fmt.Sprintf("d%d", i), Text: strings.Join(paras, "\n\n")}
}

func TestChunkBudgetNeverExceeded(t *testing.T) {
	// WHY: the budget is a hard limit for whatever the tokenizer says, and
	//      token counts do not add: with a tokenizer that charges for
	//      separators, a chunker that sums piece counts overflows. Over 40
	//      generated documents and budgets 4 to 15, every chunk counts at most
	//      the budget by the same tokenizer, none is empty, and the words
	//      come out in order with none lost or repeated.
	// KIND: property
	// CATCHES: s01
	// CHAPTER: ag.06 section 2.2
	tok := &sepTok{}
	ctx := context.Background()
	for i := 0; i < 40; i++ {
		d := corpusDoc(i)
		budget := 4 + i%12
		cs, err := chunk.TokenChunker{Tok: tok, MaxTokens: budget}.Chunk(ctx, d)
		if err != nil {
			t.Fatal(err)
		}
		var got []string
		for _, c := range cs {
			n, _ := tok.Count(ctx, c.Text)
			if n > budget || strings.TrimSpace(c.Text) == "" || c.Tokens != n {
				t.Fatalf("doc %d budget %d: chunk %q counts %d (reported %d)", i, budget, c.Text, n, c.Tokens)
			}
			got = append(got, strings.Fields(c.Text)...)
		}
		if !reflect.DeepEqual(got, strings.Fields(d.Text)) {
			t.Fatalf("doc %d: words changed\n got %v\nwant %v", i, got, strings.Fields(d.Text))
		}
	}
}

func TestHardSplitUTF8(t *testing.T) {
	// WHY: a "word" longer than the budget (a URL, a hash, a run of é) must
	//      be cut, and never inside a character: a cut between the two bytes
	//      of é is invalid UTF-8 that the engine's tokenizer rejects or
	//      mangles.
	// KIND: boundary
	// CATCHES: s03
	// CHAPTER: ag.06 section 2.2
	word := strings.Repeat("é", 30) // 60 bytes
	cs, err := chunk.TokenChunker{Tok: chunk.Bytes{}, MaxTokens: 7}.Chunk(context.Background(), source.Doc{ID: "w", Text: word})
	if err != nil {
		t.Fatal(err)
	}
	var joined strings.Builder
	for _, c := range cs {
		if !utf8.ValidString(c.Text) || len(c.Text) > 7 || len(c.Text) != 6 {
			t.Fatalf("piece %q (%d bytes): want valid UTF-8 of exactly 6 bytes", c.Text, len(c.Text))
		}
		joined.WriteString(c.Text)
	}
	if joined.String() != word {
		t.Fatal("the pieces do not join back to the word")
	}
}

// engine is a fake engine: /v1/tokenize (bytes) and /v1/embeddings
// (hashed vectors, dim 8), answering embeddings in reverse order.
type engine struct {
	mu       sync.Mutex
	inputs   [][]string
	failNext int
	dimFor   func(text string) int
	extra    bool
}

func vecOf(text string, dim int) []float32 {
	sum := sha256.Sum256([]byte(text))
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(int(sum[i%32])-128) / 64
	}
	return v
}

func (e *engine) serve(t *testing.T) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string   `json:"model"`
			Text  string   `json:"text"`
			Input []string `json:"input"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/tokenize": // one id per character: not the byte count
			ids := []int{}
			for _, r := range body.Text {
				ids = append(ids, int(r))
			}
			json.NewEncoder(w).Encode(map[string]any{"ids": ids})
		case "/v1/embeddings":
			e.mu.Lock()
			if e.failNext > 0 {
				e.failNext--
				e.mu.Unlock()
				w.WriteHeader(503)
				w.Write([]byte(`{"error":{"message":"busy","type":"server_error","param":null,"code":"no_capacity"}}`))
				return
			}
			e.inputs = append(e.inputs, body.Input)
			e.mu.Unlock()
			var data []map[string]any
			for i := len(body.Input) - 1; i >= 0; i-- {
				dim := 8
				if e.dimFor != nil {
					dim = e.dimFor(body.Input[i])
				}
				data = append(data, map[string]any{"object": "embedding", "index": i, "embedding": vecOf(body.Input[i], dim)})
			}
			if e.extra {
				data = append(data, data[0])
			}
			json.NewEncoder(w).Encode(map[string]any{"object": "list", "model": body.Model, "data": data,
				"usage": map[string]int{"prompt_tokens": 1, "total_tokens": 1}})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func (e *engine) texts() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	n := 0
	for _, in := range e.inputs {
		n += len(in)
	}
	return n
}

func TestHTTPTokenizer(t *testing.T) {
	// WHY: the chunk budget must be counted by the tokenizer of the model
	//      that embeds the chunks, which lives in the engine: the count is
	//      the length of /v1/tokenize's ids, whatever the text's length in
	//      bytes (this fake engine gives one id per character).
	// KIND: conformance
	// CATCHES: s26
	// CHAPTER: ag.06 section 4
	srv := (&engine{}).serve(t)
	n, err := chunk.HTTPTokenizer{BaseURL: srv.URL + "/v1", Model: "smol"}.Count(context.Background(), "héllo")
	if err != nil || n != 5 {
		t.Fatalf("Count = %d, %v; want 5 (the engine's ids for héllo)", n, err)
	}
}

type sleeps struct{ d []time.Duration }

func (s *sleeps) Sleep(_ context.Context, d time.Duration) error { s.d = append(s.d, d); return nil }

func TestEmbedderBatchesAndOrder(t *testing.T) {
	// WHY: five texts with BatchSize 2 make three requests; the fake answers
	//      each batch in reverse, so vectors must be placed by their index
	//      field, not by position. A wrong count or a vector of another
	//      dimension is an error, never a silently misaligned index.
	// KIND: unit
	// CATCHES: s11
	// CHAPTER: ag.06 section 2.3
	e := &engine{}
	srv := e.serve(t)
	em := embed.HTTPEmbedder{BaseURL: srv.URL + "/v1", Model: "smol", BatchSize: 2}
	texts := []string{"a", "b", "c", "d", "e"}
	vecs, err := em.Embed(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	for i, txt := range texts {
		if !reflect.DeepEqual(vecs[i], vecOf(txt, 8)) {
			t.Fatalf("vector %d is not the embedding of %q", i, txt)
		}
	}
	if len(e.inputs) != 3 || len(e.inputs[0]) != 2 || len(e.inputs[2]) != 1 {
		t.Fatalf("batches %v, want [2 2 1]", e.inputs)
	}
	bad := &engine{extra: true}
	if _, err := (embed.HTTPEmbedder{BaseURL: bad.serve(t).URL + "/v1", Model: "m"}).Embed(context.Background(), []string{"x", "y"}); err == nil {
		t.Fatal("three embeddings for two inputs must be an error")
	}
	mixed := &engine{dimFor: func(s string) int { return 8 + len(s)%2 }}
	if _, err := (embed.HTTPEmbedder{BaseURL: mixed.serve(t).URL + "/v1", Model: "m", BatchSize: 1}).Embed(context.Background(), []string{"xx", "y"}); err == nil {
		t.Fatal("vectors of different dimensions must be an error")
	}
}

func TestEmbedderRetry(t *testing.T) {
	// WHY: an engine under load answers 503; the embedder uses the
	//      provider's policy (ag.01): wait Initial, try again, succeed.
	// KIND: fault
	// CATCHES: s13
	// CHAPTER: ag.06 section 2.3
	e := &engine{failNext: 1}
	srv := e.serve(t)
	s := &sleeps{}
	em := embed.HTTPEmbedder{BaseURL: srv.URL + "/v1", Model: "m", Retry: provider.RetryPolicy{Initial: 250 * time.Millisecond, Sleep: s.Sleep}}
	if _, err := em.Embed(context.Background(), []string{"x"}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(s.d, []time.Duration{250 * time.Millisecond}) {
		t.Fatalf("sleeps %v, want [250ms]", s.d)
	}
}

func writeDocs(t *testing.T, root string, files map[string]string) {
	t.Helper()
	for name, body := range files {
		p := filepath.Join(root, name)
		os.MkdirAll(filepath.Dir(p), 0o755)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func snapshot(t *testing.T, dir string) map[string]string {
	out := map[string]string{}
	for _, f := range []string{"meta.json", "docs.jsonl", "vectors.f32"} {
		b, err := os.ReadFile(filepath.Join(dir, f))
		if err != nil {
			t.Fatal(err)
		}
		out[f] = string(b)
	}
	return out
}

func TestIngestDedupNoOp(t *testing.T) {
	// WHY: re-ingesting an unchanged corpus is the common case (a nightly
	//      job): it must embed nothing and write nothing. Changing one
	//      paragraph re-embeds only the chunks whose fingerprints are new;
	//      the rest keep their stored vectors.
	// KIND: unit
	// CATCHES: s07, s08, s09
	// CHAPTER: ag.06 section 2.4
	root := filepath.Join(t.TempDir(), "docs")
	writeDocs(t, root, map[string]string{
		"kv.md":    "# KV cache\n\n" + p1 + "\n\n" + s1 + " " + s2 + "\n\n" + p3,
		"sched.md": "# Scheduler\n\nThe scheduler admits requests while blocks are free.\n\nIt preempts the newest request when the pool runs out.",
	})
	e := &engine{}
	srv := e.serve(t)
	ctx := context.Background()
	idxDir := filepath.Join(t.TempDir(), "rag", "docs")
	ch := chunk.TokenChunker{Tok: chunk.Bytes{}, MaxTokens: 60}
	em := embed.HTTPEmbedder{BaseURL: srv.URL + "/v1", Model: "smol"}
	x, _ := store.Open(idxDir, store.Meta{EmbeddingModel: "smol", Tokenizer: "bytes"})
	st, err := store.Ingest(ctx, source.DirSource{Root: root}, ch, em, x)
	if err != nil || st.Docs != 2 || st.Unchanged != 0 || st.Embedded != st.Chunks || st.Chunks == 0 {
		t.Fatalf("first ingest: %+v, %v", st, err)
	}
	if wrote, err := x.Save(time.Unix(0, 0)); !wrote || err != nil {
		t.Fatalf("first Save: %v, %v", wrote, err)
	}
	before, sent := snapshot(t, idxDir), e.texts()

	x, _ = store.Open(idxDir, store.Meta{})
	st, err = store.Ingest(ctx, source.DirSource{Root: root}, ch, em, x)
	if err != nil || st.Unchanged != 2 || st.Embedded != 0 || e.texts() != sent {
		t.Fatalf("unchanged re-ingest: %+v, %v, %d texts embedded", st, err, e.texts()-sent)
	}
	if wrote, _ := x.Save(time.Unix(99, 0)); wrote || !reflect.DeepEqual(snapshot(t, idxDir), before) {
		t.Fatal("an unchanged re-ingest must not write the index")
	}

	writeDocs(t, root, map[string]string{"kv.md": "# KV cache\n\n" + p1 + "\n\n" + s1 + " " + s2 + "\n\nEviction frees whole blocks at once."})
	st, err = store.Ingest(ctx, source.DirSource{Root: root}, ch, em, x)
	if err != nil || st.Unchanged != 1 || st.Embedded != 1 || e.texts() != sent+1 {
		t.Fatalf("one changed paragraph: %+v, %v; want 1 unchanged doc and 1 text embedded", st, err)
	}
}

func TestIndexFormat(t *testing.T) {
	// WHY: formats/rag-index.md is what retrieval (ag.07) and citations
	//      (ag.08) read: meta.json counts, one docs.jsonl line per chunk in
	//      order with chunk_id <doc_id>#<k> and fingerprint = SHA-256 of the
	//      text, vectors.f32 exactly n x dim little-endian float32 with every
	//      row of length 1; Open reads back the same chunks.
	// KIND: conformance
	// CATCHES: s06, s10
	// CHAPTER: ag.06 section 2.4
	root := filepath.Join(t.TempDir(), "docs")
	writeDocs(t, root, map[string]string{"a.md": p1 + "\n\n" + p3, "b.txt": s1})
	srv := (&engine{}).serve(t)
	idxDir := filepath.Join(t.TempDir(), "idx")
	x, _ := store.Open(idxDir, store.Meta{IndexID: "docs", EmbeddingModel: "smol", Tokenizer: "bytes"})
	if _, err := store.Ingest(context.Background(), source.DirSource{Root: root, Exts: []string{".md", ".txt"}},
		chunk.TokenChunker{Tok: chunk.Bytes{}, MaxTokens: 40}, embed.HTTPEmbedder{BaseURL: srv.URL + "/v1", Model: "smol"}, x); err != nil {
		t.Fatal(err)
	}
	if _, err := x.Save(time.Date(2026, 10, 9, 12, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	}
	var meta map[string]any
	b, _ := os.ReadFile(filepath.Join(idxDir, "meta.json"))
	json.Unmarshal(b, &meta)
	if meta["index_id"] != "docs" || meta["n_chunks"] != 3.0 || meta["dim"] != 8.0 || meta["created_at"] != "2026-10-09T12:00:00Z" || meta["embedding_model"] != "smol" {
		t.Fatalf("meta.json %v", meta)
	}
	lines := strings.Split(strings.TrimSpace(string(must(os.ReadFile(filepath.Join(idxDir, "docs.jsonl"))))), "\n")
	wantIDs := []string{root + "/a.md#0", root + "/a.md#1", root + "/b.txt#0"}
	for i, l := range lines {
		var d map[string]any
		json.Unmarshal([]byte(l), &d)
		if d["chunk_id"] != filepath.ToSlash(wantIDs[i]) || d["fingerprint"] != sha(d["text"].(string)) || d["source"] != "file" {
			t.Fatalf("docs.jsonl line %d: %v", i, d)
		}
	}
	vb := must(os.ReadFile(filepath.Join(idxDir, "vectors.f32")))
	if len(vb) != 3*8*4 {
		t.Fatalf("vectors.f32 is %d bytes, want %d", len(vb), 3*8*4)
	}
	for r := 0; r < 3; r++ {
		var ss float64
		for j := 0; j < 8; j++ {
			f := math.Float32frombits(binary.LittleEndian.Uint32(vb[(r*8+j)*4:]))
			ss += float64(f) * float64(f)
		}
		if math.Abs(ss-1) > 1e-6 {
			t.Fatalf("row %d has squared norm %v, want 1", r, ss)
		}
	}
	y, err := store.Open(idxDir, store.Meta{})
	if err != nil || len(y.Entries()) != 3 || !reflect.DeepEqual(y.Entries(), x.Entries()) {
		t.Fatalf("Open read back %d entries, %v", len(y.Entries()), err)
	}
}

func must(b []byte, err error) []byte {
	if err != nil {
		panic(err)
	}
	return b
}

func TestToDoc(t *testing.T) {
	// WHY: an embedding of "<div class=nav>function track()..." is an
	//      embedding of markup. Script and style vanish, block tags become
	//      paragraph breaks (which the chunker splits on), entities decode,
	//      the title comes from <title> (or a markdown "# " heading), and
	//      the doc id drops the URL fragment.
	// KIND: unit
	// CATCHES: s25
	// CHAPTER: ag.06 section 2.1
	html := `<html><head><title>KV &amp; blocks</title><style>p{color:red}</style></head>
<body><script>track("x")</script><h1>KV cache</h1><p>Blocks are   fixed&nbsp;size.</p><p>Eviction &lt;frees&gt; them.</p></body></html>`
	d := source.ToDoc(source.RawDoc{URI: "https://Docs.Example/kv#top", Source: "web", ContentType: "text/html; charset=utf-8", Body: []byte(html)})
	if d.Title != "KV & blocks" || d.ID != "https://docs.example/kv" {
		t.Fatalf("title %q id %q", d.Title, d.ID)
	}
	if want := "KV cache\n\nBlocks are fixed size.\n\nEviction <frees> them."; d.Text != want {
		t.Fatalf("text %q\nwant %q", d.Text, want)
	}
	m := source.ToDoc(source.RawDoc{URI: "docs/a.md", Source: "file", ContentType: "text/markdown", Body: []byte("# Title\r\n\r\nBody text.\r\n")})
	if m.Title != "Title" || m.Text != "# Title\n\nBody text." || m.ID != "docs/a.md" {
		t.Fatalf("markdown doc %+v", m)
	}
}
